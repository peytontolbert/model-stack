from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from specs.config import ModelConfig
from tensor.io_safetensors import safetensor_dump, safetensor_load


def save_pretrained(model: torch.nn.Module, cfg: ModelConfig, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    safetensor_dump(weights, os.path.join(outdir, "model.safetensors"))
    return outdir


def load_pretrained(model: torch.nn.Module, indir: str, *, strict: bool = True) -> torch.nn.Module:
    tensors = safetensor_load(os.path.join(indir, "model.safetensors"))
    model.load_state_dict(tensors, strict=strict)
    return model


def load_config(indir: str) -> ModelConfig:
    with open(os.path.join(indir, "config.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelConfig(**data)


def ensure_snapshot(model_id: str, cache_dir: str) -> str:
    if os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "config.json")):
        return model_id
    try:
        org_name = model_id.strip().split("/")[-2:]
        if len(org_name) == 2:
            org, name = org_name
            snapshot_root = os.path.join(cache_dir, f"models--{org}--{name}", "snapshots")
            candidates: list[str] = []
            if os.path.isdir(snapshot_root):
                candidates.extend([os.path.join(snapshot_root, d) for d in os.listdir(snapshot_root)])
            else:
                for d in os.listdir(cache_dir):
                    if not d.startswith("models--"):
                        continue
                    snap = os.path.join(cache_dir, d, "snapshots")
                    if os.path.isdir(snap):
                        for sd in os.listdir(snap):
                            candidates.append(os.path.join(snap, sd))
            candidates = [p for p in candidates if os.path.isfile(os.path.join(p, "config.json"))]
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return candidates[0]
    except Exception:
        pass
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download the model snapshot") from exc
    return snapshot_download(repo_id=model_id, cache_dir=cache_dir)


def _index_hf_safetensor_shards(model_dir: str) -> Dict[str, List[str]]:
    index_fp = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_fp):
        raise FileNotFoundError(f"Missing index file: {index_fp}")
    with open(index_fp, "r", encoding="utf-8") as fh:
        index_obj = json.load(fh)
    weight_map: Dict[str, str] = index_obj.get("weight_map", {})
    by_file: Dict[str, List[str]] = {}
    for key, filename in weight_map.items():
        by_file.setdefault(filename, []).append(key)
    return by_file


def _stack_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape[1] != up.shape[1]:
        raise ValueError(f"Gate/Up in_features mismatch: {gate.shape} vs {up.shape}")
    return torch.cat([gate, up], dim=0).contiguous()


def _stack_gate_up_bias(gate_b: torch.Tensor, up_b: torch.Tensor) -> torch.Tensor:
    if gate_b.dim() != 1 or up_b.dim() != 1:
        raise ValueError("Gate/Up biases must be 1D tensors")
    if gate_b.shape[0] != up_b.shape[0]:
        raise ValueError(f"Gate/Up bias size mismatch: {gate_b.shape} vs {up_b.shape}")
    return torch.cat([gate_b, up_b], dim=0).contiguous()


def _assert_shape(name: str, got: Tuple[int, ...], expected: Tuple[int, ...]) -> None:
    if tuple(got) != tuple(expected):
        raise ValueError(f"Shape mismatch for {name}: got {tuple(got)} expected {tuple(expected)}")


def load_hf_llama_weights_into_local(model: torch.nn.Module, model_dir: str) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    num_layers = len(model.blocks)
    d_model = int(getattr(model.cfg, "d_model"))
    n_heads = int(getattr(model.cfg, "n_heads"))
    head_dim = d_model // n_heads
    try:
        n_kv_heads = int(model.blocks[0].attn.n_kv_heads)
    except Exception:
        n_kv_heads = n_heads

    pending_in: Dict[int, Dict[str, torch.Tensor]] = {}

    try:
        from safetensors import safe_open  # type: ignore
    except Exception as exc:
        raise RuntimeError("safetensors is required to load HF checkpoints") from exc

    by_file = _index_hf_safetensor_shards(model_dir)

    def _assign(key: str, tensor: torch.Tensor) -> None:
        nonlocal pending_in
        if key == "model.embed_tokens.weight":
            model.embed.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if key == "model.norm.weight" and hasattr(model, "norm") and hasattr(model.norm, "weight"):
            model.norm.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if key == "lm_head.weight" and hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            model.lm_head.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if not key.startswith("model.layers."):
            return
        try:
            rest = key[len("model.layers.") :]
            li_str, sub = rest.split(".", 1)
            li = int(li_str)
        except Exception:
            return
        if li < 0 or li >= num_layers:
            return
        blk = model.blocks[li]
        if sub == "self_attn.q_proj.weight":
            _assert_shape(f"layers.{li}.q_proj", tuple(tensor.shape), (n_heads * head_dim, d_model))
            blk.attn.w_q.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.q_proj.bias" and getattr(blk.attn.w_q, "bias", None) is not None:
            blk.attn.w_q.bias.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.k_proj.weight":
            _assert_shape(f"layers.{li}.k_proj", tuple(tensor.shape), (n_kv_heads * head_dim, d_model))
            blk.attn.w_k.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.k_proj.bias" and getattr(blk.attn.w_k, "bias", None) is not None:
            blk.attn.w_k.bias.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.v_proj.weight":
            _assert_shape(f"layers.{li}.v_proj", tuple(tensor.shape), (n_kv_heads * head_dim, d_model))
            blk.attn.w_v.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.v_proj.bias" and getattr(blk.attn.w_v, "bias", None) is not None:
            blk.attn.w_v.bias.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.o_proj.weight":
            _assert_shape(f"layers.{li}.o_proj", tuple(tensor.shape), (d_model, n_heads * head_dim))
            blk.attn.w_o.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.o_proj.bias" and getattr(blk.attn.w_o, "bias", None) is not None:
            blk.attn.w_o.bias.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "input_layernorm.weight":
            _assert_shape(f"layers.{li}.n1", tuple(tensor.shape), (d_model,))
            blk.n1.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "post_attention_layernorm.weight":
            _assert_shape(f"layers.{li}.n2", tuple(tensor.shape), (d_model,))
            blk.n2.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "mlp.down_proj.weight":
            _assert_shape(
                f"layers.{li}.mlp.w_out",
                tuple(tensor.shape),
                (blk.mlp.w_out.weight.shape[0], blk.mlp.w_out.weight.shape[1]),
            )
            blk.mlp.w_out.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "mlp.down_proj.bias" and getattr(blk.mlp.w_out, "bias", None) is not None:
            blk.mlp.w_out.bias.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "mlp.gate_proj.weight":
            pending_in.setdefault(li, {})["gate"] = tensor
            if "up" in pending_in[li]:
                gate = pending_in[li].pop("gate")
                up = pending_in[li].pop("up")
                w_in = _stack_gate_up(gate, up)
                _assert_shape(
                    f"layers.{li}.mlp.w_in",
                    tuple(w_in.shape),
                    (blk.mlp.w_in.weight.shape[0], blk.mlp.w_in.weight.shape[1]),
                )
                blk.mlp.w_in.weight.data.copy_(w_in.to(device=device, dtype=dtype))
                pending_in.pop(li, None)
            return
        if sub == "mlp.gate_proj.bias" and getattr(blk.mlp.w_in, "bias", None) is not None:
            pending_in.setdefault(li, {})["gate_b"] = tensor
            if "up_b" in pending_in[li]:
                gb = pending_in[li].pop("gate_b")
                ub = pending_in[li].pop("up_b")
                b_in = _stack_gate_up_bias(gb, ub)
                blk.mlp.w_in.bias.data.copy_(b_in.to(device=device, dtype=dtype))
                pending_in.pop(li, None)
            return
        if sub == "mlp.up_proj.weight":
            pending_in.setdefault(li, {})["up"] = tensor
            if "gate" in pending_in[li]:
                gate = pending_in[li].pop("gate")
                up = pending_in[li].pop("up")
                w_in = _stack_gate_up(gate, up)
                _assert_shape(
                    f"layers.{li}.mlp.w_in",
                    tuple(w_in.shape),
                    (blk.mlp.w_in.weight.shape[0], blk.mlp.w_in.weight.shape[1]),
                )
                blk.mlp.w_in.weight.data.copy_(w_in.to(device=device, dtype=dtype))
                pending_in.pop(li, None)
            return
        if sub == "mlp.up_proj.bias" and getattr(blk.mlp.w_in, "bias", None) is not None:
            pending_in.setdefault(li, {})["up_b"] = tensor
            if "gate_b" in pending_in[li]:
                gb = pending_in[li].pop("gate_b")
                ub = pending_in[li].pop("up_b")
                b_in = _stack_gate_up_bias(gb, ub)
                blk.mlp.w_in.bias.data.copy_(b_in.to(device=device, dtype=dtype))
                pending_in.pop(li, None)
            return

    for shard_rel, keys in by_file.items():
        shard_fp = os.path.join(model_dir, shard_rel)
        with safe_open(shard_fp, framework="pt", device="cpu") as f:  # type: ignore
            for key in keys:
                try:
                    if not f.keys() or key not in f.keys():  # type: ignore[attr-defined]
                        continue
                except Exception:
                    pass
                try:
                    tensor = f.get_tensor(key)  # type: ignore[attr-defined]
                except Exception:
                    continue
                _assign(key, tensor)
                del tensor

    for li, parts in list(pending_in.items()):
        if "gate" in parts and "up" in parts:
            blk = model.blocks[li]
            w_in = _stack_gate_up(parts["gate"], parts["up"])
            _assert_shape(
                f"layers.{li}.mlp.w_in",
                tuple(w_in.shape),
                (blk.mlp.w_in.weight.shape[0], blk.mlp.w_in.weight.shape[1]),
            )
            blk.mlp.w_in.weight.data.copy_(w_in.to(device=device, dtype=dtype))
        pending_in.pop(li, None)

    try:
        if model.lm_head.weight.data.data_ptr() == model.embed.weight.data.data_ptr():
            model.lm_head.weight.data.copy_(model.embed.weight.data)
    except Exception:
        pass


def model_config_from_hf_llama_snapshot_config(
    cfg_obj: dict[str, object],
    *,
    torch_dtype: torch.dtype,
) -> tuple[ModelConfig, int, bool]:
    d_model = int(cfg_obj.get("hidden_size"))
    n_layers = int(cfg_obj.get("num_hidden_layers"))
    n_heads = int(cfg_obj.get("num_attention_heads"))
    d_ff = int(cfg_obj.get("intermediate_size"))
    vocab_size = int(cfg_obj.get("vocab_size"))
    head_dim = int(cfg_obj.get("head_dim", d_model // n_heads))
    n_kv_heads = int(cfg_obj.get("num_key_value_heads", n_heads))
    rope_theta = float(cfg_obj.get("rope_theta", 1e6))
    try:
        rope_parameters = cfg_obj.get("rope_parameters", None)
        if isinstance(rope_parameters, dict) and rope_parameters.get("rope_theta") is not None:
            rope_theta = float(rope_parameters.get("rope_theta"))
    except Exception:
        pass
    rms_eps = float(cfg_obj.get("rms_norm_eps", 1e-6))
    cfg = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        head_dim=head_dim,
        rope_theta=rope_theta,
        dtype=("bfloat16" if torch_dtype == torch.bfloat16 else "float32"),
        attn_impl="sdpa",
        rms_norm_eps=rms_eps,
    )
    return cfg, n_kv_heads, bool(cfg_obj.get("tie_word_embeddings", True))


def model_config_from_hf_llama_transformers_config(
    cfg_hf,
    *,
    dtype: torch.dtype,
    seq_len: int | None = None,
) -> tuple[ModelConfig, int, bool]:
    d_model = int(getattr(cfg_hf, "hidden_size"))
    n_layers = int(getattr(cfg_hf, "num_hidden_layers"))
    n_heads = int(getattr(cfg_hf, "num_attention_heads"))
    head_dim = int(getattr(cfg_hf, "head_dim", max(1, d_model // max(1, n_heads))))
    n_kv_heads = int(getattr(cfg_hf, "num_key_value_heads", n_heads))
    d_ff = int(getattr(cfg_hf, "intermediate_size"))
    vocab_size = int(getattr(cfg_hf, "vocab_size"))
    rms_eps = float(getattr(cfg_hf, "rms_norm_eps", 1e-6))
    rope_theta = None
    try:
        rope_parameters = getattr(cfg_hf, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            rope_theta = float(rope_parameters["rope_theta"])
    except Exception:
        rope_theta = None
    rope_scaling = getattr(cfg_hf, "rope_scaling", None)
    rs_type = None
    rs_factor = None
    rs_orig = None
    rs_low = None
    rs_high = None
    try:
        if isinstance(rope_scaling, dict):
            rs_type = rope_scaling.get("type")
            if rope_scaling.get("factor") is not None:
                rs_factor = float(rope_scaling["factor"])
            if rope_scaling.get("original_max_position_embeddings") is not None:
                rs_orig = int(rope_scaling["original_max_position_embeddings"])
            if rope_scaling.get("low_freq_factor") is not None:
                rs_low = float(rope_scaling["low_freq_factor"])
            if rope_scaling.get("high_freq_factor") is not None:
                rs_high = float(rope_scaling["high_freq_factor"])
    except Exception:
        pass
    cfg = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        attn_impl="sdpa",
        rope_theta=float(rope_theta if rope_theta is not None else getattr(cfg_hf, "rope_theta", 1e6)),
        max_position_embeddings=int(
            getattr(
                cfg_hf,
                "max_position_embeddings",
                int(seq_len if seq_len is not None else 2048),
            )
        ),
        dtype=("bfloat16" if dtype == torch.bfloat16 else ("float16" if dtype == torch.float16 else "float32")),
        rms_norm_eps=rms_eps,
        head_dim=head_dim,
        rope_scaling_type=rs_type,
        rope_scaling_factor=rs_factor,
        rope_scaling_original_max_position_embeddings=rs_orig,
        rope_scaling_low_freq_factor=rs_low,
        rope_scaling_high_freq_factor=rs_high,
    )
    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding  # type: ignore

        tmp_cfg = type("_TmpCfg", (), {})()
        tmp_cfg.hidden_size = d_model
        tmp_cfg.num_attention_heads = n_heads
        tmp_cfg.head_dim = head_dim
        tmp_cfg.max_position_embeddings = int(
            getattr(
                cfg_hf,
                "max_position_embeddings",
                int(seq_len if seq_len is not None else 2048),
            )
        )
        tmp_cfg.rope_parameters = {
            "rope_type": getattr(cfg_hf, "rope_type", "default") if hasattr(cfg_hf, "rope_type") else "default",
            "rope_theta": float(rope_theta if rope_theta is not None else getattr(cfg_hf, "rope_theta", 1e6)),
        }

        emb = LlamaRotaryEmbedding(config=tmp_cfg)
        setattr(cfg, "rope_attention_scaling", float(getattr(emb, "attention_scaling", 1.0)))
    except Exception:
        pass
    return cfg, n_kv_heads, bool(getattr(cfg_hf, "tie_word_embeddings", True))


def build_local_llama_from_hf_config(
    model_id_or_dir: str,
    dtype: torch.dtype,
    *,
    seq_len: int | None = None,
) -> torch.nn.Module:
    from transformers import AutoConfig  # type: ignore
    from runtime.factory import build_causal_lm

    cfg_hf = AutoConfig.from_pretrained(model_id_or_dir)
    cfg, n_kv_heads, tie_weights = model_config_from_hf_llama_transformers_config(
        cfg_hf,
        dtype=dtype,
        seq_len=seq_len,
    )
    return build_causal_lm(cfg, block="llama", n_kv_heads=n_kv_heads, tie_weights=tie_weights)


def build_local_llama_from_snapshot(
    ckpt_dir: str,
    device: str,
    torch_dtype,
    device_map: Optional[str] = None,
    gpu_ids: Optional[str] = None,
) -> Any:
    del gpu_ids
    from runtime.factory import build_causal_lm

    with open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg_obj = json.load(f)
    cfg, n_kv_heads, tie_weights = model_config_from_hf_llama_snapshot_config(
        cfg_obj,
        torch_dtype=torch_dtype,
    )
    try:
        from dist.engine import DistributedEngine  # type: ignore
        from specs.dist import DistConfig  # type: ignore

        has_dist = True
    except Exception:
        DistributedEngine = None  # type: ignore
        DistConfig = None  # type: ignore
        has_dist = False

    model = build_causal_lm(
        cfg,
        block="llama",
        n_kv_heads=n_kv_heads,
        tie_weights=tie_weights,
    )
    try:
        model = model.to(dtype=torch_dtype)
    except Exception:
        pass
    load_hf_llama_weights_into_local(model, ckpt_dir)

    target_device = device
    try:
        if str(device).startswith("cuda") and (device_map == "auto") and torch.cuda.is_available():
            num = torch.cuda.device_count()
            if num > 0:
                best_idx = 0
                best_free = -1
                for i in range(num):
                    try:
                        free, _total = torch.cuda.mem_get_info(i)  # type: ignore[arg-type]
                    except Exception:
                        try:
                            torch.cuda.set_device(i)
                            free, _total = torch.cuda.mem_get_info()
                        except Exception:
                            continue
                    if int(free) > int(best_free):
                        best_free = int(free)
                        best_idx = i
                try:
                    torch.cuda.set_device(best_idx)
                except Exception:
                    pass
                target_device = f"cuda:{best_idx}"
    except Exception:
        target_device = device

    model = model.to(target_device).eval()

    try:
        if has_dist:
            world = int(os.environ.get("WORLD_SIZE", "1"))
            if world > 1:
                cfg_dist = DistConfig(  # type: ignore[misc]
                    backend="nccl",
                    strategy="DDP",
                    precision=("bf16" if torch_dtype == torch.bfloat16 else "fp32"),
                )
                eng = DistributedEngine(cfg_dist)  # type: ignore[misc]
                eng.init()
                model = eng.wrap_model(model)
    except Exception:
        pass
    return model, cfg_obj
