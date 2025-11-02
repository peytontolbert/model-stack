import json
import os
from typing import Dict, Tuple, List

import torch


def _index_shards(model_dir: str) -> Dict[str, List[str]]:
    """Return mapping shard_filename -> list of weight keys (streaming-friendly)."""
    index_fp = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_fp):
        raise FileNotFoundError(f"Missing index file: {index_fp}")
    with open(index_fp, "r", encoding="utf-8") as fh:
        index_obj = json.load(fh)
    weight_map: Dict[str, str] = index_obj.get("weight_map", {})
    by_file: Dict[str, List[str]] = {}
    for k, fn in weight_map.items():
        by_file.setdefault(fn, []).append(k)
    return by_file


def _stack_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Concatenate gate and up projections along out_features (rows): [gate; up]."""
    if gate.shape[1] != up.shape[1]:
        raise ValueError(f"Gate/Up in_features mismatch: {gate.shape} vs {up.shape}")
    return torch.cat([gate, up], dim=0).contiguous()


def _assert_shape(name: str, got: Tuple[int, ...], exp: Tuple[int, ...]) -> None:
    if tuple(got) != tuple(exp):
        raise ValueError(f"Shape mismatch for {name}: got {tuple(got)} expected {tuple(exp)}")


def load_hf_llama_weights_into_local(model: torch.nn.Module, model_dir: str) -> None:
    """Load HF LLaMA weights (safetensors shards) from `model_dir` into our local CausalLM.

    Mapping follows docs/llama_hf_parity.md. This function mutates `model` in-place.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Pre-compute shapes
    num_layers = len(model.blocks)
    d_model = int(getattr(model.cfg, "d_model"))
    n_heads = int(getattr(model.cfg, "n_heads"))
    head_dim = d_model // n_heads
    # Support GQA if attention uses fewer KV heads
    try:
        n_kv_heads = int(model.blocks[0].attn.n_kv_heads)
    except Exception:
        n_kv_heads = n_heads

    # Pending MLP halves per layer
    pending_in: Dict[int, Dict[str, torch.Tensor]] = {}

    # Stream shards to reduce peak memory using safe_open (reads per-tensor)
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as e:
        raise RuntimeError("safetensors is required to load HF checkpoints") from e

    by_file = _index_shards(model_dir)

    def _assign(key: str, tensor: torch.Tensor) -> None:
        nonlocal pending_in
        if key == "model.embed_tokens.weight":
            model.embed.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if key == "model.norm.weight" and hasattr(model, "norm") and hasattr(model.norm, "weight"):
            model.norm.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if key == "lm_head.weight" and hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            # Always load lm_head from checkpoint (even if locally tied, we'll overwrite)
            model.lm_head.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        # Per-layer
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
        # Attention
        if sub == "self_attn.q_proj.weight":
            _assert_shape(f"layers.{li}.q_proj", tuple(tensor.shape), (n_heads * head_dim, d_model))
            blk.attn.w_q.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.k_proj.weight":
            _assert_shape(f"layers.{li}.k_proj", tuple(tensor.shape), (n_kv_heads * head_dim, d_model))
            blk.attn.w_k.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.v_proj.weight":
            _assert_shape(f"layers.{li}.v_proj", tuple(tensor.shape), (n_kv_heads * head_dim, d_model))
            blk.attn.w_v.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "self_attn.o_proj.weight":
            _assert_shape(f"layers.{li}.o_proj", tuple(tensor.shape), (d_model, n_heads * head_dim))
            blk.attn.w_o.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        # Norms
        if sub == "input_layernorm.weight":
            _assert_shape(f"layers.{li}.n1", tuple(tensor.shape), (d_model,))
            blk.n1.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "post_attention_layernorm.weight":
            _assert_shape(f"layers.{li}.n2", tuple(tensor.shape), (d_model,))
            blk.n2.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        # MLP parts
        if sub == "mlp.down_proj.weight":
            _assert_shape(f"layers.{li}.mlp.w_out", tuple(tensor.shape), (blk.mlp.w_out.weight.shape[0], blk.mlp.w_out.weight.shape[1]))
            blk.mlp.w_out.weight.data.copy_(tensor.to(device=device, dtype=dtype))
            return
        if sub == "mlp.gate_proj.weight":
            pending_in.setdefault(li, {})["gate"] = tensor
            # Try fuse if up already present
            if "up" in pending_in[li]:
                gate = pending_in[li].pop("gate")
                up = pending_in[li].pop("up")
                w_in = _stack_gate_up(gate, up)
                _assert_shape(f"layers.{li}.mlp.w_in", tuple(w_in.shape), (blk.mlp.w_in.weight.shape[0], blk.mlp.w_in.weight.shape[1]))
                blk.mlp.w_in.weight.data.copy_(w_in.to(device=device, dtype=dtype))
                pending_in.pop(li, None)
            return
        if sub == "mlp.up_proj.weight":
            pending_in.setdefault(li, {})["up"] = tensor
            if "gate" in pending_in[li]:
                gate = pending_in[li].pop("gate")
                up = pending_in[li].pop("up")
                w_in = _stack_gate_up(gate, up)
                _assert_shape(f"layers.{li}.mlp.w_in", tuple(w_in.shape), (blk.mlp.w_in.weight.shape[0], blk.mlp.w_in.weight.shape[1]))
                blk.mlp.w_in.weight.data.copy_(w_in.to(device=device, dtype=dtype))
                pending_in.pop(li, None)
            return

    # Iterate shards
    for shard_rel, keys in by_file.items():
        shard_fp = os.path.join(model_dir, shard_rel)
        # Use safe_open to avoid loading the full shard at once
        with safe_open(shard_fp, framework="pt", device="cpu") as f:  # type: ignore
            for k in keys:
                try:
                    if not f.keys() or k not in f.keys():  # type: ignore[attr-defined]
                        continue
                except Exception:
                    # Fallback: attempt get_tensor; catch if missing
                    pass
                try:
                    t = f.get_tensor(k)  # type: ignore[attr-defined]
                except Exception:
                    continue
                _assign(k, t)
                del t

    # Fuse any remaining pending halves (should be none if index complete)
    for li, parts in list(pending_in.items()):
        if "gate" in parts and "up" in parts:
            blk = model.blocks[li]
            w_in = _stack_gate_up(parts["gate"], parts["up"])
            _assert_shape(f"layers.{li}.mlp.w_in", tuple(w_in.shape), (blk.mlp.w_in.weight.shape[0], blk.mlp.w_in.weight.shape[1]))
            blk.mlp.w_in.weight.data.copy_(w_in.to(device=device, dtype=dtype))
        pending_in.pop(li, None)

    # Validate tied weights if applicable
    try:
        if model.lm_head.weight.data.data_ptr() == model.embed.weight.data.data_ptr():
            # Tied: ensure lm_head matches embed
            model.lm_head.weight.data.copy_(model.embed.weight.data)
    except Exception:
        pass


