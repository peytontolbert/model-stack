import argparse
import json
import os
import sys
from typing import Tuple

import torch

# Ensure local repo modules are importable when running as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _build_local_from_hf_config(model_id_or_dir: str, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoConfig  # type: ignore
    from specs.config import ModelConfig
    from model.factory import build_causal_lm

    cfg_hf = AutoConfig.from_pretrained(model_id_or_dir)
    d_model = int(getattr(cfg_hf, "hidden_size"))
    n_layers = int(getattr(cfg_hf, "num_hidden_layers"))
    n_heads = int(getattr(cfg_hf, "num_attention_heads"))
    head_dim = int(getattr(cfg_hf, "head_dim", max(1, d_model // max(1, n_heads))))
    n_kv_heads = int(getattr(cfg_hf, "num_key_value_heads", n_heads))
    d_ff = int(getattr(cfg_hf, "intermediate_size"))
    vocab_size = int(getattr(cfg_hf, "vocab_size"))
    rms_eps = float(getattr(cfg_hf, "rms_norm_eps", 1e-6))
    # Prefer rope_parameters dict when available (HF LLaMA uses it)
    rope_theta = None
    try:
        rp = getattr(cfg_hf, "rope_parameters", None)
        if isinstance(rp, dict) and "rope_theta" in rp:
            rope_theta = float(rp["rope_theta"])
    except Exception:
        rope_theta = None
    # Parse rope_scaling from HF config when present
    rope_scaling = getattr(cfg_hf, "rope_scaling", None)
    rs_type = None
    rs_factor = None
    rs_orig = None
    rs_low = None
    rs_high = None
    try:
        if isinstance(rope_scaling, dict):
            rs_type = rope_scaling.get("type")
            # Common HF keys: factor, original_max_position_embeddings, low_freq_factor, high_freq_factor
            if "factor" in rope_scaling and rope_scaling["factor"] is not None:
                rs_factor = float(rope_scaling["factor"])  # type: ignore
            if "original_max_position_embeddings" in rope_scaling and rope_scaling["original_max_position_embeddings"] is not None:
                rs_orig = int(rope_scaling["original_max_position_embeddings"])  # type: ignore
            if "low_freq_factor" in rope_scaling and rope_scaling["low_freq_factor"] is not None:
                rs_low = float(rope_scaling["low_freq_factor"])  # type: ignore
            if "high_freq_factor" in rope_scaling and rope_scaling["high_freq_factor"] is not None:
                rs_high = float(rope_scaling["high_freq_factor"])  # type: ignore
    except Exception:
        rs_type = rs_type or None
        rs_factor = rs_factor or None
        rs_orig = rs_orig or None
        rs_low = rs_low or None
        rs_high = rs_high or None

    mc = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        attn_impl="sdpa",
        rope_theta=float(rope_theta if rope_theta is not None else getattr(cfg_hf, "rope_theta", 1e6)),
        dtype=("bfloat16" if dtype == torch.bfloat16 else ("float16" if dtype == torch.float16 else "float32")),
        rms_norm_eps=rms_eps,
        head_dim=head_dim,
        rope_scaling_type=rs_type,
        rope_scaling_factor=rs_factor,
        rope_scaling_original_max_position_embeddings=rs_orig,
        rope_scaling_low_freq_factor=rs_low,
        rope_scaling_high_freq_factor=rs_high,
    )
    # Derive HF rotary attention scaling factor when available (helps parity for scaled RoPE)
    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding  # type: ignore
        class _TmpCfg:
            hidden_size = d_model
            num_attention_heads = n_heads
            head_dim = head_dim
            max_position_embeddings = int(getattr(cfg_hf, "max_position_embeddings", int(args.seq)))
            rope_parameters = {
                "rope_type": getattr(cfg_hf, "rope_type", "default") if hasattr(cfg_hf, "rope_type") else "default",
                "rope_theta": float(rope_theta if rope_theta is not None else getattr(cfg_hf, "rope_theta", 1e6)),
            }
        emb = LlamaRotaryEmbedding(config=_TmpCfg())
        setattr(mc, "rope_attention_scaling", float(getattr(emb, "attention_scaling", 1.0)))
    except Exception:
        pass
    # Pass GQA head count override to blocks
    model = build_causal_lm(mc, block="llama", n_kv_heads=n_kv_heads)
    return model


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True, help="HF model id or local directory with safetensors shards")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--hf-on-gpu", action="store_true", help="If set, try to place HF model on the target device; else keep on CPU to avoid OOM")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"]) 
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--report", default=None, help="Optional JSON report path")
    # Healing/offload options
    p.add_argument("--device-map", default="none", choices=["auto", "none"], help="HF device map for sharding/offload")
    p.add_argument("--offload-folder", default=None, help="Directory for HF cpu/disk offload when using device_map=auto")
    p.add_argument("--max-memory", default=None, help="Max memory map like '0:22GiB,cpu:96GiB' for HF device_map=auto")
    p.add_argument("--load-in-4bit", action="store_true", help="Load HF model in 4bit via bitsandbytes to reduce memory")
    args = p.parse_args()

    dev = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[str(args.dtype)]

    # Resolve HF source: accept model id or a cache directory pointing to snapshots
    def _find_checkpoint_root(root_dir: str) -> str:
        # If directory directly contains model files, use it
        files = set(os.listdir(root_dir)) if os.path.isdir(root_dir) else set()
        direct_ok = any(
            f in files for f in ("model.safetensors.index.json", "model.safetensors", "pytorch_model.bin")
        )
        if direct_ok:
            return root_dir
        # Walk for index.json first, then any safetensors/bin
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            if "model.safetensors.index.json" in filenames:
                return dirpath
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            if any(fn.endswith(".safetensors") or fn == "pytorch_model.bin" for fn in filenames):
                return dirpath
        raise FileNotFoundError(
            f"Could not locate model files under {root_dir}. Expected model.safetensors.index.json or shard files."
        )

    def _resolve_hf_source(hf_arg: str) -> Tuple[str, str]:
        # Returns (hf_src_for_from_pretrained, local_weights_dir_for_loader)
        if os.path.isdir(hf_arg):
            ckpt_dir = _find_checkpoint_root(hf_arg)
            return ckpt_dir, ckpt_dir
        return hf_arg, hf_arg

    hf_src, weights_dir = _resolve_hf_source(args.hf)

    # Build HF reference model
    from transformers import AutoModelForCausalLM  # type: ignore
    from_pretrained_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    # Optional quantization to reduce memory
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(torch.bfloat16 if dtype == torch.bfloat16 else torch.float16),
            )
            from_pretrained_kwargs["quantization_config"] = quant_cfg
        except Exception:
            pass
    # Device-map and offload to avoid OOM
    if args.device_map == "auto":
        from_pretrained_kwargs["device_map"] = "auto"
        # Parse max_memory spec if provided
        if args.max_memory and str(args.max_memory).strip():
            try:
                mm: dict = {}
                for part in str(args.max_memory).split(","):
                    if ":" not in part:
                        continue
                    k, v = part.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    key: object = int(k) if k.isdigit() else k
                    mm[key] = v
                if mm:
                    from_pretrained_kwargs["max_memory"] = mm
            except Exception:
                pass
        if args.offload_folder:
            try:
                os.makedirs(args.offload_folder, exist_ok=True)
                from_pretrained_kwargs["offload_folder"] = args.offload_folder
            except Exception:
                pass

    # Environment knob to reduce fragmentation (safe to set)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    hf = AutoModelForCausalLM.from_pretrained(hf_src, **from_pretrained_kwargs)
    # Determine HF input device: if device_map=auto it can be sharded; use embeddings' device
    try:
        hf_emb = hf.get_input_embeddings()
        hf_dev = hf_emb.weight.device if hf_emb is not None else (dev if args.hf_on_gpu else torch.device("cpu"))
    except Exception:
        hf_dev = (dev if args.hf_on_gpu else torch.device("cpu"))
    # If not using device_map auto and user wants GPU, move
    if args.device_map != "auto":
        # Place HF on CPU by default to avoid VRAM OOM, unless explicitly requested
        hf_target = dev if (args.hf_on_gpu or dev.type != "cuda") else torch.device("cpu")
        try:
            hf = hf.to(hf_target).eval()
            hf_dev = hf_target
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and dev.type == "cuda":
                hf = hf.to(torch.device("cpu")).eval()
                hf_dev = torch.device("cpu")
            else:
                raise
    else:
        hf.eval()

    # Build local and load weights (try target device, fallback to CPU on OOM)
    local = _build_local_from_hf_config(hf_src, dtype=dtype)
    local_dev = dev
    try:
        local = local.to(local_dev).eval()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and dev.type == "cuda":
            local_dev = torch.device("cpu")
            local = local.to(local_dev).eval()
        else:
            raise
    from model.hf_llama_loader import load_hf_llama_weights_into_local
    load_hf_llama_weights_into_local(local, weights_dir)

    # Deterministic input
    torch.manual_seed(0)
    B, T = int(args.batch), int(args.seq)
    V = int(getattr(getattr(hf, "config"), "vocab_size"))
    # Build separate inputs per device
    input_ids_loc = torch.randint(low=0, high=V, size=(B, T), device=local_dev)
    attention_mask_loc = torch.ones(B, T, device=local_dev, dtype=torch.long)
    input_ids_hf = input_ids_loc.to(hf_dev)
    attention_mask_hf = attention_mask_loc.to(hf_dev)

    with torch.no_grad():
        out_hf = hf(input_ids=input_ids_hf, attention_mask=attention_mask_hf, use_cache=False, return_dict=True)
        logits_hf = out_hf.logits  # (B,T,V)
        # Local forward
        out_loc = local(input_ids=input_ids_loc, attention_mask=attention_mask_loc, return_dict=True)
        logits_loc = out_loc["logits"]

    d_logits = _max_abs_diff(logits_loc.float().cpu(), logits_hf.float().cpu())

    rep = {
        "batch": B,
        "seq": T,
        "vocab": V,
        "dtype": str(dtype),
        "device": str(dev),
        "max_abs_diff_logits": float(d_logits),
    }
    print(json.dumps(rep, indent=2))
    if args.report:
        try:
            with open(args.report, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=2)
        except Exception:
            pass


if __name__ == "__main__":
    main()


