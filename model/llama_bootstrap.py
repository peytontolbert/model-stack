from typing import Any, Optional   


def build_local_llama_from_snapshot(ckpt_dir: str, device: str, torch_dtype, device_map: Optional[str] = None, gpu_ids: Optional[str] = None) -> Any:
    import os as _os
    import json as _json
    import torch as _torch
    cfg_obj = _json.load(open(_os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8"))
    d_model = int(cfg_obj.get("hidden_size"))
    n_layers = int(cfg_obj.get("num_hidden_layers"))
    n_heads = int(cfg_obj.get("num_attention_heads"))
    d_ff = int(cfg_obj.get("intermediate_size"))
    vocab_size = int(cfg_obj.get("vocab_size"))
    head_dim = int(cfg_obj.get("head_dim", d_model // n_heads))
    n_kv_heads = int(cfg_obj.get("num_key_value_heads", n_heads))
    rope_theta = float(cfg_obj.get("rope_theta", 1e6))
    try:
        rp = cfg_obj.get("rope_parameters", None)
        if isinstance(rp, dict) and rp.get("rope_theta") is not None:
            rope_theta = float(rp.get("rope_theta"))
    except Exception:
        pass
    rms_eps = float(cfg_obj.get("rms_norm_eps", 1e-6))
    from specs.config import ModelConfig  # type: ignore
    from model.factory import build_causal_lm  # type: ignore
    from model.hf_llama_loader import load_hf_llama_weights_into_local  # type: ignore
    # Optional distributed stack imports (best-effort)
    try:
        from dist.engine import DistributedEngine  # type: ignore
        from specs.dist import DistConfig  # type: ignore
        from dist import utils as _dist_utils  # type: ignore
        _HAS_DIST = True
    except Exception:
        DistributedEngine = None  # type: ignore
        DistConfig = None  # type: ignore
        _dist_utils = None  # type: ignore
        _HAS_DIST = False
    mc = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        head_dim=head_dim,
        rope_theta=rope_theta,
        dtype=("bfloat16" if torch_dtype == _torch.bfloat16 else "float32"),
        attn_impl="sdpa",
        rms_norm_eps=rms_eps,
    )
    model = build_causal_lm(mc, block="llama", n_kv_heads=n_kv_heads, tie_weights=bool(cfg_obj.get("tie_word_embeddings", True)))
    try:
        model = model.to(dtype=torch_dtype)
    except Exception:
        pass
    load_hf_llama_weights_into_local(model, ckpt_dir)

    # Adaptive single-device selection: if device_map=="auto", choose GPU with most free memory
    target_device = device
    try:
        if str(device).startswith("cuda") and (device_map == "auto") and _torch.cuda.is_available():
            num = _torch.cuda.device_count()
            if num and num > 0:
                # Probe free mem without permanently changing current device
                try:
                    orig = _torch.cuda.current_device()
                except Exception:
                    orig = 0
                best_idx = 0
                best_free = -1
                for i in range(num):
                    try:
                        # Prefer index-aware API when available
                        free, _total = _torch.cuda.mem_get_info(i)  # type: ignore[arg-type]
                    except Exception:
                        # Fallback: switch temporarily
                        try:
                            _torch.cuda.set_device(i)
                            free, _total = _torch.cuda.mem_get_info()
                        except Exception:
                            continue
                    if int(free) > int(best_free):
                        best_free = int(free)
                        best_idx = i
                # Set chosen device globally so subsequent .to("cuda") aligns
                try:
                    _torch.cuda.set_device(best_idx)
                except Exception:
                    pass
                target_device = f"cuda:{best_idx}"
                # Restore not needed since we set to best_idx intentionally
    except Exception:
        target_device = device

    model = model.to(target_device).eval()

    # If launched under torchrun (WORLD_SIZE>1) and dist stack is available, initialize and wrap
    try:
        if _HAS_DIST:
            world = int(os.environ.get("WORLD_SIZE", "1"))
            if world > 1:
                cfg = DistConfig(backend="nccl", strategy="DDP", precision=("bf16" if torch_dtype == _torch.bfloat16 else "fp32"))  # type: ignore
                eng = DistributedEngine(cfg)  # type: ignore
                eng.init()
                model = eng.wrap_model(model)
    except Exception:
        # Non-fatal: fall back to single-device model
        pass
    return model, cfg_obj

