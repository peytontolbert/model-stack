def ensure_snapshot(model_id: str, cache_dir: str) -> str:
    try:
        import os as _os
        if _os.path.isdir(model_id) and _os.path.isfile(_os.path.join(model_id, "config.json")):
            return model_id
        try:
            org_name = model_id.strip().split("/")[-2:]
            if len(org_name) == 2:
                org, name = org_name
                dir1 = _os.path.join(cache_dir, f"models--{org}--{name}", "snapshots")
                candidates: list[str] = []
                if _os.path.isdir(dir1):
                    candidates.extend([_os.path.join(dir1, d) for d in _os.listdir(dir1)])
                else:
                    root = _os.path.join(cache_dir)
                    for d in _os.listdir(root):
                        if not d.startswith("models--"):
                            continue
                        snap = _os.path.join(root, d, "snapshots")
                        if _os.path.isdir(snap):
                            for sd in _os.listdir(snap):
                                candidates.append(_os.path.join(snap, sd))
                candidates = [p for p in candidates if _os.path.isfile(_os.path.join(p, "config.json"))]
                if candidates:
                    candidates.sort(key=lambda p: _os.path.getmtime(p), reverse=True)
                    return candidates[0]
        except Exception:
            pass
        try:
            from huggingface_hub import snapshot_download as _snap
        except Exception:
            raise RuntimeError("huggingface_hub is required to download the model snapshot")
        return _snap(repo_id=model_id, cache_dir=cache_dir)
    except Exception as e:
        raise e


def build_local_llama_from_snapshot(ckpt_dir: str, device: str, torch_dtype) -> Any:
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
    model = model.to(device).eval()
    return model, cfg_obj



""" 

def _ensure_snapshot(model_id: str, cache_dir: str) -> str:
    # If local dir with config.json, use directly
    if os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "config.json")):
        return model_id
    # Try existing HF snapshot in cache_dir
    org_name = model_id.strip().split("/")[-2:]
    if len(org_name) == 2:
        org, name = org_name
        dir1 = os.path.join(cache_dir, f"models--{org}--{name}", "snapshots")
        cands = []
        if os.path.isdir(dir1):
            cands.extend([os.path.join(dir1, d) for d in os.listdir(dir1)])
        cands = [p for p in cands if os.path.isfile(os.path.join(p, "config.json"))]
        if cands:
            cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return cands[0]
    # Otherwise, try to download
    from huggingface_hub import snapshot_download  # type: ignore
    return snapshot_download(repo_id=model_id, cache_dir=cache_dir)

"""