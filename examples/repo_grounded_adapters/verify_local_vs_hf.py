import os
import sys
import json
import gc
import torch
from typing import Tuple


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


def _local_logits_last(model, input_ids: torch.Tensor) -> torch.Tensor:
    # Use model's own forward to obtain logits directly
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=None, return_dict=True)
        logits_last = out["logits"][:, -1, :]
    return logits_last.to(torch.float32)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--cache-dir", default="/data/transformer_10/checkpoints")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--gen-new", type=int, default=32, help="Generate this many new tokens for HF/local side-by-side")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir=args.cache_dir)
    x = tok(args.prompt, return_tensors="pt")

    use_cuda = torch.cuda.is_available()
    hf_dtype = torch.bfloat16 if use_cuda else torch.float32
    hf_device_map = "auto" if use_cuda else None

    # HF reference (load in lower precision on GPU to reduce RAM)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=hf_dtype,
        device_map=hf_device_map,
        low_cpu_mem_usage=True,
        cache_dir=args.cache_dir,
    )
    hf.eval()
    with torch.no_grad():
        # Ensure inputs are on the same device as the HF model
        try:
            hf_dev = next(hf.parameters()).device
        except Exception:
            hf_dev = torch.device("cpu")
        x_hf = {k: v.to(hf_dev) for k, v in x.items()}
        out_hf = hf(**x_hf)
        logits_hf = out_hf.logits[:, -1, :].to(device="cpu", dtype=torch.float32)

    # Also capture a short HF generation for comparison
    try:
        gen_hf = hf.generate(**x_hf, max_new_tokens=int(args.gen_new), do_sample=False)
        try:
            in_len = int(x_hf["input_ids"].shape[1])
            seq = gen_hf[0]
            new = seq[in_len:] if (hasattr(seq, "shape") and seq.shape[0] > in_len) else seq
            text_hf = tok.decode(new, skip_special_tokens=True)
        except Exception:
            text_hf = tok.decode(gen_hf[0], skip_special_tokens=True)
    except Exception:
        text_hf = "[error] hf generate failed"

    # Free HF model before loading local to avoid peak memory
    del out_hf
    del hf
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Local
    ckpt_dir = _ensure_snapshot(args.model, args.cache_dir)
    from specs.config import ModelConfig
    from model.factory import build_causal_lm
    from model.hf_llama_loader import load_hf_llama_weights_into_local

    cfg = json.load(open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8"))
    mc = ModelConfig(
        d_model=int(cfg.get("hidden_size")),
        n_heads=int(cfg.get("num_attention_heads")),
        n_layers=int(cfg.get("num_hidden_layers")),
        d_ff=int(cfg.get("intermediate_size")),
        vocab_size=int(cfg.get("vocab_size")),
        head_dim=int(cfg.get("head_dim", int(cfg.get("hidden_size")) // int(cfg.get("num_attention_heads")))),
        rope_theta=float(cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta", 1e6))),
        dtype=("bfloat16" if use_cuda else "float32"),
        attn_impl="sdpa",
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        rope_scaling_type=(cfg.get("rope_scaling", {}) or {}).get("type"),
        rope_scaling_factor=(cfg.get("rope_scaling", {}) or {}).get("factor"),
        rope_scaling_original_max_position_embeddings=(cfg.get("rope_scaling", {}) or {}).get("original_max_position_embeddings"),
        rope_scaling_low_freq_factor=(cfg.get("rope_scaling", {}) or {}).get("low_freq_factor"),
        rope_scaling_high_freq_factor=(cfg.get("rope_scaling", {}) or {}).get("high_freq_factor"),
    )
    n_kv_heads = int(cfg.get("num_key_value_heads", mc.n_heads))
    tie_we = bool(cfg.get("tie_word_embeddings", True))
    local = build_causal_lm(mc, block="llama", n_kv_heads=n_kv_heads, tie_weights=tie_we)
    load_hf_llama_weights_into_local(local, ckpt_dir)
    if use_cuda:
        local = local.to(device="cuda", dtype=torch.bfloat16).eval()
        x_cuda = {k: v.to("cuda") for k, v in x.items()}
        logits_local = _local_logits_last(local, x_cuda["input_ids"]).to(device="cpu", dtype=torch.float32)
        # Local generation (greedy)
        try:
            out_loc = local.generate(input_ids=x_cuda["input_ids"], max_new_tokens=int(args.gen_new), do_sample=False, return_dict=True)
            seq = out_loc.get("sequences", None)
            if seq is None:
                seq = out_loc  # best-effort
            in_len = int(x_cuda["input_ids"].shape[1])
            s0 = seq[0].to("cpu")
            new = s0[in_len:] if (hasattr(s0, "shape") and s0.shape[0] > in_len) else s0
            text_local = tok.decode(new, skip_special_tokens=True)
        except Exception:
            text_local = "[error] local generate failed"
    else:
        local = local.to(device="cpu", dtype=torch.float32).eval()
        logits_local = _local_logits_last(local, x["input_ids"]) 
        try:
            out_loc = local.generate(input_ids=x["input_ids"], max_new_tokens=int(args.gen_new), do_sample=False, return_dict=True)
            seq = out_loc.get("sequences", None)
            if seq is None:
                seq = out_loc
            in_len = int(x["input_ids"].shape[1])
            s0 = seq[0]
            new = s0[in_len:] if (hasattr(s0, "shape") and s0.shape[0] > in_len) else s0
            text_local = tok.decode(new, skip_special_tokens=True)
        except Exception:
            text_local = "[error] local generate failed"

    # Compare
    with torch.no_grad():
        diff_l2 = float((logits_local - logits_hf).norm().item())
        top_hf = torch.topk(logits_hf, k=10, dim=-1).indices[0].tolist()
        top_local = torch.topk(logits_local, k=10, dim=-1).indices[0].tolist()
        overlap = len(set(top_hf).intersection(set(top_local)))

    print(json.dumps({
        "diff_l2": diff_l2,
        "top_hf": top_hf,
        "top_local": top_local,
        "topk_overlap": overlap,
        "hf_text": text_hf,
        "local_text": text_local,
    }, indent=2))


if __name__ == "__main__":
    main()


