import os
import sys
import json
import gc
import torch
from typing import Tuple
from model.hf_snapshot import ensure_snapshot
from model.runtime_utils import local_logits_last

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--cache-dir", default="/data/transformer_10/checkpoints")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--gen-new", type=int, default=32, help="Generate this many new tokens for HF/local side-by-side")
    p.add_argument("--mode", default="hf_guided", choices=["independent", "hf_guided", "single_step", "independent_exact", "two_pass_exact"], help="Verification mode")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="Computation dtype for both models during verify")
    p.add_argument("--save-logits", action="store_true", help="Save HF/local last-step logits to disk to minimize RAM/VRAM use")
    p.add_argument("--parity-exact", action="store_true", help="Use explicit matmul+float32 softmax attention for exact HF parity")
    p.add_argument("--out-dir", default="", help="Directory to write saved tensors (defaults to cache-dir/trace)")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    # Force stable attention backend for parity
    os.environ.setdefault("ATTN_BACKEND", "torch")
    if args.parity_exact:
        os.environ["ATTN_PARITY_EXACT"] = "1"

    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir=args.cache_dir)
    x = tok(args.prompt, return_tensors="pt")

    use_cuda = torch.cuda.is_available()
    req_dtype = torch.bfloat16 if str(args.dtype) == "bfloat16" and use_cuda else torch.float32
    hf_device_map = "auto" if use_cuda else None

    # HF reference (load in lower precision on GPU to reduce RAM)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=req_dtype,
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
        hf_logits_path = None
        if args.save_logits:
            out_dir = args.out_dir or os.path.join(args.cache_dir, "trace")
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
            hf_logits_path = os.path.join(out_dir, "hf_logits.pt")
            try:
                torch.save(logits_hf, hf_logits_path)
            except Exception:
                hf_logits_path = None

    # Also capture a short HF generation for comparison (force pure greedy, no suppression)
    try:
        try:
            gen_cfg = hf.generation_config  # type: ignore[attr-defined]
            gen_cfg = gen_cfg.clone()
            # Force deterministic greedy without suppression/penalties
            gen_cfg.do_sample = False
            gen_cfg.temperature = 1.0
            gen_cfg.top_p = 1.0
            gen_cfg.top_k = None
            gen_cfg.repetition_penalty = 1.0
            gen_cfg.no_repeat_ngram_size = 0
            # Neutralize suppression lists if present
            if hasattr(gen_cfg, "suppress_tokens"):
                try:
                    gen_cfg.suppress_tokens = None  # type: ignore[attr-defined]
                except Exception:
                    pass
            if hasattr(gen_cfg, "begin_suppress_tokens"):
                try:
                    gen_cfg.begin_suppress_tokens = None  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            gen_cfg = None  # best-effort
        gen_hf = hf.generate(
            **x_hf,
            generation_config=gen_cfg,
            max_new_tokens=int(args.gen_new),
            do_sample=False,
            eos_token_id=tok.eos_token_id,
        )
        try:
            in_len = int(x_hf["input_ids"].shape[1])
            seq = gen_hf[0]
            new = seq[in_len:] if (hasattr(seq, "shape") and seq.shape[0] > in_len) else seq
            text_hf = tok.decode(new, skip_special_tokens=True)
        except Exception:
            text_hf = tok.decode(gen_hf[0], skip_special_tokens=True)
    except Exception:
        text_hf = "[error] hf generate failed"

    # Keep HF loaded when it's needed later for stepwise checks
    keep_hf_loaded = args.mode in ("hf_guided", "single_step", "independent_exact", "two_pass_exact")

    # If not needed later, free HF before building local to avoid peak GPU memory
    if not keep_hf_loaded:
        try:
            del out_hf
        except Exception:
            pass
        try:
            del hf
        except Exception:
            pass
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Local
    ckpt_dir = ensure_snapshot(args.model, args.cache_dir)
    from specs.config import ModelConfig
    from model.factory import build_causal_lm
    from model.hf_llama_loader import load_hf_llama_weights_into_local

    cfg = json.load(open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8"))
    # Derive pad token id from tokenizer or config
    pad_id = tok.pad_token_id
    if pad_id is None:
        try:
            pad_id = int(cfg.get("pad_token_id")) if (cfg.get("pad_token_id") is not None) else None
        except Exception:
            pad_id = None
    mc = ModelConfig(
        d_model=int(cfg.get("hidden_size")),
        n_heads=int(cfg.get("num_attention_heads")),
        n_layers=int(cfg.get("num_hidden_layers")),
        d_ff=int(cfg.get("intermediate_size")),
        vocab_size=int(cfg.get("vocab_size")),
        head_dim=int(cfg.get("head_dim", int(cfg.get("hidden_size")) // int(cfg.get("num_attention_heads")))),
        rope_theta=float(cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta", 1e6))),
        dtype=("bfloat16" if (req_dtype == torch.bfloat16) else "float32"),
        attn_impl="sdpa",
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        rope_scaling_type=(cfg.get("rope_scaling", {}) or {}).get("type"),
        rope_scaling_factor=(cfg.get("rope_scaling", {}) or {}).get("factor"),
        rope_scaling_original_max_position_embeddings=(cfg.get("rope_scaling", {}) or {}).get("original_max_position_embeddings"),
        rope_scaling_low_freq_factor=(cfg.get("rope_scaling", {}) or {}).get("low_freq_factor"),
        rope_scaling_high_freq_factor=(cfg.get("rope_scaling", {}) or {}).get("high_freq_factor"),
        pad_token_id=(int(pad_id) if pad_id is not None else None),
    )
    n_kv_heads = int(cfg.get("num_key_value_heads", mc.n_heads))
    tie_we = bool(cfg.get("tie_word_embeddings", True))
    local = build_causal_lm(mc, block="llama", n_kv_heads=n_kv_heads, tie_weights=tie_we)
    load_hf_llama_weights_into_local(local, ckpt_dir)
    if use_cuda:
        local = local.to(device="cuda", dtype=req_dtype).eval()
        x_cuda = {k: v.to("cuda") for k, v in x.items()}
        with torch.no_grad():
            out_loc_fwd = local(input_ids=x_cuda["input_ids"], attention_mask=x_cuda.get("attention_mask", None), return_dict=True)
            logits_local = out_loc_fwd["logits"][:, -1, :].to(device="cpu", dtype=torch.float32)
            loc_logits_path = None
            if args.save_logits:
                out_dir = args.out_dir or os.path.join(args.cache_dir, "trace")
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    pass
                loc_logits_path = os.path.join(out_dir, "local_logits.pt")
                try:
                    torch.save(logits_local, loc_logits_path)
                except Exception:
                    loc_logits_path = None
        # Local generation (greedy); exact parity can use stepwise no-cache recompute
        try:
            if args.parity_exact:
                seq = x_cuda["input_ids"].clone()
                attn = x_cuda.get("attention_mask", None)
                for _ in range(int(args.gen_new)):
                    out_step = local(input_ids=seq[:, -1:], attention_mask=attn, return_dict=True)
                    logits = out_step["logits"][:, -1, :]
                    nxt = torch.argmax(logits, dim=-1).view(1, 1)
                    seq = torch.cat([seq, nxt], dim=1)
                    if attn is not None:
                        attn = torch.cat([attn, torch.ones_like(nxt, device=attn.device, dtype=attn.dtype)], dim=1)
                    if tok.eos_token_id is not None and int(nxt[0,0].item()) == int(tok.eos_token_id):
                        break
                s0 = seq[0].to("cpu")
                in_len = int(x_cuda["input_ids"].shape[1])
                new = s0[in_len:] if (hasattr(s0, "shape") and s0.shape[0] > in_len) else s0
                text_local = tok.decode(new, skip_special_tokens=True)
            else:
                out_loc = local.generate(
                    input_ids=x_cuda["input_ids"],
                    attention_mask=x_cuda.get("attention_mask", None),
                    max_new_tokens=int(args.gen_new),
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    return_dict=True,
                )
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
        with torch.no_grad():
            out_loc_fwd = local(input_ids=x["input_ids"], attention_mask=x.get("attention_mask", None), return_dict=True)
            logits_local = out_loc_fwd["logits"][:, -1, :]
            loc_logits_path = None
            if args.save_logits:
                out_dir = args.out_dir or os.path.join(args.cache_dir, "trace")
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    pass
                loc_logits_path = os.path.join(out_dir, "local_logits.pt")
                try:
                    torch.save(logits_local.to(device="cpu", dtype=torch.float32), loc_logits_path)
                except Exception:
                    loc_logits_path = None
        try:
            if args.parity_exact:
                seq = x["input_ids"].clone()
                attn = x.get("attention_mask", None)
                for _ in range(int(args.gen_new)):
                    out_step = local(input_ids=seq[:, -1:], attention_mask=attn, return_dict=True)
                    logits = out_step["logits"][:, -1, :]
                    nxt = torch.argmax(logits, dim=-1).view(1, 1)
                    seq = torch.cat([seq, nxt], dim=1)
                    if attn is not None:
                        attn = torch.cat([attn, torch.ones_like(nxt, dtype=attn.dtype)], dim=1)
                    if tok.eos_token_id is not None and int(nxt[0,0].item()) == int(tok.eos_token_id):
                        break
                s0 = seq[0]
                in_len = int(x["input_ids"].shape[1])
                new = s0[in_len:] if (hasattr(s0, "shape") and s0.shape[0] > in_len) else s0
                text_local = tok.decode(new, skip_special_tokens=True)
            else:
                out_loc = local.generate(
                    input_ids=x["input_ids"],
                    attention_mask=x.get("attention_mask", None),
                    max_new_tokens=int(args.gen_new),
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    return_dict=True,
                )
                seq = out_loc.get("sequences", None)
                if seq is None:
                    seq = out_loc
                in_len = int(x["input_ids"].shape[1])
                s0 = seq[0]
                new = s0[in_len:] if (hasattr(s0, "shape") and s0.shape[0] > in_len) else s0
                text_local = tok.decode(new, skip_special_tokens=True)
        except Exception:
            text_local = "[error] local generate failed"

    # Guided/single-step parity and two-pass exact (single model in memory)
    guided_report = {}
    if args.mode in ("hf_guided", "single_step", "independent_exact"):
        import os as _os
        _trace = (_os.getenv("VERIFY_TRACE", "0") == "1")
        with torch.no_grad():
            # Devices for HF/local
            try:
                hf_dev = next(hf.parameters()).device
            except Exception:
                hf_dev = torch.device("cpu")
            loc_dev = (torch.device("cuda") if use_cuda else torch.device("cpu"))
            # Initialize sequences on devices
            seq_cpu = x["input_ids"].to(torch.long)
            seq_hf = seq_cpu.to(hf_dev)
            seq_loc = seq_cpu.to(loc_dev)
            attn_hf = x.get("attention_mask", None)
            attn_hf = attn_hf.to(hf_dev) if attn_hf is not None else None
            attn_loc = x.get("attention_mask", None)
            attn_loc = attn_loc.to(loc_dev) if attn_loc is not None else None
            steps = int(args.gen_new if args.mode in ("hf_guided", "independent_exact") else 1)
            mismatches = []
            for t in range(steps):
                # HF one-step (no cache)
                out_h = hf(input_ids=seq_hf, attention_mask=attn_hf)
                logits_h = out_h.logits[:, -1, :]
                next_h = torch.argmax(logits_h, dim=-1)
                # Local one-step (no cache, parity_exact attention if requested)
                out_l = local(input_ids=seq_loc, attention_mask=attn_loc, return_dict=True)
                logits_l = out_l["logits"][:, -1, :]
                next_l = torch.argmax(logits_l, dim=-1)
                if _trace:
                    try:
                        print(f"[verify] step={t} indep_exact next_h={int(next_h[0].item())} next_l={int(next_l[0].item())}")
                    except Exception:
                        pass
                if int(next_h[0].item()) != int(next_l[0].item()):
                    mismatches.append({
                        "step": t,
                        "hf": int(next_h[0].item()),
                        "local": int(next_l[0].item()),
                    })
                    if args.mode == "independent_exact":
                        break
                # Advance sequences
                if args.mode == "hf_guided":
                    # append HF token to both
                    seq_hf = torch.cat([seq_hf, next_h.view(1, 1)], dim=1)
                    seq_loc = torch.cat([seq_loc, next_h.to(device=seq_loc.device).view(1, 1)], dim=1)
                    if attn_hf is not None:
                        attn_hf = torch.cat([attn_hf, torch.ones_like(next_h).view(1, 1)], dim=1)
                    if attn_loc is not None:
                        attn_loc = torch.cat([attn_loc, torch.ones_like(next_h).view(1, 1).to(attn_loc.device)], dim=1)
                else:
                    # independent exact: each model appends its own token
                    seq_hf = torch.cat([seq_hf, next_h.view(1, 1)], dim=1)
                    seq_loc = torch.cat([seq_loc, next_l.view(1, 1)], dim=1)
                    if attn_hf is not None:
                        attn_hf = torch.cat([attn_hf, torch.ones_like(next_h).view(1, 1)], dim=1)
                    if attn_loc is not None:
                        attn_loc = torch.cat([attn_loc, torch.ones_like(next_l).view(1, 1)], dim=1)
                # Early stop on eos
                eos_id = tok.eos_token_id
                if eos_id is not None and int(next_h[0].item()) == int(eos_id):
                    break
            # Decode final guided sequence
            new_tokens = seq_cpu.shape[1]
            out_seq = seq_hf[0].to("cpu")
            text_guided = tok.decode(out_seq[new_tokens:], skip_special_tokens=True)
            guided_report = {"guided_text": text_guided, "mismatches": mismatches}

    if args.mode == "two_pass_exact":
        # Pass 1: HF stepwise on current device; record tokens
        import os as _os
        _trace = (_os.getenv("VERIFY_TRACE", "0") == "1")
        with torch.no_grad():
            try:
                hf_dev = next(hf.parameters()).device
            except Exception:
                hf_dev = torch.device("cpu")
            seq_cpu = x["input_ids"].to(torch.long)
            seq_hf = seq_cpu.to(hf_dev)
            attn_hf = x.get("attention_mask", None)
            attn_hf = attn_hf.to(hf_dev) if attn_hf is not None else None
            hf_tokens: list[int] = []
            for t in range(int(args.gen_new)):
                out_h = hf(input_ids=seq_hf, attention_mask=attn_hf)
                logits_h = out_h.logits[:, -1, :]
                next_h = int(torch.argmax(logits_h, dim=-1)[0].item())
                hf_tokens.append(next_h)
                if _trace:
                    print(f"[verify] pass1 step={t} hf_token={next_h}")
                seq_hf = torch.cat([seq_hf, torch.tensor([[next_h]], device=seq_hf.device)], dim=1)
                if attn_hf is not None:
                    attn_hf = torch.cat([attn_hf, torch.ones((1,1), device=attn_hf.device, dtype=attn_hf.dtype)], dim=1)
                eos_id = tok.eos_token_id
                if eos_id is not None and next_h == int(eos_id):
                    break
            # Decode HF text
            out_seq = seq_hf[0].to("cpu")
            text_guided = tok.decode(out_seq[seq_cpu.shape[1]:], skip_special_tokens=True)
            guided_report = {"guided_text": text_guided, "mismatches": []}
        # Free HF completely before local
        try:
            del hf
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # Pass 2: Local stepwise on GPU/CPU; compare tokens at each step
        # Rebuild local (already built above as `local`); if freed earlier, ensure available
        with torch.no_grad():
            # Initialize from original prompt
            if use_cuda:
                seq_loc = x["input_ids"].to("cuda")
                attn_loc = x.get("attention_mask", None)
                attn_loc = attn_loc.to("cuda") if attn_loc is not None else None
            else:
                seq_loc = x["input_ids"].clone()
                attn_loc = x.get("attention_mask", None)
            mismatches: list[dict] = []
            for step, tok_id in enumerate(hf_tokens):
                out_l = local(input_ids=seq_loc, attention_mask=attn_loc, return_dict=True)
                logits_l = out_l["logits"][:, -1, :]
                next_l = int(torch.argmax(logits_l, dim=-1)[0].item())
                if _trace:
                    print(f"[verify] pass2 step={step} hf_token={int(tok_id)} local_next={next_l}")
                if next_l != int(tok_id):
                    mismatches.append({"step": step, "hf": int(tok_id), "local": next_l})
                    break
                # Append HF token (teacher-forced) to keep prefixes identical
                t = torch.tensor([[int(tok_id)]], device=seq_loc.device)
                seq_loc = torch.cat([seq_loc, t], dim=1)
                if attn_loc is not None:
                    attn_loc = torch.cat([attn_loc, torch.ones((1,1), device=attn_loc.device, dtype=attn_loc.dtype)], dim=1)
                eos_id = tok.eos_token_id
                if eos_id is not None and int(tok_id) == int(eos_id):
                    break
            # Decode local text
            out_seq_loc = seq_loc[0].to("cpu")
            text_local_exact = tok.decode(out_seq_loc[x["input_ids"].shape[1]:], skip_special_tokens=True)
            guided_report = {"guided_text": text_guided, "local_guided_text": text_local_exact, "mismatches": mismatches}

    # Free HF if still loaded
    if keep_hf_loaded:
        try:
            del hf
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Compare last-step logits from initial prompt
    with torch.no_grad():
        nan_hf = bool(torch.isnan(logits_hf).any().item())
        nan_loc = bool(torch.isnan(logits_local).any().item())
        diff_l2 = float((logits_local - logits_hf).norm().item()) if not (nan_hf or nan_loc) else float('nan')
        vals_hf, idx_hf = torch.topk(logits_hf, k=10, dim=-1)
        vals_loc, idx_loc = torch.topk(logits_local, k=10, dim=-1)
        top_hf = idx_hf[0].tolist()
        top_local = idx_loc[0].tolist()
        topv_hf = [float(x) for x in vals_hf[0].tolist()]
        topv_local = [float(x) for x in vals_loc[0].tolist()]
        overlap = len(set(top_hf).intersection(set(top_local)))

    out_obj = {
        "diff_l2": diff_l2,
        "top_hf": top_hf,
        "top_local": top_local,
        "topv_hf": topv_hf,
        "topv_local": topv_local,
        "topk_overlap": overlap,
        "hf_text": text_hf,
        "local_text": text_local,
        "nan_hf": nan_hf,
        "nan_local": nan_loc,
    }
    if guided_report:
        out_obj.update(guided_report)
        # In exact parity modes, prefer the stepwise exact outputs for local_text to reflect verified path.
        try:
            if args.mode in ("two_pass_exact", "independent_exact", "hf_guided"):
                if isinstance(guided_report, dict):
                    if "local_guided_text" in guided_report:
                        out_obj["local_text"] = guided_report["local_guided_text"]
                    elif "guided_text" in guided_report:
                        out_obj["local_text"] = guided_report["guided_text"]
        except Exception:
            pass
    try:
        if args.save_logits:
            out_obj["logits_paths"] = {
                "hf": (hf_logits_path if ('hf_logits_path' in locals()) else None),
                "local": (loc_logits_path if ('loc_logits_path' in locals()) else None),
            }
    except Exception:
        pass
    print(json.dumps(out_obj, indent=2))


if __name__ == "__main__":
    main()


