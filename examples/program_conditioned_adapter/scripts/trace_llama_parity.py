import os
import json
import argparse
from typing import Dict, Any, List

import torch


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--cache-dir", default="/data/transformer_10/checkpoints")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-layers", type=int, default=8, help="Limit layers to trace for brevity")
    p.add_argument("--save-logits", action="store_true", help="Save HF/local logits to disk to minimize RAM/VRAM use")
    p.add_argument("--out-dir", default="", help="Directory to write saved tensors (defaults to cache-dir/trace)")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ.setdefault("ATTN_BACKEND", "torch")

    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir=args.cache_dir)
    x = tok(args.prompt, return_tensors="pt")
    x = {k: v.to(args.device) for k, v in x.items()}

    # HF model (match dtype)
    hf_dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.float32
    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=hf_dtype,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        cache_dir=args.cache_dir,
    ).eval()

    # Build masks and positions exactly once (no cache), compute HF RoPE and per-layer traces first
    input_ids = x["input_ids"]
    attention_mask = x.get("attention_mask", None)
    B, T = int(input_ids.shape[0]), int(input_ids.shape[1])

    # HF position_ids and mask
    position_ids_hf = torch.arange(T, device=args.device).unsqueeze(0)
    from transformers.models.llama.modeling_llama import LlamaModel  # type: ignore
    # Prepare rotary cos/sin via HF module
    llama_model: LlamaModel = hf.model  # type: ignore[attr-defined]
    cos_hf, sin_hf = llama_model.rotary_emb(
        llama_model.embed_tokens(input_ids), position_ids=position_ids_hf
    )

    # Run HF forward with hooks to collect per-layer hidden (store on CPU to save VRAM)
    hf_layers_out: List[torch.Tensor] = []

    def _hf_layer_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hf_layers_out.append(hs.detach().to(device="cpu", dtype=torch.float32))

    hooks = []
    try:
        for i, layer in enumerate(llama_model.layers[: int(args.max_layers) ]):
            hooks.append(layer.register_forward_hook(_hf_layer_hook))
        _ = hf(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    # Final HF logits (CPU copy)
    out_hf = hf(input_ids=input_ids, attention_mask=attention_mask)
    logits_hf = out_hf.logits[:, -1, :].to("cpu", torch.float32)
    hf_logits_path = None
    if args.save_logits:
        out_dir = args.out_dir or os.path.join(args.cache_dir, "trace")
        os.makedirs(out_dir, exist_ok=True)
        hf_logits_path = os.path.join(out_dir, "hf_logits.pt")
        torch.save(logits_hf, hf_logits_path)
        del logits_hf
        logits_hf = None

    # Free HF before building local to avoid OOM
    del out_hf
    del llama_model
    del hf
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Local model (now that HF is freed)
    from model.hf_snapshot import ensure_snapshot
    ckpt_dir = ensure_snapshot(args.model, args.cache_dir)
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
        dtype=(args.dtype if args.device.startswith("cuda") else "float32"),
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
    local = local.to(device=args.device, dtype=hf_dtype).eval()

    # Local position_ids and mask/cos/sin
    from tensor.masking import create_causal_mask
    from tensor.positional import RotaryEmbeddingHF as LocalRotary
    position_ids_local = torch.arange(T, device=args.device).unsqueeze(0)
    add_mask_local = create_causal_mask(
        input_embeds=local.embed(input_ids),
        attention_mask=attention_mask,
        cache_position=None,
        position_ids=position_ids_local,
        past_key_values=None,
    )  # (B,1,T,S)
    rope = LocalRotary(
        head_dim=int(mc.head_dim or mc.d_model // mc.n_heads),
        base_theta=float(mc.rope_theta),
        attention_scaling=float(getattr(mc, "rope_attention_scaling", 1.0) or 1.0),
        device=input_ids.device,
        scaling_type=getattr(mc, "rope_scaling_type", None),
        scaling_factor=getattr(mc, "rope_scaling_factor", None),
        original_max_position_embeddings=getattr(mc, "rope_scaling_original_max_position_embeddings", None),
        low_freq_factor=getattr(mc, "rope_scaling_low_freq_factor", None),
        high_freq_factor=getattr(mc, "rope_scaling_high_freq_factor", None),
    )
    cos_loc, sin_loc = rope.forward(local.embed(input_ids), position_ids=position_ids_local)

    def _stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        a32 = a.detach().to(torch.float32, copy=False).view(-1)
        b32 = b.detach().to(torch.float32, copy=False).view(-1)
        diff = (a32 - b32)
        l2 = float(diff.norm().item())
        max_abs = float(diff.abs().max().item())
        mean_abs = float(diff.abs().mean().item())
        denom = float(max(1e-12, a32.norm().item()))
        rel = float(l2 / denom)
        cos = float(torch.nn.functional.cosine_similarity(a32.view(1, -1), b32.view(1, -1)).item())
        return {"l2": l2, "max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel, "cos": cos}

    report: Dict[str, Any] = {"layers": []}

    # Compare masks and RoPE
    report["mask"] = {
        "local_shape": tuple(int(s) for s in add_mask_local.shape),
        "local_inf_counts": int(torch.isinf(add_mask_local).sum().item()),
    }
    report["rope"] = {
        "cos": _stats(cos_loc, cos_hf),
        "sin": _stats(sin_loc, sin_hf),
    }

    # Run local forward step-by-step to collect per-layer hidden
    local_layers_out: List[torch.Tensor] = []
    x_loc = local.embed(input_ids)
    mask_loc = add_mask_local
    pos_ids = position_ids_local
    for i, blk in enumerate(local.blocks[: int(args.max_layers) ]):
        x_loc = blk(x_loc, mask_loc, None, (cos_loc, sin_loc), pos_ids)
        local_layers_out.append(x_loc.detach().to(device="cpu", dtype=torch.float32))

    # Compare per-layer outputs
    for i in range(min(len(hf_layers_out), len(local_layers_out))):
        stats = _stats(local_layers_out[i], hf_layers_out[i])
        report["layers"].append({"index": i, **stats})

    # Final logits comparison (one step)
    out_loc = local(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits_loc = out_loc["logits"][:, -1, :].to("cpu", torch.float32)
    loc_logits_path = None
    if args.save_logits:
        out_dir = args.out_dir or os.path.join(args.cache_dir, "trace")
        os.makedirs(out_dir, exist_ok=True)
        loc_logits_path = os.path.join(out_dir, "local_logits.pt")
        torch.save(logits_loc, loc_logits_path)
    # Load from disk for stats if we freed memory
    if logits_hf is None and hf_logits_path:
        logits_hf = torch.load(hf_logits_path, map_location="cpu").to(torch.float32)
    if args.save_logits and loc_logits_path is not None:
        # ensure we use the same on-disk tensor for consistency
        logits_loc = torch.load(loc_logits_path, map_location="cpu").to(torch.float32)
    report["logits"] = _stats(logits_loc, logits_hf)
    if args.save_logits:
        report["logits_paths"] = {"hf": hf_logits_path, "local": loc_logits_path}

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


