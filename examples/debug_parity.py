#!/usr/bin/env python3
"""Debug script to compare HF LLaMA with our local implementation layer by layer."""

import argparse
import os
import sys
import torch

# Add repo root to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def find_checkpoint_root(root_dir: str) -> str:
    """Find the directory containing model checkpoint files."""
    files = set(os.listdir(root_dir)) if os.path.isdir(root_dir) else set()
    direct_ok = any(
        f in files for f in ("model.safetensors.index.json", "model.safetensors", "pytorch_model.bin")
    )
    if direct_ok:
        return root_dir
    # Walk for index.json first
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if "model.safetensors.index.json" in filenames:
            return dirpath
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if any(fn.endswith(".safetensors") or fn == "pytorch_model.bin" for fn in filenames):
            return dirpath
    raise FileNotFoundError(f"Could not locate model files under {root_dir}")


def main():
    p = argparse.ArgumentParser(description="Debug HF vs Local LLaMA parity layer by layer")
    p.add_argument("--hf", required=True, help="HF model id or local checkpoint directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seq", type=int, default=8, help="Sequence length for test")
    p.add_argument("--batch", type=int, default=1, help="Batch size for test")
    p.add_argument("--light", action="store_true", help="Run low-memory: forward parity only; skip weight/progressive checks")
    args = p.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # Find checkpoint directory
    ckpt_dir = find_checkpoint_root(args.hf)
    print(f"Using checkpoint: {ckpt_dir}")

    # Load HF model and config
    print("\n=== Loading HF Model ===")
    from transformers import AutoModelForCausalLM, AutoConfig

    cfg_hf = AutoConfig.from_pretrained(ckpt_dir)
    print(f"HF Config:")
    print(f"  hidden_size: {cfg_hf.hidden_size}")
    print(f"  num_attention_heads: {cfg_hf.num_attention_heads}")
    print(f"  num_key_value_heads: {cfg_hf.num_key_value_heads}")
    print(f"  head_dim: {getattr(cfg_hf, 'head_dim', cfg_hf.hidden_size // cfg_hf.num_attention_heads)}")
    print(f"  intermediate_size: {cfg_hf.intermediate_size}")
    print(f"  num_hidden_layers: {cfg_hf.num_hidden_layers}")
    print(f"  mlp_bias: {getattr(cfg_hf, 'mlp_bias', False)}")
    print(f"  attention_bias: {getattr(cfg_hf, 'attention_bias', False)}")
    print(f"  rms_norm_eps: {cfg_hf.rms_norm_eps}")
    
    hf = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=dtype,
        device_map="cpu",  # keep HF on CPU to minimize VRAM
        low_cpu_mem_usage=True,
    ).eval()

    # Create test input once (CPU), reuse for both models to avoid duplication
    print(f"\n=== Creating Test Input (B={args.batch}, T={args.seq}) ===")
    torch.manual_seed(42)
    input_ids_cpu = torch.randint(0, cfg_hf.vocab_size, (args.batch, args.seq), device=torch.device("cpu"))
    attention_mask_cpu = torch.ones(args.batch, args.seq, device=torch.device("cpu"), dtype=torch.long)
    print(f"Input IDs shape: {input_ids_cpu.shape}")
    print(f"First 10 tokens: {input_ids_cpu[0, :min(10, args.seq)].tolist()}")

    # HF forward on CPU
    print("\n=== HF Forward (CPU) ===")
    with torch.no_grad():
        out_hf = hf(input_ids=input_ids_cpu, attention_mask=attention_mask_cpu, use_cache=False, return_dict=True)
        logits_hf_cpu = out_hf.logits.float().cpu()

    # Free HF before loading local to avoid parallel residency
    import gc
    del out_hf
    del hf
    gc.collect()

    # Load local model
    print("\n=== Loading Local Model ===")
    from specs.config import ModelConfig
    from model.factory import build_causal_lm
    from model.hf_llama_loader import load_hf_llama_weights_into_local

    mc = ModelConfig(
        d_model=cfg_hf.hidden_size,
        n_heads=cfg_hf.num_attention_heads,
        n_layers=cfg_hf.num_hidden_layers,
        d_ff=cfg_hf.intermediate_size,
        vocab_size=cfg_hf.vocab_size,
        attn_impl="sdpa",
        rope_theta=float(getattr(cfg_hf, "rope_theta", 1e6)),
        dtype=args.dtype,
        rms_norm_eps=float(cfg_hf.rms_norm_eps),
        head_dim=getattr(cfg_hf, "head_dim", cfg_hf.hidden_size // cfg_hf.num_attention_heads),
    )
    # Propagate HF rope scaling config if present (e.g., {"type": "linear", "factor": X})
    rope_scaling = getattr(cfg_hf, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        mc.rope_scaling_type = rope_scaling.get("type")
        mc.rope_scaling_factor = float(rope_scaling.get("factor")) if rope_scaling.get("factor") is not None else None
        mc.rope_scaling_original_max_position_embeddings = int(rope_scaling.get("original_max_position_embeddings")) if rope_scaling.get("original_max_position_embeddings") is not None else None
        mc.rope_scaling_low_freq_factor = float(rope_scaling.get("low_freq_factor")) if rope_scaling.get("low_freq_factor") is not None else None
        mc.rope_scaling_high_freq_factor = float(rope_scaling.get("high_freq_factor")) if rope_scaling.get("high_freq_factor") is not None else None
    # Some HF rope types provide an attention scaling factor via rotary embedder
    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        # Build a tiny embedder to fetch its attention_scaling (works for default rope too)
        class _TmpCfg:
            hidden_size = cfg_hf.hidden_size
            num_attention_heads = cfg_hf.num_attention_heads
            head_dim = getattr(cfg_hf, "head_dim", None)
            max_position_embeddings = getattr(cfg_hf, "max_position_embeddings", args.seq)
            rope_parameters = {
                "rope_type": getattr(cfg_hf, "rope_type", "default") if hasattr(cfg_hf, "rope_type") else "default",
                "rope_theta": float(getattr(cfg_hf, "rope_theta", 1e6)),
            }
        tmp = _TmpCfg()
        emb = LlamaRotaryEmbedding(config=tmp)
        mc.rope_attention_scaling = float(getattr(emb, "attention_scaling", 1.0))
    except Exception:
        pass
    
    local = build_causal_lm(mc, block="llama", n_kv_heads=cfg_hf.num_key_value_heads, tie_weights=False)
    local = local.eval()
    
    print(f"Local Config:")
    print(f"  d_model: {mc.d_model}")
    print(f"  n_heads: {mc.n_heads}")
    print(f"  n_layers: {mc.n_layers}")
    print(f"  d_ff: {mc.d_ff}")
    print(f"  head_dim: {mc.head_dim}")

    # Load weights
    print("\n=== Loading Weights ===")
    load_hf_llama_weights_into_local(local, ckpt_dir)
    print("Weights loaded successfully")

    # Move local to target device (HF already freed)
    print(f"\n=== Preparing Local on {device} ===")
    try:
        local = local.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM on {device}, keeping local on CPU")
            # Ensure full rollback to CPU in case some params moved before OOM
            try:
                local = local.to("cpu")
            except Exception:
                pass
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            device = torch.device("cpu")
        else:
            raise

    # Local forward (reuse same tokens)
    print("\n=== Forward Pass ===")
    with torch.no_grad():
        input_ids_loc = input_ids_cpu.to(device)
        attention_mask_loc = attention_mask_cpu.to(device)
        out_local = local(input_ids=input_ids_loc, attention_mask=attention_mask_loc, return_dict=True)
        logits_local = out_local["logits"].float().cpu()

    # Compare
    print("\n=== Comparison ===")
    print(f"HF logits shape: {logits_hf_cpu.shape}")
    print(f"Local logits shape: {logits_local.shape}")
    
    # Move to CPU and float32 for comparison
    logits_local_cpu = logits_local
    
    diff = (logits_hf_cpu - logits_local_cpu).abs()
    print(f"\nLogits difference:")
    print(f"  Max abs diff: {diff.max().item():.6f}")
    print(f"  Mean abs diff: {diff.mean().item():.6f}")
    print(f"  Median abs diff: {diff.median().item():.6f}")
    print(f"  95th percentile: {diff.flatten().quantile(0.95).item():.6f}")
    
    # Check specific positions
    print(f"\nSample logits at position [0, 0, :5]:")
    print(f"  HF:    {logits_hf_cpu[0, 0, :5].tolist()}")
    print(f"  Local: {logits_local_cpu[0, 0, :5].tolist()}")
    print(f"  Diff:  {diff[0, 0, :5].tolist()}")

    # Check weight loading
    if not args.light:
        print("\n=== Weight Verification ===")
        print("Checking if weights were loaded correctly...")
    
    # Compare embedding weights
        # Reload HF on CPU for weight checks
        from transformers import AutoModelForCausalLM as _AutoHF
        hf_w = _AutoHF.from_pretrained(ckpt_dir, torch_dtype=dtype, device_map="cpu", low_cpu_mem_usage=True).eval()
        embed_diff = (hf_w.model.embed_tokens.weight.float().cpu() - local.embed.weight.float().cpu()).abs().max()
        print(f"Embedding weight max diff: {embed_diff.item():.10f}")
    
    # Compare first layer attention weights
        hf_q = hf_w.model.layers[0].self_attn.q_proj.weight.float().cpu()
        local_q = local.blocks[0].attn.w_q.weight.float().cpu()
        q_diff = (hf_q - local_q).abs().max()
        print(f"Layer 0 Q projection max diff: {q_diff.item():.10f}")
    
    # Compare first layer MLP weights
        hf_gate = hf_w.model.layers[0].mlp.gate_proj.weight.float().cpu()
        hf_up = hf_w.model.layers[0].mlp.up_proj.weight.float().cpu()
        hf_fused = torch.cat([hf_gate, hf_up], dim=0)
        local_fused = local.blocks[0].mlp.w_in.weight.float().cpu()
        mlp_diff = (hf_fused - local_fused).abs().max()
        print(f"Layer 0 MLP w_in (fused gate+up) max diff: {mlp_diff.item():.10f}")

    # Verify all layers quickly (Q, K, V, O, norms, MLP in/out) with low-memory comparisons
        try:
            bad_layers = []
            for li in range(cfg_hf.num_hidden_layers):
                b = local.blocks[li]
                qd = (hf_w.model.layers[li].self_attn.q_proj.weight.float().cpu() - b.attn.w_q.weight.float().cpu()).abs().max().item()
                kd = (hf_w.model.layers[li].self_attn.k_proj.weight.float().cpu() - b.attn.w_k.weight.float().cpu()).abs().max().item()
                vd = (hf_w.model.layers[li].self_attn.v_proj.weight.float().cpu() - b.attn.w_v.weight.float().cpu()).abs().max().item()
                od = (hf_w.model.layers[li].self_attn.o_proj.weight.float().cpu() - b.attn.w_o.weight.float().cpu()).abs().max().item()
                n1d = (hf_w.model.layers[li].input_layernorm.weight.float().cpu() - b.n1.weight.float().cpu()).abs().max().item()
                n2d = (hf_w.model.layers[li].post_attention_layernorm.weight.float().cpu() - b.n2.weight.float().cpu()).abs().max().item()
                gate = hf_w.model.layers[li].mlp.gate_proj.weight.float().cpu()
                up = hf_w.model.layers[li].mlp.up_proj.weight.float().cpu()
                w_in_local = b.mlp.w_in.weight.float().cpu()
                ind_gate = (gate - w_in_local[: gate.shape[0], :]).abs().max().item()
                ind_up = (up - w_in_local[gate.shape[0] : gate.shape[0] + up.shape[0], :]).abs().max().item()
                ind = max(ind_gate, ind_up)
                outd = (hf_w.model.layers[li].mlp.down_proj.weight.float().cpu() - b.mlp.w_out.weight.float().cpu()).abs().max().item()
                if max(qd, kd, vd, od, n1d, n2d, ind, outd) > 1e-6:
                    bad_layers.append((li, qd, kd, vd, od, n1d, n2d, ind, outd))
            if bad_layers:
                print(f"First 3 layers with weight diffs > 1e-6: {bad_layers[:3]}")
            else:
                print("All layer weights match HF within 1e-6")
        except Exception as e:
            print(f"[warn] Skipped full-layer weight scan: {e}")

    # Detailed layer-by-layer forward (first layer only)
    if not args.light:
        print("\n=== Layer 0 Detailed Comparison ===")
        with torch.no_grad():
            hf_emb = hf_w.model.embed_tokens(input_ids_cpu)
            local_emb = local.embed(input_ids_cpu)
            emb_diff = (hf_emb.float().cpu() - local_emb.float().cpu()).abs().max()
            print(f"After embedding, max diff: {emb_diff.item():.6f}")
            hf_norm1_in = hf_emb
            local_norm1_in = local_emb
            hf_norm1_out = hf_w.model.layers[0].input_layernorm(hf_norm1_in)
            local_norm1_out = local.blocks[0].n1(local_norm1_in)
            norm1_diff = (hf_norm1_out.float().cpu() - local_norm1_out.float().cpu()).abs().max()
            print(f"After layer 0 input norm, max diff: {norm1_diff.item():.6f}")
            print(f"HF norm1 output [0,0,:5]: {hf_norm1_out[0,0,:5].float().cpu().tolist()}")
            print(f"Local norm1 output [0,0,:5]: {local_norm1_out[0,0,:5].float().cpu().tolist()}")

    # Progressive per-layer comparison
    if not args.light:
        try:
            print("\n=== Progressive Layer-by-Layer Comparison (no mask) ===")
            with torch.no_grad():
                hf_hidden = hf_w.model.embed_tokens(input_ids_cpu)
                local_hidden = local.embed(input_ids_cpu)
                position_ids = torch.arange(input_ids_cpu.shape[1], device=input_ids_cpu.device).unsqueeze(0)
                position_embeddings = hf_w.model.rotary_emb(hf_hidden, position_ids=position_ids)
                for li in range(min(5, cfg_hf.num_hidden_layers)):
                    hf_hidden = hf_w.model.layers[li](
                        hf_hidden,
                        attention_mask=None,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]
                    local_hidden = local.blocks[li](local_hidden, None, None)
                    d = (hf_hidden.float().cpu() - local_hidden.float().cpu()).abs().max().item()
                    print(f"  Layer {li} hidden max diff: {d:.6f}")
        except Exception as e:
            print(f"[warn] Progressive comparison skipped: {e}")

    print("\n=== Analysis Complete ===")
    if diff.max().item() < 1e-3:
        print("✓ Models are in good parity (max diff < 1e-3)")
    elif diff.max().item() < 1e-1:
        print("⚠ Models have moderate differences (1e-3 < max diff < 1e-1)")
    else:
        print("✗ Models have significant differences (max diff > 1e-1)")
        print("\nPossible issues:")
        print("  - Weight loading mismatch")
        print("  - Attention implementation difference")
        print("  - RoPE implementation difference")
        print("  - Normalization epsilon mismatch")


if __name__ == "__main__":
    main()

