#!/usr/bin/env python3
"""Minimal parity test - just run both models on the same tiny input."""

import os
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    from transformers import AutoModelForCausalLM, AutoConfig
    from specs.config import ModelConfig
    from model.factory import build_causal_lm
    from model.hf_llama_loader import load_hf_llama_weights_into_local
    
    ckpt = "/data/transformer_10/checkpoints/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    
    print("=== Setup ===")
    cfg_hf = AutoConfig.from_pretrained(ckpt)
    
    print("\n=== Loading HF (layer 0 only test) ===")
    hf = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True).eval()
    
    print("\n=== Loading Local (layer 0 only test) ===")
    mc = ModelConfig(
        d_model=cfg_hf.hidden_size,
        n_heads=cfg_hf.num_attention_heads,
        n_layers=1,  # ONLY 1 LAYER for debugging
        d_ff=cfg_hf.intermediate_size,
        vocab_size=cfg_hf.vocab_size,
        attn_impl="sdpa",
        rope_theta=float(getattr(cfg_hf, "rope_theta", 1e6)),
        dtype="bfloat16",
        rms_norm_eps=float(cfg_hf.rms_norm_eps),
        head_dim=getattr(cfg_hf, "head_dim", cfg_hf.hidden_size // cfg_hf.num_attention_heads),
    )
    
    local = build_causal_lm(mc, block="llama", n_kv_heads=cfg_hf.num_key_value_heads, tie_weights=False).eval()
    load_hf_llama_weights_into_local(local, ckpt)
    
    # Copy ONLY layer 0 weights manually
    print("\n=== Manual single-layer test ===")
    torch.manual_seed(42)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        # HF embedding
        hf_emb = hf.model.embed_tokens(input_ids)
        local_emb = local.embed(input_ids)
        print(f"Embedding diff: {(hf_emb - local_emb).abs().max().item():.10f}")
        
        # Run through layer 0
        hf_layer0_input = hf_emb
        local_layer0_input = local_emb
        
        # Compute position_embeddings for HF (required)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_embeddings = hf.model.rotary_emb(hf_layer0_input, position_ids=position_ids)
        
        # HF layer 0
        hf_layer0_out = hf.model.layers[0](
            hf_layer0_input,
            attention_mask=None,  # No mask for simplicity
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )[0]  # Extract hidden_states from tuple
        
        # Local layer 0  
        local_layer0_out = local.blocks[0](local_layer0_input, None, None)
        
        print(f"Layer 0 output diff: {(hf_layer0_out - local_layer0_out).abs().max().item():.6f}")
        print(f"  HF layer 0 output sample: {hf_layer0_out[0, 0, :5].tolist()}")
        print(f"  Local layer 0 output sample: {local_layer0_out[0, 0, :5].tolist()}")
        
        # Final norm
        hf_normed = hf.model.norm(hf_layer0_out)
        local_normed = local.norm(local_layer0_out)
        
        print(f"After final norm diff: {(hf_normed - local_normed).abs().max().item():.6f}")
        
        # LM head
        hf_logits = hf.lm_head(hf_normed)
        local_logits = local.lm_head(local_normed)
        
        print(f"\nFinal logits diff: {(hf_logits - local_logits).abs().max().item():.6f}")
        print(f"  HF logits [0, 0, :5]: {hf_logits[0, 0, :5].tolist()}")
        print(f"  Local logits [0, 0, :5]: {local_logits[0, 0, :5].tolist()}")
        
        # Full forward for comparison
        print("\n=== Full forward pass (32 layers HF, 1 layer local) ===")
        hf_out = hf(input_ids=input_ids, attention_mask=attention_mask)
        local_out = local(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        print(f"HF full forward [0, 0, :5]: {hf_out.logits[0, 0, :5].tolist()}")
        print(f"Local full forward [0, 0, :5]: {local_out['logits'][0, 0, :5].tolist()}")


if __name__ == "__main__":
    main()

