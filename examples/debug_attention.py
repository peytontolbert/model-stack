#!/usr/bin/env python3
"""Debug script focusing on attention layer parity."""

import os
import sys
import torch
import torch.nn.functional as F

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
    
    print("=== Loading models ===")
    cfg_hf = AutoConfig.from_pretrained(ckpt)
    hf = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True).eval()
    
    mc = ModelConfig(
        d_model=cfg_hf.hidden_size,
        n_heads=cfg_hf.num_attention_heads,
        n_layers=cfg_hf.num_hidden_layers,
        d_ff=cfg_hf.intermediate_size,
        vocab_size=cfg_hf.vocab_size,
        attn_impl="sdpa",
        rope_theta=float(getattr(cfg_hf, "rope_theta", 1e6)),
        dtype="bfloat16",
        rms_norm_eps=float(cfg_hf.rms_norm_eps),
        head_dim=getattr(cfg_hf, "head_dim", cfg_hf.hidden_size // cfg_hf.num_attention_heads),
    )
    
    local = build_causal_lm(mc, block="llama", n_kv_heads=cfg_hf.num_key_value_heads).eval()
    load_hf_llama_weights_into_local(local, ckpt)
    
    print(f"HF rms_norm_eps: {cfg_hf.rms_norm_eps}")
    print(f"Local model norm eps: {local.norm.eps}")
    print(f"Local block 0 n1 eps: {local.blocks[0].n1.eps}")
    print(f"Local block 0 attention n_kv_heads: {local.blocks[0].attn.n_kv_heads}")
    print(f"Local block 0 attention scaling: {local.blocks[0].attn.scaling}")
    
    # Create simple input
    torch.manual_seed(42)
    B, T = 1, 4
    input_ids = torch.randint(0, 1000, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    
    print(f"\n=== Testing layer 0 step-by-step ===")
    with torch.no_grad():
        # Embeddings
        hf_x = hf.model.embed_tokens(input_ids)
        local_x = local.embed(input_ids)
        print(f"After embed - max diff: {(hf_x - local_x).abs().max().item():.10f}")
        
        # Pre-norm
        hf_x_norm = hf.model.layers[0].input_layernorm(hf_x)
        local_x_norm = local.blocks[0].n1(local_x)
        diff_norm = (hf_x_norm - local_x_norm).abs()
        print(f"After input norm - max diff: {diff_norm.max().item():.10f}")
        print(f"  HF norm output sample: {hf_x_norm[0, 0, :5].tolist()}")
        print(f"  Local norm output sample: {local_x_norm[0, 0, :5].tolist()}")
        
        # QKV projection
        hf_attn = hf.model.layers[0].self_attn
        local_attn = local.blocks[0].attn
        
        hf_q = hf_attn.q_proj(hf_x_norm)
        local_q = local_attn.w_q(local_x_norm)
        print(f"After Q proj - max diff: {(hf_q - local_q).abs().max().item():.10f}")
        
        hf_k = hf_attn.k_proj(hf_x_norm)
        local_k = local_attn.w_k(local_x_norm)
        print(f"After K proj - max diff: {(hf_k - local_k).abs().max().item():.10f}")
        
        hf_v = hf_attn.v_proj(hf_x_norm)
        local_v = local_attn.w_v(local_x_norm)
        print(f"After V proj - max diff: {(hf_v - local_v).abs().max().item():.10f}")
        
        # Reshape to heads
        from tensor.shape import split_heads
        hf_qh = hf_q.view(B, T, cfg_hf.num_attention_heads, 128).transpose(1, 2)
        local_qh = split_heads(local_q, cfg_hf.num_attention_heads)
        print(f"After Q reshape - max diff: {(hf_qh - local_qh).abs().max().item():.10f}")
        
        hf_kh = hf_k.view(B, T, cfg_hf.num_key_value_heads, 128).transpose(1, 2)
        local_kh = split_heads(local_k, cfg_hf.num_key_value_heads)
        print(f"After K reshape - max diff: {(hf_kh - local_kh).abs().max().item():.10f}")
        
        hf_vh = hf_v.view(B, T, cfg_hf.num_key_value_heads, 128).transpose(1, 2)
        local_vh = split_heads(local_v, cfg_hf.num_key_value_heads)
        print(f"After V reshape - max diff: {(hf_vh - local_vh).abs().max().item():.10f}")
        
        # RoPE - this is where divergence might start
        print("\n=== RoPE Application ===")
        
        # HF RoPE
        position_ids = torch.arange(T).unsqueeze(0)
        hf_rope_cos, hf_rope_sin = hf.model.rotary_emb(hf_x_norm, position_ids)
        print(f"HF RoPE cos shape: {hf_rope_cos.shape}")
        print(f"HF RoPE cos sample [0, :5, :5]:\n{hf_rope_cos[0, :5, :5]}")
        
        # Local RoPE
        local_attn._ensure_rope_cache(T, local_x.device, local_x.dtype)
        local_rope_cos = local_attn._rope_cos[:T]
        local_rope_sin = local_attn._rope_sin[:T]
        print(f"Local RoPE cos shape: {local_rope_cos.shape}")
        print(f"Local RoPE cos sample [:5, :5]:\n{local_rope_cos[:5, :5]}")
        
        # Compare RoPE caches
        # HF cos/sin is (B, T, D), ours is (T, D)
        rope_cos_diff = (hf_rope_cos[0] - local_rope_cos).abs().max()
        rope_sin_diff = (hf_rope_sin[0] - local_rope_sin).abs().max()
        print(f"RoPE cos cache diff: {rope_cos_diff.item():.10f}")
        print(f"RoPE sin cache diff: {rope_sin_diff.item():.10f}")
        
        # Apply RoPE
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        hf_qh_rope, hf_kh_rope = apply_rotary_pos_emb(hf_qh, hf_kh, hf_rope_cos, hf_rope_sin)
        
        from tensor.positional import apply_rotary
        local_qh_rope, local_kh_rope = apply_rotary(local_qh, local_kh, local_rope_cos, local_rope_sin)
        
        print(f"After RoPE Q - max diff: {(hf_qh_rope - local_qh_rope).abs().max().item():.10f}")
        print(f"After RoPE K - max diff: {(hf_kh_rope - local_kh_rope).abs().max().item():.10f}")
        
        # Repeat KV for GQA
        from transformers.models.llama.modeling_llama import repeat_kv
        hf_kh_repeated = repeat_kv(hf_kh_rope, cfg_hf.num_attention_heads // cfg_hf.num_key_value_heads)
        hf_vh_repeated = repeat_kv(hf_vh, cfg_hf.num_attention_heads // cfg_hf.num_key_value_heads)
        
        repeat = cfg_hf.num_attention_heads // cfg_hf.num_key_value_heads
        local_kh_repeated = local_kh_rope.repeat_interleave(repeat, dim=1)
        local_vh_repeated = local_vh.repeat_interleave(repeat, dim=1)
        
        print(f"After KV repeat - K max diff: {(hf_kh_repeated - local_kh_repeated).abs().max().item():.10f}")
        print(f"After KV repeat - V max diff: {(hf_vh_repeated - local_vh_repeated).abs().max().item():.10f}")
        
        # Attention scores
        scale = 128 ** -0.5
        hf_scores = torch.matmul(hf_qh_rope, hf_kh_repeated.transpose(2, 3)) * scale
        local_scores = torch.matmul(local_qh_rope, local_kh_repeated.transpose(2, 3)) * scale
        
        print(f"\n=== Attention Scores ===")
        print(f"Attention scores - max diff: {(hf_scores - local_scores).abs().max().item():.10f}")
        print(f"HF scores sample [0, 0, :3, :3]:\n{hf_scores[0, 0, :3, :3]}")
        print(f"Local scores sample [0, 0, :3, :3]:\n{local_scores[0, 0, :3, :3]}")


if __name__ == "__main__":
    main()

