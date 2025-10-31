from __future__ import annotations

from typing import Optional

import math
import torch

from attn.eager import EagerAttention
from tensor.shape import split_heads
from tensor.positional import build_rope_cache, apply_rotary
from tensor.masking import broadcast_mask
from tensor.numerics import safe_softmax, entropy_from_probs


@torch.inference_mode()
def attention_weights_for_layer(
    model,
    input_ids: torch.Tensor,
    layer_index: int,
    *,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return attention probabilities (B, H, T, S) for a given transformer layer.

    This re-computes the hidden state up to the target layer and then derives
    attention weights from the layer's attention module parameters. Works with
    `blocks.TransformerBlock` using `attn.eager.EagerAttention`.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # 1) Compute hidden state input to the target block
    x = model.embed(input_ids.to(device))
    for j, blk in enumerate(model.blocks):
        if j >= layer_index:
            break
        x = blk(x, attn_mask, None)

    block = model.blocks[layer_index]
    attn = block.attn
    if not isinstance(attn, EagerAttention):
        raise TypeError("attention_weights_for_layer currently supports EagerAttention blocks only")

    # 2) Build the attention input (prenorm vs postnorm)
    if getattr(block.bc, "norm_policy", "prenorm") == "prenorm":
        q_in = block.n1(x)
    else:
        q_in = x

    B, T, D = q_in.shape
    H = int(attn.n_heads)
    Hk = int(attn.n_kv_heads)
    Dh = int(attn.head_dim)

    # 3) Project Q,K with module weights (self-attn; K from q_in)
    q_lin = attn.w_q(q_in)
    k_lin = attn.w_k(q_in)

    qh = split_heads(q_lin, H)      # (B,H,T,Dh)
    kh_new = split_heads(k_lin, Hk) # (B,Hk,T,Dh)

    # 4) Apply RoPE if used
    if attn.use_rope:
        cos, sin = build_rope_cache(T, Dh, device=device, base_theta=float(attn.rope_theta))
        qh, kh_new = apply_rotary(qh, kh_new, cos.to(dtype=q_in.dtype), sin.to(dtype=q_in.dtype))

    # 5) Expand KV across heads for GQA
    if H != Hk:
        repeat = H // Hk
        kh_all = kh_new.repeat_interleave(repeat, dim=1)
    else:
        kh_all = kh_new

    # 6) Compute attention logits (scaled dot-product)
    #    logits[b,h,t,s] = <q_{b,h,t,:}, k_{b,h,s,:}> / sqrt(Dh)
    scale = 1.0 / math.sqrt(float(Dh))
    logits = torch.einsum("bhtd,bhsd->bhts", qh.float(), kh_all.float()) * float(scale)

    # 7) Apply mask if provided -> boolean mask (B,H,T,S) where True means masked
    if attn_mask is not None:
        m = broadcast_mask(batch_size=B, num_heads=H, tgt_len=T, src_len=T, causal_mask=None, padding_mask=attn_mask)
        logits = logits.masked_fill(m, torch.finfo(logits.dtype).min)
    elif attn.is_causal:
        # Causal mask if none provided
        i = torch.arange(T, device=device).view(T, 1)
        j = torch.arange(T, device=device).view(1, T)
        future = j > i
        m = future.view(1, 1, T, T).expand(B, H, T, T)
        logits = logits.masked_fill(m, torch.finfo(logits.dtype).min)

    # 8) Softmax for probabilities
    probs = safe_softmax(logits.to(dtype=q_in.dtype), dim=-1)
    return probs


@torch.inference_mode()
def attention_entropy_for_layer(
    model,
    input_ids: torch.Tensor,
    layer_index: int,
    *,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return per-head attention entropy (B, H, T) at the target layer."""
    probs = attention_weights_for_layer(model, input_ids, layer_index, attn_mask=attn_mask)
    ent = entropy_from_probs(probs, dim=-1)
    return ent


