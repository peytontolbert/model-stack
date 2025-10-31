from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from attn.eager import EagerAttention
from tensor.shape import split_heads, merge_heads
from tensor.positional import build_rope_cache, apply_rotary
from attn.backends import scaled_dot_product_attention, select_attention_backend


def _wrap_forward_with_head_ablation(attn: EagerAttention, zero_heads: List[int]):
    zero_set = set(int(h) for h in zero_heads)
    orig_forward = attn.forward

    def forward(q, k, v, mask, cache=None):
        # Re-implement EagerAttention.forward to intercept per-head outputs before w_o
        x = q
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        q_lin = attn.w_q(x)
        k_lin = attn.w_k(x if k is None else k)
        v_lin = attn.w_v(x if v is None else v)

        qh = split_heads(q_lin, attn.n_heads)
        kh_new = split_heads(k_lin, attn.n_kv_heads)
        vh_new = split_heads(v_lin, attn.n_kv_heads)

        if attn.use_rope:
            cos, sin = build_rope_cache(T, attn.head_dim, device=device, base_theta=float(attn.rope_theta))
            qh, kh_new = apply_rotary(qh, kh_new, cos.to(dtype=dtype), sin.to(dtype=dtype))

        if cache is not None:
            k_old, v_old = cache.read(0, cache.length())
            if k_old is not None and k_old.shape[2] > 0:
                kh_all = torch.cat([k_old, kh_new], dim=2)
                vh_all = torch.cat([v_old, vh_new], dim=2)
            else:
                kh_all, vh_all = kh_new, vh_new
        else:
            kh_all, vh_all = kh_new, vh_new

        if attn.n_kv_heads != attn.n_heads:
            repeat = attn.n_heads // attn.n_kv_heads
            kh_all = kh_all.repeat_interleave(repeat, dim=1)
            vh_all = vh_all.repeat_interleave(repeat, dim=1)

        backend = select_attention_backend(is_causal=attn.is_causal, dtype=dtype, seq=T, heads=attn.n_heads, device=device)
        out = scaled_dot_product_attention(
            qh, kh_all, vh_all, attn_mask=mask, dropout_p=attn.attn_dropout_p if attn.training else 0.0, backend=backend, is_causal=attn.is_causal
        )

        # Zero out selected heads in the per-head output tensor prior to merge and w_o
        if zero_set:
            # out shape: (B, H, T, Dh)
            for h in zero_set:
                if 0 <= h < out.shape[1]:
                    out[:, h].zero_()

        y = merge_heads(out)

        if cache is not None and T > 0:
            cache.append(kh_new, vh_new)

        return attn.w_o(y)

    return orig_forward, forward


@contextmanager
def ablate_attention_heads(model: nn.Module, mapping: Dict[int, Iterable[int]]):
    """Temporarily zero-out specified heads in `blocks.{layer}.attn` during forward.

    mapping: {layer_index: [head_indices,...]}
    Only supports `attn.eager.EagerAttention` attention modules.
    """
    handles = []
    origs = []
    try:
        for layer_idx, heads in mapping.items():
            blk = model.blocks[int(layer_idx)]
            attn = getattr(blk, "attn", None)
            if not isinstance(attn, EagerAttention):
                continue
            orig_forward, new_forward = _wrap_forward_with_head_ablation(attn, list(heads))
            origs.append((attn, orig_forward))
            attn.forward = new_forward  # type: ignore
        yield
    finally:
        for attn, orig_forward in origs:
            attn.forward = orig_forward  # type: ignore


