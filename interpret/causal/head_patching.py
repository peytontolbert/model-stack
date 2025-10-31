from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from attn.eager import EagerAttention
from tensor.shape import split_heads, merge_heads
from tensor.positional import build_rope_cache, apply_rotary
from attn.backends import scaled_dot_product_attention, select_attention_backend


def _wrap_forward_capture_heads(attn: EagerAttention, sink: Dict[int, torch.Tensor], layer_idx: int):
    orig_forward = attn.forward

    def forward(q, k, v, mask, cache=None):
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

        sink[layer_idx] = out.detach().cpu()  # (B,H,T,Dh)
        y = merge_heads(out)
        if cache is not None and T > 0:
            cache.append(kh_new, vh_new)
        return attn.w_o(y)

    return orig_forward, forward


def _wrap_forward_patch_heads(attn: EagerAttention, source: Dict[int, torch.Tensor], layer_idx: int, heads: Optional[Iterable[int]] = None):
    H_sel = set(int(h) for h in heads) if heads is not None else None
    orig_forward = attn.forward

    def forward(q, k, v, mask, cache=None):
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

        # Replace selected heads with clean source
        clean = source.get(layer_idx)
        if clean is not None:
            clean = clean.to(device=out.device, dtype=out.dtype)
            if H_sel is None:
                out.copy_(clean)
            else:
                for h in H_sel:
                    if 0 <= h < out.shape[1]:
                        out[:, h].copy_(clean[:, h])

        y = merge_heads(out)
        if cache is not None and T > 0:
            cache.append(kh_new, vh_new)
        return attn.w_o(y)

    return orig_forward, forward


@torch.inference_mode()
def causal_trace_heads_restore_table(
    model: nn.Module,
    *,
    clean_input_ids: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    target_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Return (L,H) table of restored target logit fraction by patching one head at a time.

    If target_token_id is None, uses clean argmax at `position`.
    """
    was_training = model.training
    model.eval()

    # Determine target token id from clean run if not provided
    if target_token_id is None:
        logits_clean = model(clean_input_ids, attn_mask)
        target_token_id = int(logits_clean[0, position].argmax().item())

    # 1) Capture clean per-head outputs
    clean_heads: Dict[int, torch.Tensor] = {}
    wrappers = []
    for li, blk in enumerate(model.blocks):
        attn = getattr(blk, "attn", None)
        if isinstance(attn, EagerAttention):
            orig, new = _wrap_forward_capture_heads(attn, clean_heads, li)
            wrappers.append((attn, orig))
            attn.forward = new  # type: ignore
    try:
        _ = model(clean_input_ids, attn_mask)
    finally:
        for attn, orig in wrappers:
            attn.forward = orig  # type: ignore

    L = len(model.blocks)
    H = getattr(getattr(model.blocks[0], "attn", None), "n_heads", 0) if L > 0 else 0
    table = torch.zeros(L, H)

    # 2) Baseline corrupted logits
    logits_cor = model(corrupted_input_ids, attn_mask)
    base = logits_cor[0, position, target_token_id]

    # 3) For each layer, head: patch that head only and measure recovery
    for li in range(L):
        for h in range(H):
            wrappers2 = []
            attn = getattr(model.blocks[li], "attn", None)
            if not isinstance(attn, EagerAttention):
                continue
            orig, new = _wrap_forward_patch_heads(attn, clean_heads, li, heads=[h])
            wrappers2.append((attn, orig))
            attn.forward = new  # type: ignore
            try:
                logits_patch = model(corrupted_input_ids, attn_mask)
            finally:
                for a, o in wrappers2:
                    a.forward = o  # type: ignore
            val = logits_patch[0, position, target_token_id]
            clean_val = clean_heads.get(li)
            # Use clean run logits (already computed)
            # Need clean target logit value
            # Reuse logits_clean computed earlier
            clean_logit = logits_clean[0, position, target_token_id]
            denom = (clean_logit - base).abs() + 1e-8
            table[li, h] = ((val - base) / denom).detach().cpu()

    if was_training:
        model.train()
    return table


