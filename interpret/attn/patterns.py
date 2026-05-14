from __future__ import annotations

import torch


def attention_diagonal_score(attn_probs: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
    if attn_probs.ndim != 4:
        raise ValueError("attn_probs must have shape [B,H,T,S]")
    bsz, heads, tgt_len, src_len = attn_probs.shape
    q = torch.arange(tgt_len, device=attn_probs.device)
    k = q + int(offset)
    valid = (k >= 0) & (k < src_len)
    vals = attn_probs[:, :, q[valid], k[valid]]
    return vals.mean(dim=-1) if vals.numel() else torch.zeros((bsz, heads), device=attn_probs.device, dtype=attn_probs.dtype)


def attention_previous_token_score(attn_probs: torch.Tensor) -> torch.Tensor:
    return attention_diagonal_score(attn_probs, offset=-1)


def attention_distance(attn_probs: torch.Tensor) -> torch.Tensor:
    if attn_probs.ndim != 4:
        raise ValueError("attn_probs must have shape [B,H,T,S]")
    tgt_len, src_len = int(attn_probs.shape[-2]), int(attn_probs.shape[-1])
    q = torch.arange(tgt_len, device=attn_probs.device).view(tgt_len, 1)
    k = torch.arange(src_len, device=attn_probs.device).view(1, src_len)
    dist = (q - k).abs().to(dtype=attn_probs.dtype)
    return (attn_probs * dist.view(1, 1, tgt_len, src_len)).sum(dim=-1).mean(dim=-1)


def attention_prefix_mass(attn_probs: torch.Tensor, prefix_len: int) -> torch.Tensor:
    if attn_probs.ndim != 4:
        raise ValueError("attn_probs must have shape [B,H,T,S]")
    prefix = attn_probs[..., : int(prefix_len)].sum(dim=-1)
    return prefix.mean(dim=-1)


def attention_pattern_summary(attn_probs: torch.Tensor, *, prefix_len: int | None = None) -> dict[str, torch.Tensor]:
    out = {
        "diagonal": attention_diagonal_score(attn_probs).detach().cpu(),
        "previous_token": attention_previous_token_score(attn_probs).detach().cpu(),
        "distance": attention_distance(attn_probs).detach().cpu(),
    }
    if prefix_len is not None:
        out["prefix_mass"] = attention_prefix_mass(attn_probs, prefix_len).detach().cpu()
    return out
