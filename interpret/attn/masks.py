from __future__ import annotations

import torch


def _allowed_from_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return ~mask
    if torch.is_floating_point(mask):
        return mask >= 0
    return mask != 0


def attention_mask_summary(mask: torch.Tensor) -> dict[str, float | tuple[int, ...]]:
    allowed = _allowed_from_mask(mask)
    total = allowed.numel()
    visible = allowed.float().sum()
    return {
        "shape": tuple(mask.shape),
        "visible_fraction": float((visible / max(1, total)).item()),
        "masked_fraction": float((1.0 - visible / max(1, total)).item()),
        "visible_count": float(visible.item()),
    }


def attention_receptive_field(mask: torch.Tensor) -> torch.Tensor:
    allowed = _allowed_from_mask(mask)
    if allowed.ndim < 2:
        raise ValueError("attention mask must have at least query/key dimensions")
    return allowed.float().sum(dim=-1)


def causal_mask_violation_count(mask: torch.Tensor) -> int:
    allowed = _allowed_from_mask(mask)
    q_len = int(allowed.shape[-2])
    k_len = int(allowed.shape[-1])
    q = torch.arange(q_len, device=allowed.device).view(q_len, 1)
    k = torch.arange(k_len, device=allowed.device).view(1, k_len)
    future = k > q
    while future.ndim < allowed.ndim:
        future = future.unsqueeze(0)
    return int((allowed & future).sum().item())


def summarize_attention_receptive_field(mask: torch.Tensor) -> dict[str, float | tuple[int, ...]]:
    rf = attention_receptive_field(mask)
    return {
        "shape": tuple(rf.shape),
        "mean": float(rf.float().mean().item()),
        "min": float(rf.float().min().item()),
        "max": float(rf.float().max().item()),
    }
