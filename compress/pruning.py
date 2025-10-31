"""Pruning utilities: SNIP, movement, magnitude, and mask application."""

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


def magnitude_scores(model: nn.Module, include_bias: bool = False) -> Dict[str, torch.Tensor]:
    scores: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if (not include_bias) and name.endswith(".bias"):
            continue
        if p.requires_grad and p.data.ndim >= 2:
            scores[name] = p.data.abs().detach()
    return scores


def snip_scores(
    model: nn.Module,
    loss: torch.Tensor,
    include_bias: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute SNIP scores: |dLoss/dW * W| evaluated at initialization/batch."""
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], retain_graph=False, allow_unused=True)
    scores: Dict[str, torch.Tensor] = {}
    idx = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = grads[idx]
        idx += 1
        if g is None:
            continue
        if (not include_bias) and name.endswith(".bias"):
            continue
        if p.data.ndim >= 2:
            scores[name] = (g * p).abs().detach()
    return scores


def movement_scores(
    model: nn.Module,
    ema_prev: Dict[str, torch.Tensor],
    momentum: float = 0.95,
    include_bias: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Movement pruning scores based on EMA of parameter updates.

    Returns (scores, new_ema_state).
    """
    new_ema: Dict[str, torch.Tensor] = {}
    scores: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if (not include_bias) and name.endswith(".bias"):
            continue
        if not p.requires_grad or p.grad is None or p.data.ndim < 2:
            continue
        update = -p.grad  # SGD-like direction
        prev = ema_prev.get(name)
        if prev is None:
            cur = update.detach()
        else:
            cur = momentum * prev + (1 - momentum) * update.detach()
        new_ema[name] = cur
        scores[name] = cur.abs()
    return scores, new_ema


def build_global_pruning_mask(
    scores: Dict[str, torch.Tensor],
    sparsity: float,
) -> Dict[str, torch.Tensor]:
    """Given per-parameter score tensors, return binary masks achieving sparsity."""
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1)")
    # Flatten all scores to choose global threshold
    all_scores = torch.cat([s.flatten() for s in scores.values()])
    k = int((1.0 - sparsity) * all_scores.numel())
    k = max(1, min(k, all_scores.numel()))
    topk_val = torch.topk(all_scores, k, sorted=False).values.min()
    masks: Dict[str, torch.Tensor] = {}
    for name, s in scores.items():
        masks[name] = (s >= topk_val).to(dtype=torch.float32)
    return masks


@torch.no_grad()
def apply_pruning_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        m = masks.get(name)
        if m is None:
            continue
        if p.data.shape != m.shape:
            raise ValueError(f"Mask shape mismatch for {name}: {m.shape} vs {p.data.shape}")
        p.data.mul_(m)


__all__ = [
    "magnitude_scores",
    "snip_scores",
    "movement_scores",
    "build_global_pruning_mask",
    "apply_pruning_masks",
]


