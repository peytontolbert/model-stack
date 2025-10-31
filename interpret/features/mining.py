from __future__ import annotations

from typing import List, Tuple

import torch

from interpret.activation_cache import ActivationCache


def topk_positions(cache: ActivationCache, key: str, k: int = 20) -> List[Tuple[int, int, float]]:
    """Return top-k (batch_index, time_index, value) for the max channel per position.

    For a cached tensor [B,T,D], we take max over D to get per-position score
    and return top-k positions by that score.
    """
    x = cache.get(key)
    if x is None:
        return []
    if x.ndim != 3:
        raise ValueError("Expected cached tensor of shape [B,T,D]")
    s, idx = x.float().max(dim=-1)  # (B,T)
    B, T = s.shape
    vals = s.view(-1)
    kk = min(int(k), vals.numel())
    topv, topi = torch.topk(vals, k=kk)
    out: List[Tuple[int, int, float]] = []
    for v, i in zip(topv.tolist(), topi.tolist()):
        b = i // T
        t = i % T
        out.append((int(b), int(t), float(v)))
    return out


