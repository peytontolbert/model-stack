from __future__ import annotations

from typing import Dict

import torch

from interpret.activation_cache import ActivationCache


def channel_stats(cache: ActivationCache, key: str) -> Dict[str, torch.Tensor]:
    """Compute mean/std/max per-channel for cached tensor [B,T,D].

    Returns dict with tensors shaped [D].
    """
    x = cache.get(key)
    if x is None:
        return {"mean": torch.tensor([]), "std": torch.tensor([]), "max": torch.tensor([])}
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if x.ndim != 3:
        raise ValueError("Expected tensor of shape [B,T,D] or [B,D]")
    xf = x.float()
    mean = xf.mean(dim=(0, 1))
    std = xf.std(dim=(0, 1), unbiased=False)
    mx = xf.amax(dim=(0, 1))
    return {"mean": mean.to(dtype=x.dtype), "std": std.to(dtype=x.dtype), "max": mx.to(dtype=x.dtype)}


