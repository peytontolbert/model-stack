from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch

from interpret.activation_cache import ActivationCache


@dataclass
class FeatureSlice:
    key: str
    batch_index: Optional[int] = None  # if None, include all
    time_slice: Optional[slice] = None  # e.g., slice(None) or slice(-1, None)


def flatten_features(cache: ActivationCache, slices: Iterable[FeatureSlice]) -> torch.Tensor:
    """Flatten selected cached tensors to [N, D].

    - Each selected tensor is expected to be [B, T, D] or [B, D].
    - time_slice allows selecting a subset of time steps per example.
    - batch_index can select a single example index.
    """
    feats: List[torch.Tensor] = []
    for fs in slices:
        x = cache.get(fs.key)
        if x is None:
            continue
        if x.ndim == 2:
            # [B, D] -> add time dim of size 1
            x = x.unsqueeze(1)
        if fs.batch_index is not None:
            x = x[int(fs.batch_index) : int(fs.batch_index) + 1]
        if fs.time_slice is not None:
            x = x[:, fs.time_slice]
        B, T, D = x.shape
        feats.append(x.reshape(B * T, D))
    if not feats:
        return torch.empty(0, 0)
    return torch.cat(feats, dim=0)


def targets_from_tokens(input_ids: torch.Tensor, *, position: int = -1) -> torch.Tensor:
    """Return next-token targets for language modeling from input_ids.

    For a single sequence [T], returns a tensor [T-1] of targets for positions 0..T-2.
    If batched [B,T], returns [B*(T-1)].
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [B, T]")
    # Left-shift targets by one for next-token prediction
    targets = input_ids[:, 1:].contiguous()
    return targets.reshape(-1)


