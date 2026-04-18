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
    """Flatten aligned cached tensors to [N, D_total].

    All selected tensors must resolve to the same [B, T] index grid after applying
    `batch_index` and `time_slice`. Features are concatenated along the channel
    dimension, then flattened over the aligned token rows.
    """
    feats: List[torch.Tensor] = []
    expected_bt: Optional[Tuple[int, int]] = None
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
        if expected_bt is None:
            expected_bt = (B, T)
        elif expected_bt != (B, T):
            raise ValueError(
                f"Feature slices must align to the same [B,T] grid; expected {expected_bt}, got {(B, T)} for '{fs.key}'"
            )
        feats.append(x)
    if not feats:
        return torch.empty(0, 0)
    combined = torch.cat(feats, dim=-1)
    B, T, D = combined.shape
    return combined.reshape(B * T, D)


def targets_from_tokens(
    input_ids: torch.Tensor,
    *,
    batch_index: Optional[int] = None,
    time_slice: Optional[slice] = None,
    target_shift: int = 1,
    position: int = -1,
) -> torch.Tensor:
    """Return aligned token targets from input_ids.

    The returned vector matches token rows selected from the source sequence:
    for each kept source position `t`, the target is `input_ids[..., t + target_shift]`.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [B, T]")
    x = input_ids
    if batch_index is not None:
        x = x[int(batch_index) : int(batch_index) + 1]
    _, seq_len = x.shape
    positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
    valid = (positions + int(target_shift) >= 0) & (positions + int(target_shift) < seq_len)
    if time_slice is not None:
        mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
        mask[time_slice] = True
        valid &= mask
    elif position != -1:
        mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
        idx = int(position if position >= 0 else (seq_len + position))
        if 0 <= idx < seq_len:
            mask[idx] = True
        valid &= mask
    keep = positions[valid]
    if keep.numel() == 0:
        return torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)
    targets = x[:, keep + int(target_shift)].contiguous()
    return targets.reshape(-1)

