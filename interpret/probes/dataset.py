from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from interpret.activation_cache import ActivationCache


@dataclass(frozen=True)
class ProbeFeatureSlice:
    key: str
    time_offset: int = 0


@dataclass
class ProbeDataset:
    x: torch.Tensor
    y: torch.Tensor
    batch_index: torch.Tensor
    time_index: torch.Tensor
    feature_keys: list[str]


@dataclass(frozen=True)
class ProbeDatasetSummary:
    num_rows: int
    feature_dim: int
    num_feature_keys: int
    num_batches: int
    num_positions: int
    target_kind: str
    num_targets: int
    target_counts: dict[int, int]


def _normalize_feature(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim != 3:
        raise ValueError("Probe features must be [B,T,D] or [B,D]")
    return x


def _subset_probe_dataset(dataset: ProbeDataset, indices: torch.Tensor) -> ProbeDataset:
    idx = indices.to(dtype=torch.long)
    return ProbeDataset(
        x=dataset.x[idx],
        y=dataset.y[idx],
        batch_index=dataset.batch_index[idx],
        time_index=dataset.time_index[idx],
        feature_keys=list(dataset.feature_keys),
    )


def summarize_probe_dataset(dataset: ProbeDataset, *, max_classes: int = 32) -> ProbeDatasetSummary:
    num_rows = int(dataset.x.shape[0])
    feature_dim = int(dataset.x.shape[-1]) if dataset.x.ndim >= 2 and num_rows > 0 else 0
    num_batches = int(dataset.batch_index.unique().numel()) if dataset.batch_index.numel() > 0 else 0
    num_positions = int(dataset.time_index.unique().numel()) if dataset.time_index.numel() > 0 else 0

    target_counts: dict[int, int] = {}
    if dataset.y.ndim == 1 and dataset.y.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    }:
        target_kind = "classification"
        unique, counts = dataset.y.unique(sorted=True, return_counts=True)
        for cls, count in zip(unique.tolist()[: int(max_classes)], counts.tolist()[: int(max_classes)]):
            target_counts[int(cls)] = int(count)
        num_targets = int(unique.numel())
    else:
        target_kind = "regression"
        num_targets = 1 if dataset.y.ndim <= 1 else int(dataset.y.shape[-1])

    return ProbeDatasetSummary(
        num_rows=num_rows,
        feature_dim=feature_dim,
        num_feature_keys=len(dataset.feature_keys),
        num_batches=num_batches,
        num_positions=num_positions,
        target_kind=target_kind,
        num_targets=num_targets,
        target_counts=target_counts,
    )


def split_probe_dataset(
    dataset: ProbeDataset,
    *,
    val_fraction: float = 0.2,
    generator: Optional[torch.Generator] = None,
    stratify: bool = False,
) -> tuple[ProbeDataset, ProbeDataset]:
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be in the open interval (0, 1)")
    num_rows = int(dataset.x.shape[0])
    if num_rows <= 1:
        raise ValueError("ProbeDataset must contain at least two rows to split")

    if stratify:
        if dataset.y.ndim != 1 or dataset.y.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
        }:
            raise ValueError("stratify=True requires a 1D integral target tensor")
        val_parts: list[torch.Tensor] = []
        train_parts: list[torch.Tensor] = []
        for cls in dataset.y.unique(sorted=True):
            cls_idx = torch.nonzero(dataset.y == cls, as_tuple=False).view(-1)
            if cls_idx.numel() == 1:
                train_parts.append(cls_idx)
                continue
            perm = cls_idx[torch.randperm(cls_idx.numel(), generator=generator)]
            n_val = min(int(cls_idx.numel()) - 1, max(1, int(round(float(cls_idx.numel()) * float(val_fraction)))))
            val_parts.append(perm[:n_val])
            train_parts.append(perm[n_val:])
        val_idx = torch.cat(val_parts, dim=0) if val_parts else torch.empty(0, dtype=torch.long)
        train_idx = torch.cat(train_parts, dim=0) if train_parts else torch.empty(0, dtype=torch.long)
        if val_idx.numel() == 0 or train_idx.numel() == 0:
            raise ValueError("stratified split could not allocate both train and validation rows")
        val_idx = val_idx[torch.randperm(val_idx.numel(), generator=generator)]
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
    else:
        perm = torch.randperm(num_rows, generator=generator)
        n_val = min(num_rows - 1, max(1, int(round(float(num_rows) * float(val_fraction)))))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    return _subset_probe_dataset(dataset, train_idx), _subset_probe_dataset(dataset, val_idx)


def build_probe_dataset(
    cache: ActivationCache,
    slices: Iterable[ProbeFeatureSlice],
    *,
    input_ids: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    target_shift: int = 1,
    sample_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> ProbeDataset:
    """Build an aligned token-level probe dataset from cached activations.

    Each feature slice contributes a [B,T,D] tensor aligned to the same base token
    index `t`, optionally shifted by `time_offset`. Targets default to language-model
    next-token labels via `input_ids[:, t + target_shift]`.
    """
    feature_slices = list(slices)
    if not feature_slices:
        raise ValueError("At least one ProbeFeatureSlice is required")

    tensors = []
    for fs in feature_slices:
        x = cache.get(fs.key)
        if x is None:
            raise KeyError(f"ActivationCache has no key '{fs.key}'")
        tensors.append(_normalize_feature(x))

    batch_size, seq_len = tensors[0].shape[:2]
    for x in tensors[1:]:
        if x.shape[:2] != (batch_size, seq_len):
            raise ValueError("All probe feature tensors must have matching [B,T] dimensions")

    if targets is None:
        if input_ids is None:
            raise ValueError("Either targets or input_ids must be provided")
        if input_ids.ndim != 2 or input_ids.shape[0] != batch_size or input_ids.shape[1] != seq_len:
            raise ValueError("input_ids must have shape [B,T] matching the cached features")

    base_positions = torch.arange(seq_len, dtype=torch.long)
    valid = torch.ones(seq_len, dtype=torch.bool)
    for fs in feature_slices:
        pos = base_positions + int(fs.time_offset)
        valid &= (pos >= 0) & (pos < seq_len)
    if targets is None:
        target_pos = base_positions + int(target_shift)
        valid &= (target_pos >= 0) & (target_pos < seq_len)

    if mask is not None:
        if mask.shape != (batch_size, seq_len):
            raise ValueError("mask must have shape [B,T] matching the cached features")
        valid_mask = mask.to(dtype=torch.bool)
    else:
        valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    rows = []
    batch_rows = []
    time_rows = []
    target_rows = []
    for t in torch.nonzero(valid, as_tuple=False).view(-1).tolist():
        pieces = []
        for fs, x in zip(feature_slices, tensors):
            pieces.append(x[:, t + int(fs.time_offset), :])
        feats = torch.cat(pieces, dim=-1)
        keep = valid_mask[:, t]
        if targets is None:
            y = input_ids[:, t + int(target_shift)]
            if mask is not None:
                keep = keep & mask[:, t + int(target_shift)].to(dtype=torch.bool)
        else:
            if targets.shape == (batch_size, seq_len):
                y = targets[:, t]
            else:
                raise ValueError("Explicit targets must have shape [B,T]")
        if keep.any():
            rows.append(feats[keep])
            target_rows.append(y[keep])
            batch_rows.append(torch.nonzero(keep, as_tuple=False).view(-1))
            time_rows.append(torch.full((int(keep.sum().item()),), int(t), dtype=torch.long))

    if not rows:
        empty = torch.empty(0, 0)
        return ProbeDataset(empty, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), [fs.key for fs in feature_slices])

    x = torch.cat(rows, dim=0)
    y = torch.cat(target_rows, dim=0)
    batch_index = torch.cat(batch_rows, dim=0)
    time_index = torch.cat(time_rows, dim=0)

    if sample_size is not None and int(sample_size) < x.shape[0]:
        perm = torch.randperm(x.shape[0], generator=generator)[: int(sample_size)]
        x = x[perm]
        y = y[perm]
        batch_index = batch_index[perm]
        time_index = time_index[perm]

    return ProbeDataset(
        x=x,
        y=y,
        batch_index=batch_index,
        time_index=time_index,
        feature_keys=[fs.key for fs in feature_slices],
    )


__all__ = [
    "ProbeDataset",
    "ProbeDatasetSummary",
    "ProbeFeatureSlice",
    "build_probe_dataset",
    "split_probe_dataset",
    "summarize_probe_dataset",
]
