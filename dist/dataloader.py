from __future__ import annotations

from typing import Optional

import torch


def build_distributed_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int = 4,
    drop_last: bool = True,
    pin_memory: bool = True,
    seed: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    from torch.utils.data import DataLoader, DistributedSampler

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=drop_last)
    if seed is not None:
        # type: ignore[attr-defined]
        sampler.seed = int(seed)  # for reproducibility across epochs if accessed
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
    )


