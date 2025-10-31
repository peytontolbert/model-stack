from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from tensor.losses import masked_ece


@dataclass
class CalibrationResult:
    ece: float
    num_tokens: int


@torch.no_grad()
def evaluate_ece(
    model: torch.nn.Module,
    data: Iterable,
    *,
    device: Optional[str | torch.device] = None,
    n_bins: int = 15,
    max_batches: Optional[int] = None,
) -> CalibrationResult:
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    total_ece = 0.0
    total_tokens = 0

    for i, batch in enumerate(data):
        if max_batches is not None and i >= max_batches:
            break
        x = batch.input_ids.to(dev)
        mask = batch.attn_mask
        logits = model(x, mask, None)
        logits_shift = logits[:, :-1, :]
        targets = x[:, 1:]
        mask_shift = None if mask is None else mask[:, 1:]
        e = masked_ece(logits_shift, targets, mask_shift, n_bins=n_bins)
        total_ece += float(e.item()) * (int(targets.numel()) if mask_shift is None else int(mask_shift.sum().item()))
        total_tokens += int(targets.numel()) if mask_shift is None else int(mask_shift.sum().item())

    mean_ece = total_ece / max(total_tokens, 1)
    return CalibrationResult(ece=mean_ece, num_tokens=total_tokens)


