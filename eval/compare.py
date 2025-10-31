from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .loop import evaluate_lm_next_token, EvalResult


@dataclass
class CompareResult:
    a: EvalResult
    b: EvalResult
    delta_ppl: float
    delta_acc: Optional[float]


def compare_models(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    data: Iterable,
    *,
    device: Optional[str | torch.device] = None,
    max_batches: Optional[int] = None,
) -> CompareResult:
    a = evaluate_lm_next_token(model_a, data, device=device, max_batches=max_batches, report_accuracy=True)
    b = evaluate_lm_next_token(model_b, data, device=device, max_batches=max_batches, report_accuracy=True)
    d_ppl = b.ppl - a.ppl
    d_acc = None
    if a.acc is not None and b.acc is not None:
        d_acc = b.acc - a.acc
    return CompareResult(a=a, b=b, delta_ppl=d_ppl, delta_acc=d_acc)


