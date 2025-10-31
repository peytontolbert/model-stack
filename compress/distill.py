"""Knowledge Distillation utilities."""

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor.numerics import masked_log_softmax, safe_softmax
from tensor.dtypes import cast_logits_for_loss


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    hard_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard KD loss combining soft teacher targets and optional hard labels.

    loss = alpha * T^2 * KL(softmax(s/T), softmax(t/T)) + (1-alpha) * CE(s, y)
    """
    T = float(temperature)
    # Use tensor numerics to improve stability and dtype handling
    s_logp = masked_log_softmax(cast_logits_for_loss(student_logits) / T, mask=None, dim=-1)
    t_p = safe_softmax(cast_logits_for_loss(teacher_logits) / T, mask=None, dim=-1)
    soft = F.kl_div(s_logp, t_p, reduction="batchmean") * (T * T)
    if hard_labels is None:
        return soft
    hard = F.cross_entropy(student_logits, hard_labels)
    return alpha * soft + (1.0 - alpha) * hard


def mse_match(features_s: Sequence[torch.Tensor], features_t: Sequence[torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    """MSE match a sequence of student and teacher intermediate tensors."""
    if len(features_s) != len(features_t):
        raise ValueError("features_s and features_t must have same length")
    loss = torch.zeros((), device=features_s[0].device)
    for s, t in zip(features_s, features_t):
        if s.shape != t.shape:
            raise ValueError(f"Shape mismatch: {s.shape} vs {t.shape}")
        loss = loss + F.mse_loss(s, t)
    return weight * loss


class DistillHooks:
    """Helper to register hooks for capturing intermediate representations."""

    def __init__(self, modules: Iterable[nn.Module]):
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.values: List[torch.Tensor] = []
        for m in modules:
            self.handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module: nn.Module, inp, out):
        if isinstance(out, torch.Tensor):
            self.values.append(out.detach())
        elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            self.values.append(out[0].detach())

    def close(self) -> None:
        for h in self.handles:
            h.remove()


__all__ = [
    "kd_loss",
    "mse_match",
    "DistillHooks",
]


