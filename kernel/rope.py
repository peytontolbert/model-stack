from __future__ import annotations

import torch

from .registry import register
from tensor.positional import apply_rotary as _apply_rotary_ref


@register("rope.apply")
def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply_rotary_ref(q, k, cos, sin)


