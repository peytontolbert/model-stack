from __future__ import annotations

import torch

from runtime.ops import add_rms_norm as runtime_add_rms_norm
from tensor.norms import RMSNorm


def can_fuse_add_rms_norm(norm, training: bool) -> bool:
    return (not training) and isinstance(norm, RMSNorm)


def fused_add_rms_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    norm: RMSNorm,
    residual_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if can_fuse_add_rms_norm(norm, norm.training):
        return runtime_add_rms_norm(
            x,
            update,
            norm.weight,
            residual_scale=float(residual_scale),
            eps=float(norm.eps),
        )
    combined = x + (update * float(residual_scale))
    return combined, norm(combined)
