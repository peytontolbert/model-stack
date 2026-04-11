from __future__ import annotations

import torch
import torch.nn as nn

from runtime.ops import add_layer_norm as runtime_add_layer_norm
from runtime.ops import add_rms_norm as runtime_add_rms_norm
from runtime.ops import layer_norm as runtime_layer_norm
from runtime.ops import rms_norm as runtime_rms_norm
from tensor.norms import RMSNorm


def can_apply_native_norm(norm, training: bool) -> bool:
    return (not training) and isinstance(norm, (RMSNorm, nn.LayerNorm))


def apply_native_norm(x: torch.Tensor, norm):
    if can_apply_native_norm(norm, norm.training):
        if isinstance(norm, RMSNorm):
            return runtime_rms_norm(x, weight=norm.weight, eps=float(norm.eps))
        return runtime_layer_norm(x, weight=norm.weight, bias=norm.bias, eps=float(norm.eps))
    return norm(x)


def fused_add_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    norm,
    residual_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if can_apply_native_norm(norm, norm.training):
        if isinstance(norm, RMSNorm):
            return runtime_add_rms_norm(
                x,
                update,
                norm.weight,
                residual_scale=float(residual_scale),
                eps=float(norm.eps),
            )
        return runtime_add_layer_norm(
            x,
            update,
            norm.weight,
            norm.bias,
            residual_scale=float(residual_scale),
            eps=float(norm.eps),
        )
    combined = x + (update * float(residual_scale))
    return combined, norm(combined)
