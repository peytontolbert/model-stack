"""Quantization utilities and wrappers.

Features:
- INT8 per-channel symmetric quantization for weights
- Fake-quant FP8/FP4 helpers for experimentation
- QuantizedLinearInt8 wrapper for weight-only int8
"""

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor.numerics import percentile_scale, mse_scale


def _symmetric_per_channel_weight_quantize_int8(
    weight: torch.Tensor, ch_axis: int = 0, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to int8 with symmetric per-channel scaling.

    Returns (qweight[int8], scale[float32]) with broadcasting across ch_axis.
    """
    assert weight.dtype.is_floating_point
    # Default: absmax scale per channel
    max_abs = weight.abs().amax(dim=ch_axis, keepdim=True).clamp_min(eps)
    scale = (127.0 / max_abs).to(dtype=torch.float32)
    q = torch.round(weight * scale).clamp_(-127, 127).to(dtype=torch.int8)
    inv_scale = (1.0 / scale).squeeze(ch_axis).to(dtype=torch.float32)
    return q, inv_scale


def calibrate_int8_scales(
    weight: torch.Tensor,
    method: str = "absmax",
    ch_axis: int = 0,
    p: float = 0.999,
) -> torch.Tensor:
    """Return per-channel inverse scales using tensor.numerics calibrations.

    Methods:
    - "absmax": 1/absmax
    - "percentile": 1/percentile(|w|, p)
    - "mse": 1/mse_scale(|w|)
    """
    if method == "absmax":
        max_abs = weight.abs().amax(dim=ch_axis)
        return (1.0 / max_abs.clamp_min(1e-8)).to(dtype=torch.float32)
    if method == "percentile":
        # percentile_scale returns scalar if dim=None; provide dim for per-channel
        inv = []
        for idx in range(weight.size(ch_axis)):
            slicer = [slice(None)] * weight.ndim
            slicer[ch_axis] = idx
            wch = weight[tuple(slicer)]
            s = percentile_scale(wch, p=float(p))
            inv.append((1.0 / s.clamp_min(1e-8)).to(dtype=torch.float32))
        return torch.stack(inv, dim=0)
    if method == "mse":
        inv = []
        for idx in range(weight.size(ch_axis)):
            slicer = [slice(None)] * weight.ndim
            slicer[ch_axis] = idx
            wch = weight[tuple(slicer)]
            s = mse_scale(wch)
            inv.append((1.0 / s.clamp_min(1e-8)).to(dtype=torch.float32))
        return torch.stack(inv, dim=0)
    raise ValueError("Unknown calibration method")


def _dequantize_int8_per_channel(
    qweight: torch.Tensor, inv_scale: torch.Tensor, ch_axis: int = 0
) -> torch.Tensor:
    while inv_scale.ndim < qweight.ndim:
        inv_scale = inv_scale.unsqueeze(ch_axis)
    return (qweight.to(dtype=torch.float32) * inv_scale)


def fake_quantize_fp8(
    x: torch.Tensor, dtype: str = "e4m3"
) -> torch.Tensor:
    """Fake-quantize to FP8 by clamping and rounding mantissa in float space.

    Note: If PyTorch supports torch.float8_e4m3fn, prefer using it directly.
    """
    if dtype not in {"e4m3", "e5m2"}:
        raise ValueError("dtype must be 'e4m3' or 'e5m2'")
    # Simple k-bit mantissa emulation by scaling
    mant_bits = 3 if dtype == "e4m3" else 2
    scale = 2.0 ** mant_bits
    return torch.round(x * scale) / scale


def fake_quantize_fp4(x: torch.Tensor) -> torch.Tensor:
    """Very coarse fake-quantization to ~4-bit dynamic range."""
    scale = 2.0 ** 2  # 4 levels per sign
    x_clamped = x.clamp(-2.0, 2.0)
    return torch.round(x_clamped * scale) / scale


class QuantizedLinearInt8(nn.Module):
    """Weight-only INT8 linear with per-out-feature scaling.

    Quantization is performed per out_channel (dim=0) on the weight matrix.
    Forward pass dequantizes on-the-fly for correctness and portability.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("inv_scale", torch.ones(out_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @torch.no_grad()
    def from_float(self, module: nn.Linear, calibration: str = "absmax", percentile: float = 0.999) -> "QuantizedLinearInt8":
        assert module.in_features == self.in_features and module.out_features == self.out_features
        if calibration == "absmax":
            q, inv_s = _symmetric_per_channel_weight_quantize_int8(module.weight.data, ch_axis=0)
        else:
            inv_s = calibrate_int8_scales(module.weight.data, method=calibration, ch_axis=0, p=float(percentile))
            # Quantize with chosen inv_s
            while inv_s.ndim < module.weight.data.ndim:
                inv_s = inv_s.unsqueeze(1)
            scale = (1.0 / inv_s).to(dtype=torch.float32)
            q = torch.round(module.weight.data * scale).clamp_(-127, 127).to(dtype=torch.int8)
        self.qweight.copy_(q)
        self.inv_scale.copy_(inv_s)
        if module.bias is not None and self.bias is not None:
            self.bias.copy_(module.bias.data)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight on the fly
        w = _dequantize_int8_per_channel(self.qweight, self.inv_scale, ch_axis=0)
        return F.linear(x, w, self.bias)


def quantize_linear_modules(
    model: nn.Module,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, QuantizedLinearInt8]:
    """Replace nn.Linear modules with QuantizedLinearInt8 in-place.

    Returns a mapping of module-name -> quantized module.
    """
    replacements: Dict[str, QuantizedLinearInt8] = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if include is not None and len(list(include)) > 0 and not any(p in name for p in include):
            continue
        if exclude is not None and any(p in name for p in exclude):
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        attr_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        ql = QuantizedLinearInt8(module.in_features, module.out_features, bias=(module.bias is not None))
        ql.from_float(module)
        setattr(parent, attr_name, ql)
        replacements[name] = ql
    return replacements


__all__ = [
    "_symmetric_per_channel_weight_quantize_int8",
    "_dequantize_int8_per_channel",
    "fake_quantize_fp8",
    "fake_quantize_fp4",
    "QuantizedLinearInt8",
    "quantize_linear_modules",
    "calibrate_int8_scales",
]


