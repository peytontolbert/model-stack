"""Quantization utilities and wrappers.

Features:
- INT8 per-channel symmetric quantization for weights
- INT4 packed per-channel symmetric quantization for weights
- NF4 packed codebook quantization for weights
- Fake-quant FP8/FP4 helpers for experimentation
- QuantizedLinear* wrappers for weight-only inference
- Trainable runtime-row BitNet QAT linear with packed runtime export
"""

import math
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from runtime.native import has_native_op, native_module
from runtime.ops import linear as runtime_linear
from runtime.ops import bitnet_transform_input as runtime_bitnet_transform_input
from runtime.ops import pack_bitnet_weight as runtime_pack_bitnet_weight
from runtime.quant import bitnet_linear as runtime_bitnet_linear
from runtime.quant import bitnet_linear_compute_packed as runtime_bitnet_linear_compute_packed
from runtime.quant import bitnet_int8_linear_from_float as runtime_bitnet_int8_linear_from_float
from runtime.quant import bitnet_linear_from_float as runtime_bitnet_linear_from_float
from runtime.quant import _dequantize_int8_activation
from runtime.quant import _dequantize_int8_weight
from runtime.quant import _should_use_eager_autograd_fallback
from runtime.quant import nf4_dequantize as runtime_nf4_dequantize
from runtime.quant import nf4_linear as runtime_nf4_linear
from runtime.quant import nf4_quantize as runtime_nf4_quantize
from runtime.quant import int4_linear as runtime_int4_linear
from runtime.quant import int8_linear as runtime_int8_linear
from runtime.quant import int8_linear_from_quantized_activation as runtime_int8_linear_from_quantized_activation
from runtime.quant import quantize_activation_int8_rowwise as runtime_quantize_activation_int8_rowwise
from runtime.quant import fp8_linear as runtime_fp8_linear
from tensor.numerics import percentile_scale, mse_scale


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(f"axis {axis} out of bounds for tensor rank {ndim}")
    return axis


def _channel_reduce_dims(ndim: int, ch_axis: int) -> tuple[int, ...]:
    axis = _normalize_axis(ch_axis, ndim)
    return tuple(dim for dim in range(ndim) if dim != axis)


def _reshape_channel_values(values: torch.Tensor, like: torch.Tensor, ch_axis: int) -> torch.Tensor:
    axis = _normalize_axis(ch_axis, like.ndim)
    if values.ndim == 1:
        view_shape = [1] * like.ndim
        view_shape[axis] = values.shape[0]
        return values.view(*view_shape)
    out = values
    while out.ndim < like.ndim:
        out = out.unsqueeze(-1)
    return out


def _parameter_signature(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    version = 0
    try:
        version = int(getattr(tensor, "_version", 0))
    except Exception:
        version = 0
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(tensor.shape),
        version,
        int(tensor.data_ptr()),
    )


def _spin_enabled(spin_enabled_flag: torch.Tensor) -> bool:
    return bool(int(spin_enabled_flag.item()) != 0)


def _tensor_version_safe(tensor: torch.Tensor) -> int:
    try:
        return int(getattr(tensor, "_version", 0))
    except Exception:
        return 0


def _power_of_two_segments(size: int) -> tuple[int, ...]:
    if size <= 0:
        return ()
    out: list[int] = []
    remaining = int(size)
    while remaining > 0:
        seg = 1 << (remaining.bit_length() - 1)
        out.append(seg)
        remaining -= seg
    return tuple(out)


def _hadamard_lastdim_power2(x: torch.Tensor) -> torch.Tensor:
    width = int(x.shape[-1])
    if width <= 1:
        return x
    if width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    y = x.reshape(-1, width)
    block = 1
    while block < width:
        y = y.view(-1, width // (block * 2), 2, block)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.cat((a + b, a - b), dim=-1)
        block *= 2
    return (y.view_as(x) / math.sqrt(float(width))).contiguous()


def apply_spin_transform(x: torch.Tensor, spin_signs: torch.Tensor) -> torch.Tensor:
    """Apply a segmented signed Hadamard transform on the last dimension."""
    if x.shape[-1] != spin_signs.numel():
        raise ValueError("spin_signs must match the last dimension of x")
    work_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    signs = spin_signs.to(device=x.device, dtype=work_dtype)
    x_local = x.to(dtype=work_dtype) * signs
    segments = _power_of_two_segments(int(signs.numel()))
    if not segments:
        return x_local
    if len(segments) == 1:
        return _hadamard_lastdim_power2(x_local)
    parts = []
    start = 0
    for seg in segments:
        stop = start + seg
        part = x_local[..., start:stop]
        parts.append(_hadamard_lastdim_power2(part) if seg > 1 else part)
        start = stop
    return torch.cat(parts, dim=-1).contiguous()


def undo_spin_transform(x: torch.Tensor, spin_signs: torch.Tensor) -> torch.Tensor:
    """Undo :func:`apply_spin_transform` on the last dimension."""
    if x.shape[-1] != spin_signs.numel():
        raise ValueError("spin_signs must match the last dimension of x")
    work_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    signs = spin_signs.to(device=x.device, dtype=work_dtype)
    segments = _power_of_two_segments(int(signs.numel()))
    x_local = x.to(dtype=work_dtype)
    if not segments:
        return x_local
    if len(segments) == 1:
        return (_hadamard_lastdim_power2(x_local) * signs).contiguous()
    parts = []
    start = 0
    for seg in segments:
        stop = start + seg
        part = x_local[..., start:stop]
        part = _hadamard_lastdim_power2(part) if seg > 1 else part
        parts.append(part)
        start = stop
    return (torch.cat(parts, dim=-1) * signs).contiguous()


def _configure_spin_state(
    spin_enabled_flag: torch.Tensor,
    spin_signs: torch.Tensor,
    *,
    enabled: bool,
    random_signs: bool = True,
    seed: int = 0,
) -> None:
    spin_enabled_flag.copy_(torch.tensor(1 if enabled else 0, device=spin_enabled_flag.device, dtype=spin_enabled_flag.dtype))
    if not enabled:
        spin_signs.fill_(1.0)
        return
    if not random_signs:
        spin_signs.fill_(1.0)
        return
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(0, 2, (spin_signs.numel(),), generator=generator, dtype=torch.int64).mul_(2).sub_(1)
    spin_signs.copy_(signs.to(device=spin_signs.device, dtype=spin_signs.dtype))


def _symmetric_per_channel_weight_quantize_int8(
    weight: torch.Tensor, ch_axis: int = 0, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to int8 with symmetric per-channel scaling.

    Returns (qweight[int8], inv_scale[float32]) where inv_scale is the
    per-channel dequant multiplier for broadcasting across ch_axis.
    """
    assert weight.dtype.is_floating_point
    axis = _normalize_axis(ch_axis, weight.ndim)
    reduce_dims = _channel_reduce_dims(weight.ndim, axis)
    max_abs = weight.abs().amax(dim=reduce_dims, keepdim=False).clamp_min(eps).to(dtype=torch.float32)
    scale = 127.0 / _reshape_channel_values(max_abs, weight, axis)
    q = torch.round(weight.float() * scale).clamp_(-127, 127).to(dtype=torch.int8)
    inv_scale = (max_abs / 127.0).to(dtype=torch.float32)
    return q, inv_scale


def _pack_int4_signed(qweight: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit values from [-8, 7] into uint8 along the last dim."""
    if qweight.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("qweight must be an integer tensor")
    q = qweight.to(dtype=torch.int16)
    if bool(((q < -8) | (q > 7)).any()):
        raise ValueError("signed int4 values must be in [-8, 7]")
    q = q + 8
    if q.size(-1) % 2 != 0:
        pad = torch.full((*q.shape[:-1], 1), 8, dtype=q.dtype, device=q.device)
        q = torch.cat([q, pad], dim=-1)
    low = q[..., 0::2]
    high = q[..., 1::2]
    return (low | (high << 4)).to(dtype=torch.uint8)


def _unpack_int4_signed(packed: torch.Tensor, original_last_dim: int) -> torch.Tensor:
    """Unpack signed 4-bit values from uint8 into int8 along the last dim."""
    if packed.dtype != torch.uint8:
        raise TypeError("packed int4 tensor must use uint8 storage")
    packed_i = packed.to(dtype=torch.int16)
    low = packed_i & 0x0F
    high = (packed_i >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)[..., :original_last_dim]
    return (unpacked - 8).to(dtype=torch.int8)


def _pack_nf4_codes(qcodes: torch.Tensor) -> torch.Tensor:
    return _pack_int4_signed(qcodes.to(dtype=torch.int16) - 8)


def _unpack_nf4_codes(packed: torch.Tensor, original_last_dim: int) -> torch.Tensor:
    return (_unpack_int4_signed(packed, original_last_dim).to(dtype=torch.int16) + 8).to(dtype=torch.uint8)


def _unpack_bitnet_signed(packed: torch.Tensor, original_last_dim: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError("packed BitNet tensor must use uint8 storage")
    packed_i = packed.to(dtype=torch.int16)
    c0 = packed_i & 0x03
    c1 = (packed_i >> 2) & 0x03
    c2 = (packed_i >> 4) & 0x03
    c3 = (packed_i >> 6) & 0x03
    codes = torch.stack((c0, c1, c2, c3), dim=-1).flatten(-2)[..., :original_last_dim]
    lut = torch.tensor([-1, 0, 1, 0], dtype=torch.int8, device=packed.device)
    return lut[codes.to(dtype=torch.long)]


def _bitnet_row_scales(
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
) -> torch.Tensor:
    logical_out = int(layout_header[3].item())
    scale_granularity = int(layout_header[7].item())
    scale_group_size = int(layout_header[8].item())
    out = torch.zeros(logical_out, device=scale_values.device, dtype=torch.float32)
    values = scale_values.flatten()
    if scale_granularity == 0:
        out.fill_(float(values[0].item()))
        return out
    if scale_granularity == 1:
        for idx in range(segment_offsets.numel() - 1):
            start = int(segment_offsets[idx].item())
            end = int(segment_offsets[idx + 1].item())
            out[start:end] = float(values[idx].item())
        return out
    if scale_granularity == 2:
        if scale_group_size <= 0:
            raise ValueError("BitNet scale_group_size must be positive for per-output-group scaling")
        for idx in range(values.numel()):
            start = idx * scale_group_size
            end = min(logical_out, start + scale_group_size)
            out[start:end] = float(values[idx].item())
        return out
    raise ValueError(f"Unsupported BitNet scale granularity: {scale_granularity}")


def _dequantize_bitnet_weight(
    packed_weight: torch.Tensor,
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    logical_out = int(layout_header[3].item())
    logical_in = int(layout_header[4].item())
    unpacked = _unpack_bitnet_signed(packed_weight, original_last_dim=logical_in)[:logical_out]
    row_scales = _bitnet_row_scales(scale_values, layout_header, segment_offsets)
    return unpacked.to(dtype=torch.float32).mul(row_scales.unsqueeze(-1)).to(dtype=dtype)


_BITNET_COMPUTE_TILE_N = 128
_BITNET_DECODE_TILE_N = 128
_BITNET_DECODE_CHUNK_K = 32


def _pack_bitnet_compute_weight(
    packed_weight: torch.Tensor,
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
    *,
    tile_n: int = _BITNET_COMPUTE_TILE_N,
) -> tuple[torch.Tensor, torch.Tensor]:
    if packed_weight.ndim != 2:
        raise ValueError("BitNet packed_weight must be rank-2")
    if tile_n <= 0:
        raise ValueError("BitNet compute tile_n must be positive")
    rows = int(packed_weight.shape[0])
    packed_cols = int(packed_weight.shape[1])
    word_cols = (packed_cols + 3) // 4
    pad_cols = word_cols * 4 - packed_cols
    if pad_cols > 0:
        pad_value = torch.full(
            (rows, pad_cols),
            0x55,
            device=packed_weight.device,
            dtype=torch.uint8,
        )
        packed_bytes = torch.cat((packed_weight, pad_value), dim=1)
    else:
        packed_bytes = packed_weight
    packed_bytes_i64 = packed_bytes.reshape(rows, word_cols, 4).to(dtype=torch.int64)
    packed_words = (
        packed_bytes_i64[..., 0]
        | (packed_bytes_i64[..., 1] << 8)
        | (packed_bytes_i64[..., 2] << 16)
        | (packed_bytes_i64[..., 3] << 24)
    ).to(dtype=torch.int32)
    padded_rows = ((rows + tile_n - 1) // tile_n) * tile_n
    if padded_rows > rows:
        pad_rows = torch.full(
            (padded_rows - rows, word_cols),
            int(0x55555555),
            device=packed_words.device,
            dtype=torch.int32,
        )
        packed_words = torch.cat((packed_words, pad_rows), dim=0)
    packed_words = packed_words.view(padded_rows // tile_n, tile_n, word_cols).permute(0, 2, 1).contiguous()

    row_scales = _bitnet_row_scales(scale_values, layout_header, segment_offsets)
    if padded_rows > int(row_scales.shape[0]):
        row_scales = torch.cat(
            (
                row_scales,
                torch.zeros(
                    padded_rows - int(row_scales.shape[0]),
                    device=row_scales.device,
                    dtype=torch.float32,
                ),
            ),
            dim=0,
        )
    row_scales = row_scales.view(padded_rows // tile_n, tile_n).contiguous()
    return packed_words, row_scales


def _pack_bitnet_decode_backend_weight(
    packed_weight: torch.Tensor,
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
    *,
    tile_n: int = _BITNET_DECODE_TILE_N,
    chunk_k: int = _BITNET_DECODE_CHUNK_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if packed_weight.ndim != 2:
        raise ValueError("BitNet packed_weight must be rank-2")
    if tile_n <= 0:
        raise ValueError("BitNet decode tile_n must be positive")
    if chunk_k <= 0 or chunk_k > 32:
        raise ValueError("BitNet decode chunk_k must be in [1, 32]")
    if chunk_k != 32:
        raise ValueError("BitNet decode backend currently requires chunk_k == 32")

    logical_out = int(layout_header[3].item())
    logical_in = int(layout_header[4].item())
    signed = _unpack_bitnet_signed(packed_weight[:logical_out], original_last_dim=logical_in).to(dtype=torch.int8)

    chunk_cols = (logical_in + chunk_k - 1) // chunk_k
    padded_in = chunk_cols * chunk_k
    if padded_in > logical_in:
        signed = torch.cat(
            (
                signed,
                torch.zeros(
                    logical_out,
                    padded_in - logical_in,
                    device=signed.device,
                    dtype=torch.int8,
                ),
            ),
            dim=1,
        )

    padded_rows = ((logical_out + tile_n - 1) // tile_n) * tile_n
    if padded_rows > logical_out:
        signed = torch.cat(
            (
                signed,
                torch.zeros(
                    padded_rows - logical_out,
                    padded_in,
                    device=signed.device,
                    dtype=torch.int8,
                ),
            ),
            dim=0,
        )

    chunked = signed.view(padded_rows, chunk_cols, chunk_k)
    bit_positions = (1 << torch.arange(chunk_k, device=chunked.device, dtype=torch.int64)).view(1, 1, chunk_k)
    nz_masks = ((chunked != 0).to(dtype=torch.int64) * bit_positions).sum(dim=-1).to(dtype=torch.int32)
    sign_masks = ((chunked > 0).to(dtype=torch.int64) * bit_positions).sum(dim=-1).to(dtype=torch.int32)
    nz_masks = nz_masks.view(padded_rows // tile_n, tile_n, chunk_cols).permute(0, 2, 1).contiguous()
    sign_masks = sign_masks.view(padded_rows // tile_n, tile_n, chunk_cols).permute(0, 2, 1).contiguous()

    row_scales = _bitnet_row_scales(scale_values, layout_header, segment_offsets)
    if padded_rows > int(row_scales.shape[0]):
        row_scales = torch.cat(
            (
                row_scales,
                torch.zeros(
                    padded_rows - int(row_scales.shape[0]),
                    device=row_scales.device,
                    dtype=torch.float32,
                ),
            ),
            dim=0,
        )
    row_scales = row_scales.view(padded_rows // tile_n, tile_n).contiguous()
    return nz_masks, sign_masks, row_scales


def _bitnet_row_scale_symmetric(
    row: torch.Tensor,
    *,
    calibration: str,
    percentile: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    method = str(calibration).strip().lower()
    if method == "absmax":
        clip = row.abs().amax()
    elif method == "percentile":
        clip = percentile_scale(row, p=float(percentile))
    elif method == "mse":
        clip = mse_scale(row)
    else:
        raise ValueError(f"Unknown BitNet calibration method: {calibration}")
    return clip.clamp_min(eps).to(dtype=torch.float32)


def _bitnet_quantize_rows(
    weight: torch.Tensor,
    *,
    calibration: str,
    percentile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_f = weight.float()
    method = str(calibration).strip().lower()
    if method == "absmax":
        row_scales = weight_f.abs().amax(dim=1).clamp_min(1e-8).to(dtype=torch.float32)
    else:
        row_scales = torch.stack(
            [
                _bitnet_row_scale_symmetric(row, calibration=method, percentile=percentile)
                for row in weight_f
            ],
            dim=0,
        )
    q = torch.round(weight_f / _reshape_channel_values(row_scales, weight_f, 0)).clamp_(-1, 1).to(dtype=torch.int8)
    return q, row_scales


def _pack_bitnet_quantized(
    qweight: torch.Tensor,
    *,
    scale_values: torch.Tensor,
    scale_granularity: int,
    scale_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if qweight.ndim != 2:
        raise ValueError("BitNet qweight must be rank-2")
    logical_out, logical_in = int(qweight.shape[0]), int(qweight.shape[1])
    padded_out = ((logical_out + 15) // 16) * 16
    padded_in = ((logical_in + 31) // 32) * 32
    codes = torch.ones((padded_out, padded_in), device=qweight.device, dtype=torch.uint8)
    codes[:logical_out, :logical_in] = qweight.to(dtype=torch.int16).clamp_(-1, 1).add(1).to(dtype=torch.uint8)
    packed_weight = (
        codes[:, 0::4]
        | (codes[:, 1::4] << 2)
        | (codes[:, 2::4] << 4)
        | (codes[:, 3::4] << 6)
    ).contiguous()
    packed_scale_values = scale_values.to(device=qweight.device, dtype=torch.float32).contiguous()
    layout_header = torch.tensor(
        [
            1,
            16,
            32,
            logical_out,
            logical_in,
            padded_out,
            padded_in,
            int(scale_granularity),
            int(scale_group_size),
            1,
            80,
            1,
            0,
        ],
        device=qweight.device,
        dtype=torch.int32,
    )
    segment_offsets = torch.tensor([0, logical_out], device=qweight.device, dtype=torch.int32)
    return packed_weight, packed_scale_values, layout_header, segment_offsets


def _symmetric_per_channel_weight_quantize_int4(
    weight: torch.Tensor, ch_axis: int = 0, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to packed int4 with symmetric per-channel scaling."""
    assert weight.dtype.is_floating_point
    axis = _normalize_axis(ch_axis, weight.ndim)
    reduce_dims = _channel_reduce_dims(weight.ndim, axis)
    max_abs = weight.abs().amax(dim=reduce_dims, keepdim=False).clamp_min(eps).to(dtype=torch.float32)
    scale = 7.0 / _reshape_channel_values(max_abs, weight, axis)
    q = torch.round(weight.float() * scale).clamp_(-7, 7).to(dtype=torch.int8)
    packed = _pack_int4_signed(q)
    inv_scale = (max_abs / 7.0).to(dtype=torch.float32)
    return packed, inv_scale


def calibrate_int8_scales(
    weight: torch.Tensor,
    method: str = "absmax",
    ch_axis: int = 0,
    p: float = 0.999,
) -> torch.Tensor:
    """Return per-channel dequant multipliers using tensor.numerics calibrations.

    Methods:
    - "absmax": absmax / 127
    - "percentile": percentile(|w|, p) / 127
    - "mse": mse_scale(|w|) / 127
    """
    axis = _normalize_axis(ch_axis, weight.ndim)
    reduce_dims = _channel_reduce_dims(weight.ndim, axis)
    if method == "absmax":
        clip = weight.abs().amax(dim=reduce_dims, keepdim=False)
        return (clip.clamp_min(1e-8) / 127.0).to(dtype=torch.float32)
    if method == "percentile":
        # percentile_scale returns scalar if dim=None; provide dim for per-channel
        inv = []
        for idx in range(weight.size(axis)):
            slicer = [slice(None)] * weight.ndim
            slicer[axis] = idx
            wch = weight[tuple(slicer)]
            s = percentile_scale(wch, p=float(p))
            inv.append((s.clamp_min(1e-8) / 127.0).to(dtype=torch.float32))
        return torch.stack(inv, dim=0)
    if method == "mse":
        inv = []
        for idx in range(weight.size(axis)):
            slicer = [slice(None)] * weight.ndim
            slicer[axis] = idx
            wch = weight[tuple(slicer)]
            s = mse_scale(wch)
            inv.append((s.clamp_min(1e-8) / 127.0).to(dtype=torch.float32))
        return torch.stack(inv, dim=0)
    raise ValueError("Unknown calibration method")


def calibrate_int4_scales(
    weight: torch.Tensor,
    method: str = "absmax",
    ch_axis: int = 0,
    p: float = 0.999,
) -> torch.Tensor:
    """Return per-channel dequant multipliers for symmetric int4 weight quantization."""
    axis = _normalize_axis(ch_axis, weight.ndim)
    reduce_dims = _channel_reduce_dims(weight.ndim, axis)
    if method == "absmax":
        clip = weight.abs().amax(dim=reduce_dims, keepdim=False)
        return (clip.clamp_min(1e-8) / 7.0).to(dtype=torch.float32)
    if method == "percentile":
        inv = []
        for idx in range(weight.size(axis)):
            slicer = [slice(None)] * weight.ndim
            slicer[axis] = idx
            wch = weight[tuple(slicer)]
            s = percentile_scale(wch, p=float(p))
            inv.append((s.clamp_min(1e-8) / 7.0).to(dtype=torch.float32))
        return torch.stack(inv, dim=0)
    if method == "mse":
        inv = []
        for idx in range(weight.size(axis)):
            slicer = [slice(None)] * weight.ndim
            slicer[axis] = idx
            wch = weight[tuple(slicer)]
            s = mse_scale(wch)
            inv.append((s.clamp_min(1e-8) / 7.0).to(dtype=torch.float32))
        return torch.stack(inv, dim=0)
    raise ValueError("Unknown calibration method")


def calibrate_nf4_scales(
    weight: torch.Tensor,
    method: str = "absmax",
    ch_axis: int = 0,
    p: float = 0.999,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return per-channel dequant multipliers for NF4 codebook quantization."""
    axis = _normalize_axis(ch_axis, weight.ndim)
    reduce_dims = _channel_reduce_dims(weight.ndim, axis)
    if method == "absmax":
        clip = weight.abs().amax(dim=reduce_dims, keepdim=False)
        return clip.clamp_min(eps).to(dtype=torch.float32)
    if method == "percentile":
        out = []
        for idx in range(weight.size(axis)):
            slicer = [slice(None)] * weight.ndim
            slicer[axis] = idx
            wch = weight[tuple(slicer)]
            out.append(percentile_scale(wch, p=float(p)).clamp_min(eps).to(dtype=torch.float32))
        return torch.stack(out, dim=0)
    if method == "mse":
        out = []
        for idx in range(weight.size(axis)):
            slicer = [slice(None)] * weight.ndim
            slicer[axis] = idx
            wch = weight[tuple(slicer)]
            out.append(mse_scale(wch).clamp_min(eps).to(dtype=torch.float32))
        return torch.stack(out, dim=0)
    raise ValueError("Unknown calibration method")


def _dequantize_int8_per_channel(
    qweight: torch.Tensor, inv_scale: torch.Tensor, ch_axis: int = 0
) -> torch.Tensor:
    scale = _reshape_channel_values(inv_scale, qweight, ch_axis).to(device=qweight.device, dtype=torch.float32)
    return qweight.to(dtype=torch.float32) * scale


def _dequantize_int4_per_channel(
    qweight_packed: torch.Tensor,
    inv_scale: torch.Tensor,
    original_last_dim: int,
    ch_axis: int = 0,
) -> torch.Tensor:
    qweight = _unpack_int4_signed(qweight_packed, original_last_dim)
    scale = _reshape_channel_values(inv_scale, qweight, ch_axis).to(device=qweight.device, dtype=torch.float32)
    return qweight.to(dtype=torch.float32) * scale


def _dequantize_nf4_per_channel(
    qweight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    original_last_dim: int,
    ch_axis: int = 0,
) -> torch.Tensor:
    qcodes = _unpack_nf4_codes(qweight_packed, original_last_dim)
    scale = _reshape_channel_values(weight_scale, qcodes, ch_axis).to(device=qcodes.device, dtype=torch.float32)
    return runtime_nf4_dequantize(qcodes, scale).to(dtype=torch.float32)


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


def calibrate_fp8_scale(
    weight: torch.Tensor,
    method: str = "absmax",
    p: float = 0.999,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a scalar dequant multiplier for fake-FP8 weight quantization."""
    if method == "absmax":
        clip = weight.abs().amax()
    elif method == "percentile":
        clip = percentile_scale(weight, p=float(p))
    elif method == "mse":
        clip = mse_scale(weight)
    else:
        raise ValueError("Unknown calibration method")
    return clip.clamp_min(eps).to(dtype=torch.float32)


def _flatten_calibration_inputs(
    calibration_inputs: torch.Tensor | None,
    *,
    in_features: int,
    max_samples: int = 2048,
) -> torch.Tensor | None:
    if calibration_inputs is None:
        return None
    x = calibration_inputs.detach()
    if not isinstance(x, torch.Tensor):
        raise TypeError("calibration_inputs must be a tensor")
    if x.ndim < 2 or int(x.shape[-1]) != int(in_features):
        raise ValueError("calibration_inputs must have last dimension equal to in_features")
    x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float32)
    if max_samples > 0 and x.shape[0] > max_samples:
        x = x[:max_samples]
    return x.contiguous()


def _quant_max(bits: int) -> int:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return (1 << (bits - 1)) - 1


def calibrate_activation_scale(
    inputs: torch.Tensor,
    *,
    method: str = "absmax",
    bits: int = 8,
    p: float = 0.999,
    eps: float = 1e-8,
) -> torch.Tensor:
    qmax = float(_quant_max(bits))
    x = inputs.detach().float()
    if method == "absmax":
        clip = x.abs().amax()
    elif method == "percentile":
        clip = percentile_scale(x, p=float(p))
    elif method == "mse":
        clip = mse_scale(x)
    else:
        raise ValueError("Unknown activation calibration method")
    return (clip.clamp_min(eps) / qmax).to(dtype=torch.float32)


def fake_quantize_activation(
    x: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    bits: int = 8,
) -> torch.Tensor:
    qmax = float(_quant_max(bits))
    scale_t = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, device=x.device, dtype=torch.float32)
    scale_t = scale_t.to(device=x.device, dtype=torch.float32).clamp_min(1e-8)
    while scale_t.ndim < x.ndim:
        scale_t = scale_t.unsqueeze(0)
    q = torch.round(x.float() / scale_t).clamp_(-qmax, qmax)
    return (q * scale_t).to(dtype=x.dtype)


def _normalize_pre_scale(scale: torch.Tensor, *, eps: float = 1e-4, clamp: float = 16.0) -> torch.Tensor:
    scale = scale.detach().float().clamp_min(eps)
    geom = scale.log().mean().exp()
    normalized = scale / geom.clamp_min(eps)
    return normalized.clamp(min=1.0 / clamp, max=clamp).to(dtype=torch.float32)


def _pre_scale_active(pre_scale: torch.Tensor) -> bool:
    return bool((pre_scale.detach().float() - 1.0).abs().max().item() > 1e-6)


def _apply_pre_scale_to_weight(weight: torch.Tensor, pre_scale: torch.Tensor) -> torch.Tensor:
    return weight * pre_scale.to(device=weight.device, dtype=torch.float32).view(1, -1)


def _undo_pre_scale_from_weight(weight: torch.Tensor, pre_scale: torch.Tensor) -> torch.Tensor:
    return weight / pre_scale.to(device=weight.device, dtype=torch.float32).view(1, -1)


def _apply_pre_scale_to_input(x: torch.Tensor, pre_scale: torch.Tensor) -> torch.Tensor:
    view_shape = [1] * x.ndim
    view_shape[-1] = pre_scale.numel()
    return x / pre_scale.to(device=x.device, dtype=x.dtype).view(*view_shape)


def _apply_activation_quantization(
    x: torch.Tensor,
    *,
    mode: str,
    act_scale: torch.Tensor,
    bits: int,
    method: str,
    percentile: float,
) -> torch.Tensor:
    mode_name = str(mode).strip().lower()
    if mode_name in {"", "none", "off"}:
        return x
    if mode_name == "dynamic_int8":
        scale = calibrate_activation_scale(x, method=method, bits=bits, p=percentile)
        return fake_quantize_activation(x, scale, bits=bits)
    if mode_name == "static_int8":
        return fake_quantize_activation(x, act_scale, bits=bits)
    raise ValueError(f"Unknown activation quantization mode: {mode}")


def _inject_missing_state_defaults(state_dict, defaults: dict[str, torch.Tensor]) -> list[str]:
    inserted: list[str] = []
    for key, tensor in defaults.items():
        if key in state_dict:
            continue
        state_dict[key] = tensor.detach().clone()
        inserted.append(key)
    return inserted


def _remove_injected_state_defaults(state_dict, inserted: list[str]) -> None:
    for key in inserted:
        state_dict.pop(key, None)


def _int8_quantize_dequantize(
    weight: torch.Tensor,
    *,
    calibration: str,
    percentile: float,
) -> torch.Tensor:
    if calibration == "absmax":
        qweight, inv_scale = _symmetric_per_channel_weight_quantize_int8(weight, ch_axis=0)
    else:
        inv_scale = calibrate_int8_scales(weight, method=calibration, ch_axis=0, p=percentile)
        scale = (1.0 / _reshape_channel_values(inv_scale, weight, 0)).to(dtype=torch.float32)
        qweight = torch.round(weight.float() * scale).clamp_(-127, 127).to(dtype=torch.int8)
    return _dequantize_int8_per_channel(qweight, inv_scale, ch_axis=0)


def _int4_quantize_dequantize(
    weight: torch.Tensor,
    *,
    calibration: str,
    percentile: float,
) -> torch.Tensor:
    if calibration == "absmax":
        qweight, inv_scale = _symmetric_per_channel_weight_quantize_int4(weight, ch_axis=0)
    else:
        inv_scale = calibrate_int4_scales(weight, method=calibration, ch_axis=0, p=percentile)
        scale = (1.0 / _reshape_channel_values(inv_scale, weight, 0)).to(dtype=torch.float32)
        qweight = _pack_int4_signed(torch.round(weight.float() * scale).clamp_(-7, 7).to(dtype=torch.int8))
    return _dequantize_int4_per_channel(qweight, inv_scale, original_last_dim=int(weight.shape[-1]), ch_axis=0)


def _nf4_quantize_dequantize(
    weight: torch.Tensor,
    *,
    calibration: str,
    percentile: float,
) -> torch.Tensor:
    weight_scale = calibrate_nf4_scales(weight, method=calibration, ch_axis=0, p=percentile)
    qweight, _ = runtime_nf4_quantize(weight, _reshape_channel_values(weight_scale, weight, 0))
    packed = _pack_nf4_codes(qweight)
    return _dequantize_nf4_per_channel(
        packed,
        weight_scale,
        original_last_dim=int(weight.shape[-1]),
        ch_axis=0,
    ).to(dtype=weight.dtype)


def _bitnet_quantize_dequantize(
    weight: torch.Tensor,
    *,
    calibration: str = "absmax",
    percentile: float = 0.999,
) -> torch.Tensor:
    method = str(calibration).strip().lower()
    if method == "absmax":
        packed_weight, scale_values, layout_header, segment_offsets = runtime_pack_bitnet_weight(weight)
        return _dequantize_bitnet_weight(
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
            dtype=weight.dtype,
        )
    qweight, row_scales = _bitnet_quantize_rows(
        weight,
        calibration=method,
        percentile=float(percentile),
    )
    return qweight.to(dtype=torch.float32).mul(row_scales.unsqueeze(-1)).to(dtype=weight.dtype)


def _bitnet_runtime_row_codes_and_scale(
    weight: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError("runtime-row BitNet quantization expects a rank-2 weight")
    weight_f = weight.float()
    row_scale = weight_f.abs().mean(dim=-1, keepdim=True).clamp_min(float(eps))
    qweight = torch.round(weight_f / row_scale).clamp_(-1, 1).to(dtype=torch.int8)
    return qweight, row_scale.squeeze(-1).to(dtype=torch.float32)


def _bitnet_runtime_row_ste_weight(
    weight: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    qweight, row_scale = _bitnet_runtime_row_codes_and_scale(weight, eps=eps)
    quantized = qweight.to(dtype=torch.float32).mul(row_scale.unsqueeze(-1))
    return weight + (quantized.to(device=weight.device, dtype=weight.dtype) - weight).detach()


def _fp8_quantize_dequantize(
    weight: torch.Tensor,
    *,
    calibration: str,
    percentile: float,
    fp8_dtype: str,
) -> torch.Tensor:
    scale = calibrate_fp8_scale(weight, method=calibration, p=percentile)
    quantized = fake_quantize_fp8(weight.float() / scale, dtype=fp8_dtype)
    return (quantized.to(dtype=torch.float32) * scale.to(dtype=torch.float32)).to(dtype=weight.dtype)


def awq_optimize_pre_scale(
    weight: torch.Tensor,
    calibration_inputs: torch.Tensor,
    *,
    quantize_dequantize,
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> torch.Tensor:
    x = _flatten_calibration_inputs(calibration_inputs, in_features=int(weight.shape[-1]))
    if x is None or x.numel() == 0:
        return torch.ones(weight.shape[-1], device=weight.device, dtype=torch.float32)
    ref = x.to(device=weight.device, dtype=torch.float32).matmul(weight.float().t())
    act_scale = x.abs().mean(dim=0).clamp_min(1e-6).to(device=weight.device, dtype=torch.float32)
    best_scale = torch.ones_like(act_scale)
    best_error = torch.tensor(float("inf"), device=weight.device)
    for alpha in alpha_grid:
        pre_scale = _normalize_pre_scale(act_scale.pow(float(alpha)))
        weight_scaled = _apply_pre_scale_to_weight(weight.float(), pre_scale)
        weight_q = quantize_dequantize(weight_scaled)
        pred = _apply_pre_scale_to_input(x.to(device=weight.device, dtype=torch.float32), pre_scale).matmul(weight_q.t())
        err = (pred - ref).pow(2).mean()
        if err < best_error:
            best_error = err
            best_scale = pre_scale
    return best_scale


def _row_inv_scale_symmetric(
    row: torch.Tensor,
    *,
    bits: int,
    calibration: str,
    percentile: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    qmax = float(_quant_max(bits))
    if calibration == "absmax":
        clip = row.abs().amax()
    elif calibration == "percentile":
        clip = percentile_scale(row, p=float(percentile))
    elif calibration == "mse":
        clip = mse_scale(row)
    else:
        raise ValueError("Unknown calibration method")
    return (clip.clamp_min(eps) / qmax).to(dtype=torch.float32)


def _row_scale_nf4(
    row: torch.Tensor,
    *,
    calibration: str,
    percentile: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if calibration == "absmax":
        clip = row.abs().amax()
    elif calibration == "percentile":
        clip = percentile_scale(row, p=float(percentile))
    elif calibration == "mse":
        clip = mse_scale(row)
    else:
        raise ValueError("Unknown calibration method")
    return clip.clamp_min(eps).to(dtype=torch.float32)


def _gptq_dequantize_rows(
    weight: torch.Tensor,
    calibration_inputs: torch.Tensor,
    *,
    bits: int,
    calibration: str,
    percentile: float,
    damp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _flatten_calibration_inputs(calibration_inputs, in_features=int(weight.shape[-1]))
    if x is None or x.numel() == 0:
        raise ValueError("GPTQ optimization requires non-empty calibration_inputs")
    x = x.to(device=weight.device, dtype=torch.float32)
    h = x.t().matmul(x) / max(int(x.shape[0]), 1)
    diag_mean = h.diag().mean().clamp_min(1e-6)
    h = h + torch.eye(h.shape[0], device=h.device, dtype=h.dtype) * (float(damp) * diag_mean)
    h_inv = torch.linalg.inv(h)
    h_diag = h_inv.diag().clamp_min(1e-6)
    qmax = float(_quant_max(bits))
    dequant_rows = []
    inv_scales = []
    for row in weight.float():
        inv_scale = _row_inv_scale_symmetric(
            row,
            bits=bits,
            calibration=calibration,
            percentile=percentile,
        )
        work = row.clone()
        out = torch.empty_like(work)
        for idx in range(work.numel()):
            q = torch.round(work[idx] / inv_scale).clamp_(-qmax, qmax)
            dq = q * inv_scale
            out[idx] = dq
            if idx + 1 < work.numel():
                err = (work[idx] - dq) / h_diag[idx]
                work[idx + 1 :] -= err * h_inv[idx, idx + 1 :]
        dequant_rows.append(out)
        inv_scales.append(inv_scale)
    return torch.stack(dequant_rows, dim=0), torch.stack(inv_scales, dim=0)


def _gptq_dequantize_rows_nf4(
    weight: torch.Tensor,
    calibration_inputs: torch.Tensor,
    *,
    calibration: str,
    percentile: float,
    damp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _flatten_calibration_inputs(calibration_inputs, in_features=int(weight.shape[-1]))
    if x is None or x.numel() == 0:
        raise ValueError("GPTQ optimization requires non-empty calibration_inputs")
    x = x.to(device=weight.device, dtype=torch.float32)
    h = x.t().matmul(x) / max(int(x.shape[0]), 1)
    diag_mean = h.diag().mean().clamp_min(1e-6)
    h = h + torch.eye(h.shape[0], device=h.device, dtype=h.dtype) * (float(damp) * diag_mean)
    h_inv = torch.linalg.inv(h)
    h_diag = h_inv.diag().clamp_min(1e-6)
    codebook = runtime_nf4_dequantize(
        torch.arange(16, device=weight.device, dtype=torch.uint8),
        torch.tensor(1.0, device=weight.device, dtype=torch.float32),
    ).to(dtype=torch.float32)
    code_min = float(codebook[0].item())
    code_max = float(codebook[-1].item())
    dequant_rows = []
    weight_scales = []
    for row in weight.float():
        row_scale = _row_scale_nf4(
            row,
            calibration=calibration,
            percentile=percentile,
        )
        work = row.clone()
        out = torch.empty_like(work)
        for idx in range(work.numel()):
            norm = (work[idx] / row_scale).clamp_(code_min, code_max)
            code_idx = torch.argmin((codebook - norm).abs())
            dq = codebook[code_idx] * row_scale
            out[idx] = dq
            if idx + 1 < work.numel():
                err = (work[idx] - dq) / h_diag[idx]
                work[idx + 1 :] -= err * h_inv[idx, idx + 1 :]
        dequant_rows.append(out)
        weight_scales.append(row_scale)
    return torch.stack(dequant_rows, dim=0), torch.stack(weight_scales, dim=0)


@torch.no_grad()
def collect_linear_calibration_inputs(
    model: nn.Module,
    *model_args,
    model_kwargs: dict | None = None,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    max_samples: int = 2048,
) -> Dict[str, torch.Tensor]:
    include_list = None if include is None else list(include)
    exclude_list = None if exclude is None else list(exclude)
    collected: Dict[str, torch.Tensor] = {}
    hooks = []

    def _should_keep(name: str) -> bool:
        if include_list is not None and include_list and not any(pattern in name for pattern in include_list):
            return False
        if exclude_list is not None and any(pattern in name for pattern in exclude_list):
            return False
        return True

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not _should_keep(name):
            continue

        def _hook(inputs, *, module_name=name, in_features=module.in_features):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            flat = x.detach().reshape(-1, x.shape[-1]).to(dtype=torch.float32, device="cpu")
            if int(flat.shape[-1]) != int(in_features):
                return
            existing = collected.get(module_name)
            if existing is None:
                collected[module_name] = flat[:max_samples] if max_samples > 0 else flat
                return
            if max_samples > 0:
                remaining = max_samples - int(existing.shape[0])
                if remaining <= 0:
                    return
                flat = flat[:remaining]
            if flat.numel() > 0:
                collected[module_name] = torch.cat((existing, flat), dim=0)

        hooks.append(module.register_forward_pre_hook(lambda mod, inp, hook=_hook: hook(inp)))
    try:
        model(*model_args, **({} if model_kwargs is None else model_kwargs))
    finally:
        for hook in hooks:
            hook.remove()
    return collected


class _AmaxTracker:
    def __init__(self, owner: "QuantizedLinearFP8") -> None:
        self._owner = owner

    def update(self, x: torch.Tensor) -> None:
        self._owner.amax_observed.copy_(x.detach().abs().amax().to(self._owner.amax_observed.device))


class QuantizedLinearInt8(nn.Module):
    """Weight-only INT8 linear with per-out-feature scaling.

    Quantization is performed per out_channel (dim=0) on the weight matrix.
    Forward pass uses a cached dequantized weight on the module device.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_calibration = "absmax"
        self.quant_percentile = 0.999
        self.weight_opt = "none"
        self.act_quant_mode = "none"
        self.act_quant_method = "absmax"
        self.act_quant_percentile = 0.999
        self.act_quant_bits = 8
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("inv_scale", torch.ones(out_features, dtype=torch.float32))
        self.register_buffer("spin_signs", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("spin_enabled_flag", torch.zeros((), dtype=torch.uint8))
        self.register_buffer("pre_scale", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("act_scale", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._cached_weight_key: tuple[object, ...] | None = None
        self._cached_weight: torch.Tensor | None = None
        self._cached_shared_int8_input_signature_key: tuple[object, ...] | None = None
        self._cached_shared_int8_input_signature: tuple[object, ...] | None = None

    def _invalidate_weight_cache(self) -> None:
        self._cached_weight_key = None
        self._cached_weight = None
        self._cached_shared_int8_input_signature_key = None
        self._cached_shared_int8_input_signature = None

    def _assign_quantized_state(self, qweight: torch.Tensor, inv_scale: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        self.qweight.copy_(qweight.to(device=self.qweight.device, dtype=self.qweight.dtype))
        self.inv_scale.copy_(inv_scale.to(device=self.inv_scale.device, dtype=self.inv_scale.dtype))
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
        self._invalidate_weight_cache()

    def _assign_float_state(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        calibration_inputs: torch.Tensor | None = None,
    ) -> None:
        weight = weight.float()
        if _spin_enabled(self.spin_enabled_flag):
            weight = apply_spin_transform(weight, self.spin_signs.to(device=weight.device))
        flat_inputs = _flatten_calibration_inputs(calibration_inputs, in_features=self.in_features)
        requested_weight_opt = str(self.weight_opt).strip().lower()
        pre_scale = torch.ones(self.in_features, device=weight.device, dtype=torch.float32)
        if requested_weight_opt == "awq" and flat_inputs is not None:
            pre_scale = awq_optimize_pre_scale(
                weight,
                flat_inputs.to(device=weight.device, dtype=torch.float32),
                quantize_dequantize=lambda w: _int8_quantize_dequantize(
                    w,
                    calibration=self.quant_calibration,
                    percentile=float(self.quant_percentile),
                ),
            )
        self.pre_scale.copy_(pre_scale.to(device=self.pre_scale.device, dtype=self.pre_scale.dtype))
        weight = _apply_pre_scale_to_weight(weight, pre_scale)
        if requested_weight_opt == "gptq" and flat_inputs is not None:
            dequant_weight, inv_s = _gptq_dequantize_rows(
                weight,
                flat_inputs.to(device=weight.device, dtype=torch.float32),
                bits=8,
                calibration=self.quant_calibration,
                percentile=float(self.quant_percentile),
            )
            q = torch.round(dequant_weight / _reshape_channel_values(inv_s, dequant_weight, 0)).clamp_(-127, 127).to(dtype=torch.int8)
        elif self.quant_calibration == "absmax":
            q, inv_s = _symmetric_per_channel_weight_quantize_int8(weight, ch_axis=0)
        else:
            inv_s = calibrate_int8_scales(
                weight,
                method=self.quant_calibration,
                ch_axis=0,
                p=float(self.quant_percentile),
            )
            scale = (1.0 / _reshape_channel_values(inv_s, weight, 0)).to(dtype=torch.float32)
            q = torch.round(weight * scale).clamp_(-127, 127).to(dtype=torch.int8)
        self._assign_quantized_state(q, inv_s, bias)
        if self.act_quant_mode == "static_int8" and flat_inputs is not None:
            act_inputs = flat_inputs.to(device=weight.device, dtype=torch.float32)
            if _spin_enabled(self.spin_enabled_flag):
                act_inputs = apply_spin_transform(act_inputs, self.spin_signs.to(device=weight.device))
            if _pre_scale_active(pre_scale):
                act_inputs = _apply_pre_scale_to_input(act_inputs, pre_scale)
            self.act_scale.copy_(
                calibrate_activation_scale(
                    act_inputs,
                    method=self.act_quant_method,
                    bits=int(self.act_quant_bits),
                    p=float(self.act_quant_percentile),
                ).to(device=self.act_scale.device, dtype=self.act_scale.dtype)
            )
        else:
            self.act_scale.fill_(1.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        qweight_key = prefix + "qweight"
        inv_scale_key = prefix + "inv_scale"
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        if qweight_key in state_dict and inv_scale_key in state_dict:
            inserted = _inject_missing_state_defaults(
                state_dict,
                {
                    prefix + "spin_signs": self.spin_signs,
                    prefix + "spin_enabled_flag": self.spin_enabled_flag,
                    prefix + "pre_scale": self.pre_scale,
                    prefix + "act_scale": self.act_scale,
                },
            )
            try:
                super()._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
            finally:
                _remove_injected_state_defaults(state_dict, inserted)
            self._invalidate_weight_cache()
            return
        if weight_key in state_dict:
            try:
                weight = state_dict.pop(weight_key)
                bias = state_dict.pop(bias_key, None)
                self._assign_float_state(weight, bias)
            except Exception as exc:
                error_msgs.append(f"While quantizing checkpoint tensor for {prefix[:-1] or prefix}: {exc}")
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._invalidate_weight_cache()

    def _dequantized_weight(self, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.qweight.device if device is None else torch.device(device)
        key = (
            str(target_device),
            str(dtype),
            tuple(self.qweight.shape),
            int(getattr(self.qweight, "_version", 0)),
            int(getattr(self.inv_scale, "_version", 0)),
            int(self.qweight.data_ptr()),
            int(self.inv_scale.data_ptr()),
        )
        if self._cached_weight_key != key or self._cached_weight is None:
            qweight = self.qweight if self.qweight.device == target_device else self.qweight.to(device=target_device)
            inv_scale = self.inv_scale if self.inv_scale.device == target_device else self.inv_scale.to(device=target_device)
            self._cached_weight = _dequantize_int8_per_channel(qweight, inv_scale, ch_axis=0).to(
                device=target_device,
                dtype=dtype,
            ).contiguous()
            self._cached_weight_key = key
        return self._cached_weight

    @property
    def weight(self) -> torch.Tensor:
        return self.runtime_weight()

    def runtime_weight(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target_dtype = torch.float32 if dtype is None else dtype
        target_device = self.qweight.device if device is None else torch.device(device)
        weight = self._dequantized_weight(target_dtype, device=target_device)
        if _pre_scale_active(self.pre_scale):
            weight = _undo_pre_scale_from_weight(weight, self.pre_scale.to(device=target_device))
        if _spin_enabled(self.spin_enabled_flag):
            return undo_spin_transform(weight, self.spin_signs.to(device=target_device))
        return weight

    def runtime_bias(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias.to(
            device=self.bias.device if device is None else torch.device(device),
            dtype=self.bias.dtype if dtype is None else dtype,
        )

    def runtime_signature(self):
        return (
            "int8_pc",
            _parameter_signature(self.qweight),
            _parameter_signature(self.inv_scale),
            str(self.weight_opt),
            int(self.spin_enabled_flag.item()),
            _parameter_signature(self.spin_signs),
            _parameter_signature(self.pre_scale),
            str(self.act_quant_mode),
            str(self.act_quant_method),
            int(self.act_quant_bits),
            _parameter_signature(self.act_scale),
            _parameter_signature(self.bias),
        )

    def runtime_supports_packed_backend(self, backend: str) -> bool:
        del backend
        return False

    def runtime_shared_int8_input_signature(self):
        mode_name = str(self.act_quant_mode).strip().lower()
        if int(self.act_quant_bits) != 8 or mode_name not in {"dynamic_int8", "static_int8"}:
            return None
        cache_key = (
            mode_name,
            int(self.act_quant_bits),
            int(self.spin_enabled_flag.item()),
            _parameter_signature(self.spin_signs),
            _parameter_signature(self.pre_scale),
            str(self.act_quant_method),
            float(self.act_quant_percentile),
            _parameter_signature(self.act_scale),
        )
        if (
            self._cached_shared_int8_input_signature_key != cache_key
            or self._cached_shared_int8_input_signature is None
        ):
            signature = (
                int(self.spin_enabled_flag.item()),
                tuple(self.spin_signs.detach().cpu().tolist()),
                tuple(self.pre_scale.detach().cpu().tolist()),
                mode_name,
                int(self.act_quant_bits),
            )
            if mode_name == "dynamic_int8":
                signature = signature + (
                    str(self.act_quant_method),
                    float(self.act_quant_percentile),
                )
            else:
                signature = signature + (
                    tuple(self.act_scale.detach().cpu().reshape(-1).tolist()),
                )
            self._cached_shared_int8_input_signature_key = cache_key
            self._cached_shared_int8_input_signature = signature
        return self._cached_shared_int8_input_signature

    def runtime_quantize_int8_input(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
        target_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        target_device = x.device
        x_local = x.to(device=target_device, dtype=target_dtype)
        if _spin_enabled(self.spin_enabled_flag):
            x_local = apply_spin_transform(x_local, self.spin_signs.to(device=target_device))
        if _pre_scale_active(self.pre_scale):
            x_local = _apply_pre_scale_to_input(x_local, self.pre_scale.to(device=target_device))
        mode_name = str(self.act_quant_mode).strip().lower()
        if int(self.act_quant_bits) != 8 or mode_name not in {"dynamic_int8", "static_int8"}:
            raise RuntimeError("runtime_quantize_int8_input requires activation_quant in {dynamic_int8, static_int8} with 8 bits")
        qx, row_scale = runtime_quantize_activation_int8_rowwise(
            x_local,
            scale=self.act_scale.to(device=target_device) if mode_name == "static_int8" else None,
            method=self.act_quant_method,
            percentile=float(self.act_quant_percentile),
        )
        return qx, row_scale, target_dtype

    def runtime_linear_from_quantized_input(
        self,
        qx: torch.Tensor,
        row_scale: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_dtype = torch.float32 if out_dtype is None else out_dtype
        bias = self.runtime_bias(dtype=target_dtype)
        return runtime_int8_linear_from_quantized_activation(
            qx,
            row_scale,
            self.qweight,
            self.inv_scale,
            bias,
            out_dtype=target_dtype,
        )

    @torch.no_grad()
    def from_float(
        self,
        module: nn.Linear,
        calibration: str = "absmax",
        percentile: float = 0.999,
        calibration_inputs: torch.Tensor | None = None,
        weight_opt: str = "none",
        activation_quant: str = "none",
        activation_quant_bits: int = 8,
        activation_quant_method: str = "absmax",
        activation_quant_percentile: float = 0.999,
        spin: bool = False,
        spin_random: bool = True,
        spin_seed: int = 0,
    ) -> "QuantizedLinearInt8":
        assert module.in_features == self.in_features and module.out_features == self.out_features
        self.quant_calibration = str(calibration)
        self.quant_percentile = float(percentile)
        self.weight_opt = str(weight_opt)
        self.act_quant_mode = str(activation_quant)
        self.act_quant_bits = int(activation_quant_bits)
        self.act_quant_method = str(activation_quant_method)
        self.act_quant_percentile = float(activation_quant_percentile)
        _configure_spin_state(
            self.spin_enabled_flag,
            self.spin_signs,
            enabled=bool(spin),
            random_signs=bool(spin_random),
            seed=int(spin_seed),
        )
        self._assign_float_state(module.weight.data, None if module.bias is None else module.bias.data, calibration_inputs)
        return self

    def runtime_linear(self, x: torch.Tensor, *, backend: str | None = None) -> torch.Tensor:
        target_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        target_device = x.device
        mode_name = str(self.act_quant_mode).strip().lower()
        if int(self.act_quant_bits) == 8 and mode_name in {"dynamic_int8", "static_int8"}:
            x_local = x.to(device=target_device, dtype=target_dtype)
            if _spin_enabled(self.spin_enabled_flag):
                x_local = apply_spin_transform(x_local, self.spin_signs.to(device=target_device))
            if _pre_scale_active(self.pre_scale):
                x_local = _apply_pre_scale_to_input(x_local, self.pre_scale.to(device=target_device))
            return runtime_int8_linear(
                x_local,
                self.qweight,
                self.inv_scale,
                bias=self.runtime_bias(dtype=target_dtype, device=target_device),
                act_scale=self.act_scale.to(device=target_device) if mode_name == "static_int8" else None,
                act_method=self.act_quant_method,
                act_percentile=float(self.act_quant_percentile),
            )
        x_local = x.to(device=target_device, dtype=target_dtype)
        if _spin_enabled(self.spin_enabled_flag):
            x_local = apply_spin_transform(x_local, self.spin_signs.to(device=target_device))
        if _pre_scale_active(self.pre_scale):
            x_local = _apply_pre_scale_to_input(x_local, self.pre_scale.to(device=target_device))
        bias = self.runtime_bias(dtype=target_dtype, device=target_device)
        x_local = _apply_activation_quantization(
            x_local,
            mode=self.act_quant_mode,
            act_scale=self.act_scale.to(device=target_device),
            bits=int(self.act_quant_bits),
            method=self.act_quant_method,
            percentile=float(self.act_quant_percentile),
        )
        w = self._dequantized_weight(target_dtype, device=target_device)
        if backend is None:
            return runtime_linear(x_local, w, bias)
        return runtime_linear(x_local, w, bias, backend=backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.runtime_linear(x)


class QuantizedLinearInt4(nn.Module):
    """Weight-only packed INT4 linear with per-out-feature scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_calibration = "absmax"
        self.quant_percentile = 0.999
        self.weight_opt = "none"
        self.act_quant_mode = "none"
        self.act_quant_method = "absmax"
        self.act_quant_percentile = 0.999
        self.act_quant_bits = 8
        self.register_buffer("qweight_packed", torch.empty(out_features, (in_features + 1) // 2, dtype=torch.uint8))
        self.register_buffer("inv_scale", torch.ones(out_features, dtype=torch.float32))
        self.register_buffer("spin_signs", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("spin_enabled_flag", torch.zeros((), dtype=torch.uint8))
        self.register_buffer("pre_scale", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("act_scale", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._cached_weight_key: tuple[object, ...] | None = None
        self._cached_weight: torch.Tensor | None = None

    def _invalidate_weight_cache(self) -> None:
        self._cached_weight_key = None
        self._cached_weight = None

    def _assign_quantized_state(
        self,
        qweight_packed: torch.Tensor,
        inv_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.qweight_packed.copy_(qweight_packed.to(device=self.qweight_packed.device, dtype=self.qweight_packed.dtype))
        self.inv_scale.copy_(inv_scale.to(device=self.inv_scale.device, dtype=self.inv_scale.dtype))
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
        self._invalidate_weight_cache()

    def _assign_float_state(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        calibration_inputs: torch.Tensor | None = None,
    ) -> None:
        weight_local = weight.float()
        if _spin_enabled(self.spin_enabled_flag):
            weight_local = apply_spin_transform(weight_local, self.spin_signs.to(device=weight.device))
        flat_inputs = _flatten_calibration_inputs(calibration_inputs, in_features=self.in_features)
        requested_weight_opt = str(self.weight_opt).strip().lower()
        pre_scale = torch.ones(self.in_features, device=weight_local.device, dtype=torch.float32)
        if requested_weight_opt == "awq" and flat_inputs is not None:
            pre_scale = awq_optimize_pre_scale(
                weight_local,
                flat_inputs.to(device=weight_local.device, dtype=torch.float32),
                quantize_dequantize=lambda w: _int4_quantize_dequantize(
                    w,
                    calibration=self.quant_calibration,
                    percentile=float(self.quant_percentile),
                ),
            )
        self.pre_scale.copy_(pre_scale.to(device=self.pre_scale.device, dtype=self.pre_scale.dtype))
        weight_local = _apply_pre_scale_to_weight(weight_local, pre_scale)
        if requested_weight_opt == "gptq" and flat_inputs is not None:
            dequant_weight, inv_s = _gptq_dequantize_rows(
                weight_local,
                flat_inputs.to(device=weight_local.device, dtype=torch.float32),
                bits=4,
                calibration=self.quant_calibration,
                percentile=float(self.quant_percentile),
            )
            q = _pack_int4_signed(
                torch.round(dequant_weight / _reshape_channel_values(inv_s, dequant_weight, 0)).clamp_(-7, 7).to(dtype=torch.int8)
            )
        elif self.quant_calibration == "absmax":
            q, inv_s = _symmetric_per_channel_weight_quantize_int4(weight_local, ch_axis=0)
        else:
            inv_s = calibrate_int4_scales(
                weight_local,
                method=self.quant_calibration,
                ch_axis=0,
                p=float(self.quant_percentile),
            )
            scale = (1.0 / _reshape_channel_values(inv_s, weight_local, 0)).to(dtype=torch.float32)
            q = _pack_int4_signed(torch.round(weight_local * scale).clamp_(-7, 7).to(dtype=torch.int8))
        self._assign_quantized_state(q, inv_s, bias)
        if self.act_quant_mode == "static_int8" and flat_inputs is not None:
            act_inputs = flat_inputs.to(device=weight_local.device, dtype=torch.float32)
            if _spin_enabled(self.spin_enabled_flag):
                act_inputs = apply_spin_transform(act_inputs, self.spin_signs.to(device=weight_local.device))
            if _pre_scale_active(pre_scale):
                act_inputs = _apply_pre_scale_to_input(act_inputs, pre_scale)
            self.act_scale.copy_(
                calibrate_activation_scale(
                    act_inputs,
                    method=self.act_quant_method,
                    bits=int(self.act_quant_bits),
                    p=float(self.act_quant_percentile),
                ).to(device=self.act_scale.device, dtype=self.act_scale.dtype)
            )
        else:
            self.act_scale.fill_(1.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        qweight_packed_key = prefix + "qweight_packed"
        inv_scale_key = prefix + "inv_scale"
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        if qweight_packed_key in state_dict and inv_scale_key in state_dict:
            inserted = _inject_missing_state_defaults(
                state_dict,
                {
                    prefix + "spin_signs": self.spin_signs,
                    prefix + "spin_enabled_flag": self.spin_enabled_flag,
                    prefix + "pre_scale": self.pre_scale,
                    prefix + "act_scale": self.act_scale,
                },
            )
            try:
                super()._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
            finally:
                _remove_injected_state_defaults(state_dict, inserted)
            self._invalidate_weight_cache()
            return
        if weight_key in state_dict:
            try:
                weight = state_dict.pop(weight_key)
                bias = state_dict.pop(bias_key, None)
                self._assign_float_state(weight, bias)
            except Exception as exc:
                error_msgs.append(f"While quantizing checkpoint tensor for {prefix[:-1] or prefix}: {exc}")
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._invalidate_weight_cache()

    def _dequantized_weight(self, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.qweight_packed.device if device is None else torch.device(device)
        key = (
            str(target_device),
            str(dtype),
            tuple(self.qweight_packed.shape),
            int(getattr(self.qweight_packed, "_version", 0)),
            int(getattr(self.inv_scale, "_version", 0)),
            int(self.qweight_packed.data_ptr()),
            int(self.inv_scale.data_ptr()),
        )
        if self._cached_weight_key != key or self._cached_weight is None:
            qweight_packed = (
                self.qweight_packed if self.qweight_packed.device == target_device else self.qweight_packed.to(device=target_device)
            )
            inv_scale = self.inv_scale if self.inv_scale.device == target_device else self.inv_scale.to(device=target_device)
            self._cached_weight = _dequantize_int4_per_channel(
                qweight_packed,
                inv_scale,
                original_last_dim=self.in_features,
                ch_axis=0,
            ).to(device=target_device, dtype=dtype).contiguous()
            self._cached_weight_key = key
        return self._cached_weight

    @property
    def weight(self) -> torch.Tensor:
        return self.runtime_weight()

    def runtime_weight(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target_dtype = torch.float32 if dtype is None else dtype
        target_device = self.qweight_packed.device if device is None else torch.device(device)
        weight = self._dequantized_weight(target_dtype, device=target_device)
        if _pre_scale_active(self.pre_scale):
            weight = _undo_pre_scale_from_weight(weight, self.pre_scale.to(device=target_device))
        if _spin_enabled(self.spin_enabled_flag):
            return undo_spin_transform(weight, self.spin_signs.to(device=target_device))
        return weight

    def runtime_bias(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias.to(
            device=self.bias.device if device is None else torch.device(device),
            dtype=self.bias.dtype if dtype is None else dtype,
        )

    def runtime_signature(self):
        return (
            "int4_pc_packed",
            _parameter_signature(self.qweight_packed),
            _parameter_signature(self.inv_scale),
            str(self.weight_opt),
            int(self.spin_enabled_flag.item()),
            _parameter_signature(self.spin_signs),
            _parameter_signature(self.pre_scale),
            str(self.act_quant_mode),
            str(self.act_quant_method),
            int(self.act_quant_bits),
            _parameter_signature(self.act_scale),
            _parameter_signature(self.bias),
        )

    def runtime_supports_packed_backend(self, backend: str) -> bool:
        del backend
        return False

    @torch.no_grad()
    def from_float(
        self,
        module: nn.Linear,
        calibration: str = "absmax",
        percentile: float = 0.999,
        calibration_inputs: torch.Tensor | None = None,
        weight_opt: str = "none",
        activation_quant: str = "none",
        activation_quant_bits: int = 8,
        activation_quant_method: str = "absmax",
        activation_quant_percentile: float = 0.999,
        spin: bool = False,
        spin_random: bool = True,
        spin_seed: int = 0,
    ) -> "QuantizedLinearInt4":
        assert module.in_features == self.in_features and module.out_features == self.out_features
        self.quant_calibration = str(calibration)
        self.quant_percentile = float(percentile)
        self.weight_opt = str(weight_opt)
        self.act_quant_mode = str(activation_quant)
        self.act_quant_bits = int(activation_quant_bits)
        self.act_quant_method = str(activation_quant_method)
        self.act_quant_percentile = float(activation_quant_percentile)
        _configure_spin_state(
            self.spin_enabled_flag,
            self.spin_signs,
            enabled=bool(spin),
            random_signs=bool(spin_random),
            seed=int(spin_seed),
        )
        self._assign_float_state(module.weight.data, None if module.bias is None else module.bias.data, calibration_inputs)
        return self

    def runtime_linear(self, x: torch.Tensor, *, backend: str | None = None) -> torch.Tensor:
        del backend
        target_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        target_device = x.device
        x_local = x.to(device=target_device, dtype=target_dtype)
        if _spin_enabled(self.spin_enabled_flag):
            x_local = apply_spin_transform(x_local, self.spin_signs.to(device=target_device))
        if _pre_scale_active(self.pre_scale):
            x_local = _apply_pre_scale_to_input(x_local, self.pre_scale.to(device=target_device))
        x_local = _apply_activation_quantization(
            x_local,
            mode=self.act_quant_mode,
            act_scale=self.act_scale.to(device=target_device),
            bits=int(self.act_quant_bits),
            method=self.act_quant_method,
            percentile=float(self.act_quant_percentile),
        )
        bias = self.runtime_bias(dtype=target_dtype, device=target_device)
        return runtime_int4_linear(
            x_local,
            self.qweight_packed,
            self.inv_scale,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.runtime_linear(x)


class QuantizedLinearNF4(nn.Module):
    """Weight-only packed NF4 linear with per-out-feature scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_calibration = "absmax"
        self.quant_percentile = 0.999
        self.weight_opt = "none"
        self.act_quant_mode = "none"
        self.act_quant_method = "absmax"
        self.act_quant_percentile = 0.999
        self.act_quant_bits = 8
        self.register_buffer("qweight_packed", torch.empty(out_features, (in_features + 1) // 2, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float32))
        self.register_buffer("spin_signs", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("spin_enabled_flag", torch.zeros((), dtype=torch.uint8))
        self.register_buffer("pre_scale", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("act_scale", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._cached_weight_key: tuple[object, ...] | None = None
        self._cached_weight: torch.Tensor | None = None

    def _invalidate_weight_cache(self) -> None:
        self._cached_weight_key = None
        self._cached_weight = None

    def _assign_quantized_state(
        self,
        qweight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.qweight_packed.copy_(qweight_packed.to(device=self.qweight_packed.device, dtype=self.qweight_packed.dtype))
        self.weight_scale.copy_(weight_scale.to(device=self.weight_scale.device, dtype=self.weight_scale.dtype))
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
        self._invalidate_weight_cache()

    def _assign_float_state(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        calibration_inputs: torch.Tensor | None = None,
    ) -> None:
        weight_local = weight.float()
        if _spin_enabled(self.spin_enabled_flag):
            weight_local = apply_spin_transform(weight_local, self.spin_signs.to(device=weight.device))
        flat_inputs = _flatten_calibration_inputs(calibration_inputs, in_features=self.in_features)
        requested_weight_opt = str(self.weight_opt).strip().lower()
        pre_scale = torch.ones(self.in_features, device=weight_local.device, dtype=torch.float32)
        if requested_weight_opt == "awq" and flat_inputs is not None:
            pre_scale = awq_optimize_pre_scale(
                weight_local,
                flat_inputs.to(device=weight_local.device, dtype=torch.float32),
                quantize_dequantize=lambda w: _nf4_quantize_dequantize(
                    w,
                    calibration=self.quant_calibration,
                    percentile=float(self.quant_percentile),
                ),
            )
        self.pre_scale.copy_(pre_scale.to(device=self.pre_scale.device, dtype=self.pre_scale.dtype))
        weight_local = _apply_pre_scale_to_weight(weight_local, pre_scale)
        if requested_weight_opt == "gptq" and flat_inputs is not None:
            dequant_weight, weight_scale = _gptq_dequantize_rows_nf4(
                weight_local,
                flat_inputs.to(device=weight_local.device, dtype=torch.float32),
                calibration=self.quant_calibration,
                percentile=float(self.quant_percentile),
            )
            qcodes, _ = runtime_nf4_quantize(dequant_weight, _reshape_channel_values(weight_scale, dequant_weight, 0))
        else:
            weight_scale = calibrate_nf4_scales(
                weight_local,
                method=self.quant_calibration,
                ch_axis=0,
                p=float(self.quant_percentile),
            )
            qcodes, _ = runtime_nf4_quantize(weight_local, _reshape_channel_values(weight_scale, weight_local, 0))
        self._assign_quantized_state(_pack_nf4_codes(qcodes), weight_scale, bias)
        if self.act_quant_mode == "static_int8" and flat_inputs is not None:
            act_inputs = flat_inputs.to(device=weight_local.device, dtype=torch.float32)
            if _spin_enabled(self.spin_enabled_flag):
                act_inputs = apply_spin_transform(act_inputs, self.spin_signs.to(device=weight_local.device))
            if _pre_scale_active(pre_scale):
                act_inputs = _apply_pre_scale_to_input(act_inputs, pre_scale)
            self.act_scale.copy_(
                calibrate_activation_scale(
                    act_inputs,
                    method=self.act_quant_method,
                    bits=int(self.act_quant_bits),
                    p=float(self.act_quant_percentile),
                ).to(device=self.act_scale.device, dtype=self.act_scale.dtype)
            )
        else:
            self.act_scale.fill_(1.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        qweight_packed_key = prefix + "qweight_packed"
        weight_scale_key = prefix + "weight_scale"
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        if qweight_packed_key in state_dict and weight_scale_key in state_dict:
            inserted = _inject_missing_state_defaults(
                state_dict,
                {
                    prefix + "spin_signs": self.spin_signs,
                    prefix + "spin_enabled_flag": self.spin_enabled_flag,
                    prefix + "pre_scale": self.pre_scale,
                    prefix + "act_scale": self.act_scale,
                },
            )
            try:
                super()._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
            finally:
                _remove_injected_state_defaults(state_dict, inserted)
            self._invalidate_weight_cache()
            return
        if weight_key in state_dict:
            try:
                weight = state_dict.pop(weight_key)
                bias = state_dict.pop(bias_key, None)
                self._assign_float_state(weight, bias)
            except Exception as exc:
                error_msgs.append(f"While quantizing checkpoint tensor for {prefix[:-1] or prefix}: {exc}")
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._invalidate_weight_cache()

    def _dequantized_weight(self, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.qweight_packed.device if device is None else torch.device(device)
        key = (
            str(target_device),
            str(dtype),
            tuple(self.qweight_packed.shape),
            int(getattr(self.qweight_packed, "_version", 0)),
            int(getattr(self.weight_scale, "_version", 0)),
            int(self.qweight_packed.data_ptr()),
            int(self.weight_scale.data_ptr()),
        )
        if self._cached_weight_key != key or self._cached_weight is None:
            qweight_packed = (
                self.qweight_packed if self.qweight_packed.device == target_device else self.qweight_packed.to(device=target_device)
            )
            weight_scale = self.weight_scale if self.weight_scale.device == target_device else self.weight_scale.to(device=target_device)
            self._cached_weight = _dequantize_nf4_per_channel(
                qweight_packed,
                weight_scale,
                original_last_dim=self.in_features,
                ch_axis=0,
            ).to(device=target_device, dtype=dtype).contiguous()
            self._cached_weight_key = key
        return self._cached_weight

    @property
    def weight(self) -> torch.Tensor:
        return self.runtime_weight()

    def runtime_weight(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target_dtype = torch.float32 if dtype is None else dtype
        target_device = self.qweight_packed.device if device is None else torch.device(device)
        weight = self._dequantized_weight(target_dtype, device=target_device)
        if _pre_scale_active(self.pre_scale):
            weight = _undo_pre_scale_from_weight(weight, self.pre_scale.to(device=target_device))
        if _spin_enabled(self.spin_enabled_flag):
            return undo_spin_transform(weight, self.spin_signs.to(device=target_device))
        return weight

    def runtime_bias(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias.to(
            device=self.bias.device if device is None else torch.device(device),
            dtype=self.bias.dtype if dtype is None else dtype,
        )

    def runtime_signature(self):
        return (
            "nf4_codebook_packed",
            _parameter_signature(self.qweight_packed),
            _parameter_signature(self.weight_scale),
            str(self.weight_opt),
            int(self.spin_enabled_flag.item()),
            _parameter_signature(self.spin_signs),
            _parameter_signature(self.pre_scale),
            str(self.act_quant_mode),
            str(self.act_quant_method),
            int(self.act_quant_bits),
            _parameter_signature(self.act_scale),
            _parameter_signature(self.bias),
        )

    def runtime_supports_packed_backend(self, backend: str) -> bool:
        del backend
        return False

    @torch.no_grad()
    def from_float(
        self,
        module: nn.Linear,
        calibration: str = "absmax",
        percentile: float = 0.999,
        calibration_inputs: torch.Tensor | None = None,
        weight_opt: str = "none",
        activation_quant: str = "none",
        activation_quant_bits: int = 8,
        activation_quant_method: str = "absmax",
        activation_quant_percentile: float = 0.999,
        spin: bool = False,
        spin_random: bool = True,
        spin_seed: int = 0,
    ) -> "QuantizedLinearNF4":
        assert module.in_features == self.in_features and module.out_features == self.out_features
        self.quant_calibration = str(calibration)
        self.quant_percentile = float(percentile)
        self.weight_opt = str(weight_opt)
        self.act_quant_mode = str(activation_quant)
        self.act_quant_bits = int(activation_quant_bits)
        self.act_quant_method = str(activation_quant_method)
        self.act_quant_percentile = float(activation_quant_percentile)
        _configure_spin_state(
            self.spin_enabled_flag,
            self.spin_signs,
            enabled=bool(spin),
            random_signs=bool(spin_random),
            seed=int(spin_seed),
        )
        self._assign_float_state(module.weight.data, None if module.bias is None else module.bias.data, calibration_inputs)
        return self

    def runtime_linear(self, x: torch.Tensor, *, backend: str | None = None) -> torch.Tensor:
        del backend
        target_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        target_device = x.device
        x_local = x.to(device=target_device, dtype=target_dtype)
        if _spin_enabled(self.spin_enabled_flag):
            x_local = apply_spin_transform(x_local, self.spin_signs.to(device=target_device))
        if _pre_scale_active(self.pre_scale):
            x_local = _apply_pre_scale_to_input(x_local, self.pre_scale.to(device=target_device))
        x_local = _apply_activation_quantization(
            x_local,
            mode=self.act_quant_mode,
            act_scale=self.act_scale.to(device=target_device),
            bits=int(self.act_quant_bits),
            method=self.act_quant_method,
            percentile=float(self.act_quant_percentile),
        )
        bias = self.runtime_bias(dtype=target_dtype, device=target_device)
        return runtime_nf4_linear(
            x_local,
            self.qweight_packed,
            self.weight_scale,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.runtime_linear(x)


class QuantizedLinearBitNet(nn.Module):
    """Weight-only BitNet linear using 2-bit ternary packed storage."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_calibration = "absmax"
        self.quant_percentile = 0.999
        self.weight_opt = "none"
        self.act_quant_mode = "none"
        self.act_quant_method = "absmax"
        self.act_quant_percentile = 0.999
        self.act_quant_bits = 8
        self.register_buffer("packed_weight", torch.empty(0, 0, dtype=torch.uint8))
        self.register_buffer("scale_values", torch.empty(0, dtype=torch.float32))
        self.register_buffer("layout_header", torch.zeros(13, dtype=torch.int32))
        self.register_buffer("segment_offsets", torch.tensor([0, out_features], dtype=torch.int32))
        self.register_buffer("spin_signs", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("spin_enabled_flag", torch.zeros((), dtype=torch.uint8))
        self.register_buffer("pre_scale", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("act_scale", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.layout_meta: dict[str, int] | None = None
        self._cached_weight_key: tuple[object, ...] | None = None
        self._cached_weight: torch.Tensor | None = None
        self._cached_compute_backend_key: tuple[object, ...] | None = None
        self._cached_compute_backend_words: torch.Tensor | None = None
        self._cached_compute_backend_row_scales: torch.Tensor | None = None
        self._cached_decode_backend_key: tuple[object, ...] | None = None
        self._cached_decode_backend_nz_masks: torch.Tensor | None = None
        self._cached_decode_backend_sign_masks: torch.Tensor | None = None
        self._cached_decode_backend_row_scales: torch.Tensor | None = None
        self._cached_int8_backend_key: tuple[object, ...] | None = None
        self._cached_int8_backend_qweight: torch.Tensor | None = None
        self._cached_int8_backend_inv_scale: torch.Tensor | None = None
        self._cached_int8_dense_weight_key: tuple[object, ...] | None = None
        self._cached_int8_dense_weight: torch.Tensor | None = None
        self._cached_shared_int8_input_signature_key: tuple[object, ...] | None = None
        self._cached_shared_int8_input_signature: tuple[object, ...] | None = None
        self._cached_spin_enabled_key: tuple[object, ...] | None = None
        self._cached_spin_enabled_value: bool | None = None
        self._cached_pre_scale_active_key: tuple[object, ...] | None = None
        self._cached_pre_scale_active_value: bool | None = None

    def _invalidate_weight_cache(self) -> None:
        self._cached_weight_key = None
        self._cached_weight = None
        self._cached_compute_backend_key = None
        self._cached_compute_backend_words = None
        self._cached_compute_backend_row_scales = None
        self._cached_decode_backend_key = None
        self._cached_decode_backend_nz_masks = None
        self._cached_decode_backend_sign_masks = None
        self._cached_decode_backend_row_scales = None
        self._cached_int8_backend_key = None
        self._cached_int8_backend_qweight = None
        self._cached_int8_backend_inv_scale = None
        self._cached_int8_dense_weight_key = None
        self._cached_int8_dense_weight = None
        self._cached_shared_int8_input_signature_key = None
        self._cached_shared_int8_input_signature = None
        self._cached_spin_enabled_key = None
        self._cached_spin_enabled_value = None
        self._cached_pre_scale_active_key = None
        self._cached_pre_scale_active_value = None

    def _spin_enabled_runtime(self) -> bool:
        key = (
            str(self.spin_enabled_flag.device),
            int(getattr(self.spin_enabled_flag, "_version", 0)),
            int(self.spin_enabled_flag.data_ptr()) if self.spin_enabled_flag.numel() > 0 else 0,
        )
        if self._cached_spin_enabled_key != key or self._cached_spin_enabled_value is None:
            self._cached_spin_enabled_value = _spin_enabled(self.spin_enabled_flag)
            self._cached_spin_enabled_key = key
        return bool(self._cached_spin_enabled_value)

    def _pre_scale_active_runtime(self) -> bool:
        key = (
            str(self.pre_scale.device),
            int(getattr(self.pre_scale, "_version", 0)),
            tuple(self.pre_scale.shape),
            int(self.pre_scale.data_ptr()) if self.pre_scale.numel() > 0 else 0,
        )
        if self._cached_pre_scale_active_key != key or self._cached_pre_scale_active_value is None:
            self._cached_pre_scale_active_value = _pre_scale_active(self.pre_scale)
            self._cached_pre_scale_active_key = key
        return bool(self._cached_pre_scale_active_value)

    def _assign_packed_state(
        self,
        packed_weight: torch.Tensor,
        scale_values: torch.Tensor,
        layout_header: torch.Tensor,
        segment_offsets: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.packed_weight = packed_weight.to(device=self.packed_weight.device, dtype=torch.uint8).contiguous()
        self.scale_values = scale_values.to(device=self.scale_values.device, dtype=torch.float32).contiguous()
        self.layout_header = layout_header.to(device=self.layout_header.device, dtype=torch.int32).contiguous()
        self.segment_offsets = segment_offsets.to(device=self.segment_offsets.device, dtype=torch.int32).contiguous()
        self.layout_meta = {
            "format_version": int(self.layout_header[0].item()),
            "tile_n": int(self.layout_header[1].item()),
            "tile_k": int(self.layout_header[2].item()),
            "logical_out_features": int(self.layout_header[3].item()),
            "logical_in_features": int(self.layout_header[4].item()),
            "padded_out_features": int(self.layout_header[5].item()),
            "padded_in_features": int(self.layout_header[6].item()),
            "scale_granularity": int(self.layout_header[7].item()),
            "scale_group_size": int(self.layout_header[8].item()),
            "interleave_mode": int(self.layout_header[9].item()),
            "arch_min": int(self.layout_header[10].item()),
            "segment_count": int(self.layout_header[11].item()),
            "flags": int(self.layout_header[12].item()),
        }
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
        self._invalidate_weight_cache()

    def _assign_float_state(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        calibration_inputs: torch.Tensor | None = None,
    ) -> None:
        weight_local = weight.float()
        if self._spin_enabled_runtime():
            weight_local = apply_spin_transform(weight_local, self.spin_signs.to(device=weight.device))
        flat_inputs = _flatten_calibration_inputs(calibration_inputs, in_features=self.in_features)
        requested_weight_opt = str(self.weight_opt).strip().lower()
        calibration_name = str(self.quant_calibration).strip().lower()
        if requested_weight_opt == "awq" and flat_inputs is not None:
            pre_scale = awq_optimize_pre_scale(
                weight_local,
                flat_inputs.to(device=weight_local.device, dtype=torch.float32),
                quantize_dequantize=lambda w: _bitnet_quantize_dequantize(
                    w,
                    calibration=calibration_name,
                    percentile=float(self.quant_percentile),
                ),
            )
        else:
            pre_scale = torch.ones(self.in_features, device=weight_local.device, dtype=torch.float32)
        self.pre_scale.copy_(pre_scale.to(device=self.pre_scale.device, dtype=self.pre_scale.dtype))
        weight_local = _apply_pre_scale_to_weight(weight_local, pre_scale)
        packed_weight, scale_values, layout_header, segment_offsets = runtime_pack_bitnet_weight(
            weight_local,
            calibration=calibration_name,
            percentile=float(self.quant_percentile),
            weight_opt="gptq" if requested_weight_opt == "gptq" else "none",
            calibration_inputs=None if flat_inputs is None else flat_inputs.to(device=weight_local.device, dtype=torch.float32),
        )
        self._assign_packed_state(packed_weight, scale_values, layout_header, segment_offsets, bias)
        if self.act_quant_mode == "static_int8" and flat_inputs is not None:
            act_inputs = flat_inputs.to(device=weight_local.device, dtype=torch.float32)
            if self._spin_enabled_runtime():
                act_inputs = apply_spin_transform(act_inputs, self.spin_signs.to(device=weight_local.device))
            if _pre_scale_active(pre_scale):
                act_inputs = _apply_pre_scale_to_input(act_inputs, pre_scale)
            self.act_scale.copy_(
                calibrate_activation_scale(
                    act_inputs,
                    method=self.act_quant_method,
                    bits=int(self.act_quant_bits),
                    p=float(self.act_quant_percentile),
                ).to(device=self.act_scale.device, dtype=self.act_scale.dtype)
            )
        else:
            self.act_scale.fill_(1.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        packed_key = prefix + "packed_weight"
        scale_key = prefix + "scale_values"
        header_key = prefix + "layout_header"
        offsets_key = prefix + "segment_offsets"
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        if packed_key in state_dict and scale_key in state_dict and header_key in state_dict and offsets_key in state_dict:
            inserted = _inject_missing_state_defaults(
                state_dict,
                {
                    prefix + "spin_signs": self.spin_signs,
                    prefix + "spin_enabled_flag": self.spin_enabled_flag,
                    prefix + "pre_scale": self.pre_scale,
                    prefix + "act_scale": self.act_scale,
                },
            )
            try:
                packed_weight = state_dict.pop(packed_key)
                scale_values = state_dict.pop(scale_key)
                layout_header = state_dict.pop(header_key)
                segment_offsets = state_dict.pop(offsets_key)
                bias = state_dict.pop(bias_key, None)
                self._assign_packed_state(packed_weight, scale_values, layout_header, segment_offsets, bias)
            except Exception as exc:
                error_msgs.append(f"While loading BitNet checkpoint tensor for {prefix[:-1] or prefix}: {exc}")
            finally:
                _remove_injected_state_defaults(state_dict, inserted)
            return
        if weight_key in state_dict:
            try:
                weight = state_dict.pop(weight_key)
                bias = state_dict.pop(bias_key, None)
                self._assign_float_state(weight, bias)
            except Exception as exc:
                error_msgs.append(f"While quantizing checkpoint tensor for {prefix[:-1] or prefix}: {exc}")
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._invalidate_weight_cache()

    def _dequantized_weight(self, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.packed_weight.device if device is None else torch.device(device)
        key = (
            str(target_device),
            str(dtype),
            tuple(self.packed_weight.shape),
            tuple(self.scale_values.shape),
            tuple(self.layout_header.shape),
            tuple(self.segment_offsets.shape),
            int(getattr(self.packed_weight, "_version", 0)),
            int(getattr(self.scale_values, "_version", 0)),
            int(getattr(self.layout_header, "_version", 0)),
            int(getattr(self.segment_offsets, "_version", 0)),
            int(self.packed_weight.data_ptr()) if self.packed_weight.numel() > 0 else 0,
            int(self.scale_values.data_ptr()) if self.scale_values.numel() > 0 else 0,
            int(self.layout_header.data_ptr()) if self.layout_header.numel() > 0 else 0,
            int(self.segment_offsets.data_ptr()) if self.segment_offsets.numel() > 0 else 0,
        )
        if self._cached_weight_key != key or self._cached_weight is None:
            packed_weight = (
                self.packed_weight if self.packed_weight.device == target_device else self.packed_weight.to(device=target_device)
            )
            scale_values = (
                self.scale_values if self.scale_values.device == target_device else self.scale_values.to(device=target_device)
            )
            layout_header = (
                self.layout_header if self.layout_header.device == target_device else self.layout_header.to(device=target_device)
            )
            segment_offsets = (
                self.segment_offsets if self.segment_offsets.device == target_device else self.segment_offsets.to(device=target_device)
            )
            self._cached_weight = _dequantize_bitnet_weight(
                packed_weight,
                scale_values,
                layout_header,
                segment_offsets,
                dtype=dtype,
            ).to(device=target_device, dtype=dtype).contiguous()
            self._cached_weight_key = key
        return self._cached_weight

    @property
    def weight(self) -> torch.Tensor:
        return self.runtime_weight()

    def runtime_weight(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target_dtype = torch.float32 if dtype is None else dtype
        target_device = self.packed_weight.device if device is None else torch.device(device)
        weight = self._dequantized_weight(target_dtype, device=target_device)
        if self._pre_scale_active_runtime():
            weight = _undo_pre_scale_from_weight(weight, self.pre_scale.to(device=target_device))
        if self._spin_enabled_runtime():
            return undo_spin_transform(weight, self.spin_signs.to(device=target_device))
        return weight

    def runtime_bias(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias.to(
            device=self.bias.device if device is None else torch.device(device),
            dtype=self.bias.dtype if dtype is None else dtype,
        )

    def runtime_signature(self):
        spin_enabled = self._spin_enabled_runtime()
        pre_scale_active = self._pre_scale_active_runtime()
        return (
            "bitnet_w2a8",
            _parameter_signature(self.packed_weight),
            _parameter_signature(self.scale_values),
            _parameter_signature(self.layout_header),
            _parameter_signature(self.segment_offsets),
            str(self.weight_opt),
            int(spin_enabled),
            _parameter_signature(self.spin_signs) if spin_enabled else None,
            _parameter_signature(self.pre_scale) if pre_scale_active else None,
            str(self.act_quant_mode),
            str(self.act_quant_method),
            int(self.act_quant_bits),
            float(self.act_quant_percentile),
            _parameter_signature(self.act_scale),
            _parameter_signature(self.bias),
        )

    def _uses_int8_packed_backend(self) -> bool:
        mode_name = str(self.act_quant_mode).strip().lower()
        return (not self._spin_enabled_runtime()) and mode_name in {"dynamic_int8", "static_int8"}

    def _compute_backend_weight(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_device = self.packed_weight.device if device is None else torch.device(device)
        key = (
            str(target_device),
            int(_BITNET_COMPUTE_TILE_N),
            tuple(self.packed_weight.shape),
            tuple(self.scale_values.shape),
            tuple(self.layout_header.shape),
            tuple(self.segment_offsets.shape),
            int(getattr(self.packed_weight, "_version", 0)),
            int(getattr(self.scale_values, "_version", 0)),
            int(getattr(self.layout_header, "_version", 0)),
            int(getattr(self.segment_offsets, "_version", 0)),
            int(self.packed_weight.data_ptr()) if self.packed_weight.numel() > 0 else 0,
            int(self.scale_values.data_ptr()) if self.scale_values.numel() > 0 else 0,
            int(self.layout_header.data_ptr()) if self.layout_header.numel() > 0 else 0,
            int(self.segment_offsets.data_ptr()) if self.segment_offsets.numel() > 0 else 0,
        )
        if (
            self._cached_compute_backend_key != key
            or self._cached_compute_backend_words is None
            or self._cached_compute_backend_row_scales is None
        ):
            packed_weight = (
                self.packed_weight if self.packed_weight.device == target_device else self.packed_weight.to(device=target_device)
            )
            scale_values = (
                self.scale_values if self.scale_values.device == target_device else self.scale_values.to(device=target_device)
            )
            layout_header = (
                self.layout_header if self.layout_header.device == target_device else self.layout_header.to(device=target_device)
            )
            segment_offsets = (
                self.segment_offsets if self.segment_offsets.device == target_device else self.segment_offsets.to(device=target_device)
            )
            packed_words, row_scales = _pack_bitnet_compute_weight(
                packed_weight,
                scale_values,
                layout_header,
                segment_offsets,
                tile_n=_BITNET_COMPUTE_TILE_N,
            )
            self._cached_compute_backend_words = packed_words.to(device=target_device, dtype=torch.int32).contiguous()
            self._cached_compute_backend_row_scales = row_scales.to(device=target_device, dtype=torch.float32).contiguous()
            self._cached_compute_backend_key = key
        return self._cached_compute_backend_words, self._cached_compute_backend_row_scales

    def _decode_backend_weight(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_device = self.packed_weight.device if device is None else torch.device(device)
        key = (
            str(target_device),
            int(_BITNET_DECODE_TILE_N),
            int(_BITNET_DECODE_CHUNK_K),
            tuple(self.packed_weight.shape),
            tuple(self.scale_values.shape),
            tuple(self.layout_header.shape),
            tuple(self.segment_offsets.shape),
            int(getattr(self.packed_weight, "_version", 0)),
            int(getattr(self.scale_values, "_version", 0)),
            int(getattr(self.layout_header, "_version", 0)),
            int(getattr(self.segment_offsets, "_version", 0)),
            int(self.packed_weight.data_ptr()) if self.packed_weight.numel() > 0 else 0,
            int(self.scale_values.data_ptr()) if self.scale_values.numel() > 0 else 0,
            int(self.layout_header.data_ptr()) if self.layout_header.numel() > 0 else 0,
            int(self.segment_offsets.data_ptr()) if self.segment_offsets.numel() > 0 else 0,
        )
        if (
            self._cached_decode_backend_key != key
            or self._cached_decode_backend_nz_masks is None
            or self._cached_decode_backend_sign_masks is None
            or self._cached_decode_backend_row_scales is None
        ):
            packed_weight = (
                self.packed_weight if self.packed_weight.device == target_device else self.packed_weight.to(device=target_device)
            )
            scale_values = (
                self.scale_values if self.scale_values.device == target_device else self.scale_values.to(device=target_device)
            )
            layout_header = (
                self.layout_header if self.layout_header.device == target_device else self.layout_header.to(device=target_device)
            )
            segment_offsets = (
                self.segment_offsets if self.segment_offsets.device == target_device else self.segment_offsets.to(device=target_device)
            )
            nz_masks, sign_masks, row_scales = _pack_bitnet_decode_backend_weight(
                packed_weight,
                scale_values,
                layout_header,
                segment_offsets,
                tile_n=_BITNET_DECODE_TILE_N,
                chunk_k=_BITNET_DECODE_CHUNK_K,
            )
            self._cached_decode_backend_nz_masks = nz_masks.to(device=target_device, dtype=torch.int32).contiguous()
            self._cached_decode_backend_sign_masks = sign_masks.to(device=target_device, dtype=torch.int32).contiguous()
            self._cached_decode_backend_row_scales = row_scales.to(device=target_device, dtype=torch.float32).contiguous()
            self._cached_decode_backend_key = key
        return (
            self._cached_decode_backend_nz_masks,
            self._cached_decode_backend_sign_masks,
            self._cached_decode_backend_row_scales,
        )

    def _int8_backend_weight(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_device = self.packed_weight.device if device is None else torch.device(device)
        key = (
            str(target_device),
            tuple(self.packed_weight.shape),
            tuple(self.scale_values.shape),
            tuple(self.layout_header.shape),
            tuple(self.segment_offsets.shape),
            int(getattr(self.packed_weight, "_version", 0)),
            int(getattr(self.scale_values, "_version", 0)),
            int(getattr(self.layout_header, "_version", 0)),
            int(getattr(self.segment_offsets, "_version", 0)),
            int(self.packed_weight.data_ptr()) if self.packed_weight.numel() > 0 else 0,
            int(self.scale_values.data_ptr()) if self.scale_values.numel() > 0 else 0,
            int(self.layout_header.data_ptr()) if self.layout_header.numel() > 0 else 0,
            int(self.segment_offsets.data_ptr()) if self.segment_offsets.numel() > 0 else 0,
        )
        if (
            self._cached_int8_backend_key != key
            or self._cached_int8_backend_qweight is None
            or self._cached_int8_backend_inv_scale is None
        ):
            packed_weight = (
                self.packed_weight if self.packed_weight.device == target_device else self.packed_weight.to(device=target_device)
            )
            scale_values = (
                self.scale_values if self.scale_values.device == target_device else self.scale_values.to(device=target_device)
            )
            layout_header = (
                self.layout_header if self.layout_header.device == target_device else self.layout_header.to(device=target_device)
            )
            segment_offsets = (
                self.segment_offsets if self.segment_offsets.device == target_device else self.segment_offsets.to(device=target_device)
            )
            logical_in = int(layout_header[4].item())
            logical_out = int(layout_header[3].item())
            self._cached_int8_backend_qweight = (
                _unpack_bitnet_signed(packed_weight, original_last_dim=logical_in)[:logical_out]
                .to(device=target_device, dtype=torch.int8)
                .contiguous()
            )
            self._cached_int8_backend_inv_scale = (
                _bitnet_row_scales(scale_values, layout_header, segment_offsets)
                .to(device=target_device, dtype=torch.float32)
                .contiguous()
            )
            self._cached_int8_backend_key = key
        return self._cached_int8_backend_qweight, self._cached_int8_backend_inv_scale

    def _dequantized_int8_backend_weight(
        self,
        dtype: torch.dtype,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target_device = self.packed_weight.device if device is None else torch.device(device)
        qweight, inv_scale = self._int8_backend_weight(device=target_device)
        key = (
            str(target_device),
            str(dtype),
            tuple(qweight.shape),
            tuple(inv_scale.shape),
            _tensor_version_safe(qweight),
            _tensor_version_safe(inv_scale),
            int(qweight.data_ptr()) if qweight.numel() > 0 else 0,
            int(inv_scale.data_ptr()) if inv_scale.numel() > 0 else 0,
        )
        if self._cached_int8_dense_weight_key != key or self._cached_int8_dense_weight is None:
            self._cached_int8_dense_weight = _dequantize_int8_weight(
                qweight,
                inv_scale,
                dtype=dtype,
            ).to(device=target_device, dtype=dtype).contiguous()
            self._cached_int8_dense_weight_key = key
        return self._cached_int8_dense_weight

    def _prefer_dynamic_int8_dense_decode_fallback(self, x: torch.Tensor) -> bool:
        if not x.is_cuda or x.ndim < 2:
            return False
        rows = int(x.numel() // max(int(x.shape[-1]), 1))
        if rows <= 0 or rows > 8:
            return False
        if self._pre_scale_active_runtime():
            return False
        if int(self.out_features) >= 32768:
            return False
        # Production decode fallback is intentionally 8-bit only. Sub-8-bit
        # dynamic decode still relies on the native int8-from-float path and
        # needs separate policy work if we want it to route differently.
        return int(self.in_features) >= 2048 or int(self.out_features) >= 2048

    def _prefer_dynamic_int8_dense_hopper_prefill_fallback(self, x: torch.Tensor) -> bool:
        if not x.is_cuda or x.ndim < 2:
            return False
        rows = int(x.numel() // max(int(x.shape[-1]), 1))
        if rows <= 8:
            return False
        if self._pre_scale_active_runtime():
            return False
        try:
            major, minor = torch.cuda.get_device_capability(x.device)
        except Exception:
            return False
        if (int(major), int(minor)) < (9, 0):
            return False
        if self._prefer_dynamic_int8_direct_hopper_prefill(x):
            return False
        # Hopper BF16 GEMMs are already strong enough on large prefill tiles
        # that the dense cached-weight path beats the W2A8 frontend for the
        # Parameter Golf BitNet projections we care about, including 1024-wide
        # attention/output layers. Keep this policy 8-bit-only rather than
        # mixing it into sub-8-bit dynamic tuning.
        return int(self.act_quant_bits) == 8

    def _prefer_dynamic_int8_direct_hopper_prefill(self, x: torch.Tensor) -> bool:
        cutlass_enabled = os.getenv("MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED", "").strip().lower()
        if cutlass_enabled not in {"1", "true", "yes", "on"}:
            return False
        if not x.is_cuda or x.ndim < 2:
            return False
        rows = int(x.numel() // max(int(x.shape[-1]), 1))
        if rows <= 8:
            return False
        if self._pre_scale_active_runtime():
            return False
        try:
            major, minor = torch.cuda.get_device_capability(x.device)
        except Exception:
            return False
        if (int(major), int(minor)) < (9, 0):
            return False
        if int(self.act_quant_bits) != 8:
            return False
        shape = (int(self.in_features), int(self.out_features))
        return shape in {(1024, 3072), (3072, 1024)}

    def runtime_packed_linear_signature(self, backend: str):
        if not self.runtime_supports_packed_backend(backend):
            return None
        spin_enabled = self._spin_enabled_runtime()
        pre_scale_active = self._pre_scale_active_runtime()
        if self._uses_int8_packed_backend():
            return (
                "bitnet_w2a8_int8",
                str(backend).strip().lower(),
                _parameter_signature(self.packed_weight),
                _parameter_signature(self.scale_values),
                _parameter_signature(self.layout_header),
                _parameter_signature(self.segment_offsets),
                int(spin_enabled),
                _parameter_signature(self.spin_signs) if spin_enabled else None,
                _parameter_signature(self.pre_scale) if pre_scale_active else None,
                str(self.act_quant_mode),
                str(self.act_quant_method),
                int(self.act_quant_bits),
                float(self.act_quant_percentile),
                _parameter_signature(self.act_scale),
                _parameter_signature(self.bias),
            )
        return (
            "bitnet_w2a8",
            str(backend).strip().lower(),
            _parameter_signature(self.packed_weight),
            _parameter_signature(self.scale_values),
            _parameter_signature(self.layout_header),
            _parameter_signature(self.segment_offsets),
            int(spin_enabled),
            _parameter_signature(self.spin_signs) if spin_enabled else None,
            _parameter_signature(self.pre_scale) if pre_scale_active else None,
            str(self.act_quant_mode),
            str(self.act_quant_method),
            int(self.act_quant_bits),
            float(self.act_quant_percentile),
            _parameter_signature(self.act_scale),
            _parameter_signature(self.bias),
        )

    def runtime_packed_linear_spec(
        self,
        *,
        backend: str,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        if not self.runtime_supports_packed_backend(backend):
            return None
        target_device = self.packed_weight.device if device is None else torch.device(device)
        spin_enabled = self._spin_enabled_runtime()
        pre_scale_active = self._pre_scale_active_runtime()
        bias = self.runtime_bias(
            dtype=self.bias.dtype if dtype is None else dtype,
            device=target_device,
        )
        if isinstance(bias, torch.Tensor):
            bias = bias.detach().contiguous()
        if self._uses_int8_packed_backend():
            qweight, inv_scale = self._int8_backend_weight(device=target_device)
            return {
                "format": "bitnet_w2a8_int8",
                "backend": str(backend).strip().lower(),
                "qweight": qweight.contiguous(),
                "inv_scale": inv_scale.contiguous(),
                "bias": bias,
                "spin_enabled": False,
                "spin_signs": None,
                "pre_scale": self.pre_scale.to(device=target_device, dtype=torch.float32).contiguous() if pre_scale_active else None,
                "act_quant_mode": str(self.act_quant_mode),
                "act_quant_method": str(self.act_quant_method),
                "act_quant_bits": int(self.act_quant_bits),
                "act_quant_percentile": float(self.act_quant_percentile),
                "act_scale": self.act_scale.to(device=target_device, dtype=torch.float32).contiguous(),
            }
        packed_weight = self.packed_weight if self.packed_weight.device == target_device else self.packed_weight.to(device=target_device)
        scale_values = self.scale_values if self.scale_values.device == target_device else self.scale_values.to(device=target_device)
        layout_header = self.layout_header if self.layout_header.device == target_device else self.layout_header.to(device=target_device)
        segment_offsets = (
            self.segment_offsets if self.segment_offsets.device == target_device else self.segment_offsets.to(device=target_device)
        )
        return {
            "format": "bitnet_w2a8",
            "backend": str(backend).strip().lower(),
            "packed_weight": packed_weight.contiguous(),
            "scale_values": scale_values.contiguous(),
            "layout_header": layout_header.contiguous(),
            "segment_offsets": segment_offsets.contiguous(),
            "bias": bias,
            "spin_enabled": spin_enabled,
            "spin_signs": self.spin_signs.to(device=target_device, dtype=torch.float32).contiguous() if spin_enabled else None,
            "pre_scale": self.pre_scale.to(device=target_device, dtype=torch.float32).contiguous() if pre_scale_active else None,
            "act_quant_mode": str(self.act_quant_mode),
            "act_quant_method": str(self.act_quant_method),
            "act_quant_bits": int(self.act_quant_bits),
            "act_quant_percentile": float(self.act_quant_percentile),
            "act_scale": self.act_scale.to(device=target_device, dtype=torch.float32).contiguous(),
        }

    def runtime_supports_packed_backend(self, backend: str) -> bool:
        return str(backend).strip().lower() == "bitnet"

    def runtime_shared_int8_input_signature(self):
        mode_name = str(self.act_quant_mode).strip().lower()
        bits = int(self.act_quant_bits)
        if bits < 2 or bits > 8 or mode_name not in {"dynamic_int8", "static_int8"}:
            return None
        spin_enabled = self._spin_enabled_runtime()
        pre_scale_active = self._pre_scale_active_runtime()
        cache_key = (
            mode_name,
            bits,
            int(spin_enabled),
            _parameter_signature(self.spin_signs) if spin_enabled else None,
            _parameter_signature(self.pre_scale) if pre_scale_active else None,
            str(self.act_quant_method),
            float(self.act_quant_percentile),
            _parameter_signature(self.act_scale),
        )
        if (
            self._cached_shared_int8_input_signature_key != cache_key
            or self._cached_shared_int8_input_signature is None
        ):
            signature = (
                int(spin_enabled),
                tuple(self.spin_signs.detach().cpu().tolist()) if spin_enabled else (),
                tuple(self.pre_scale.detach().cpu().tolist()) if pre_scale_active else (),
                mode_name,
                bits,
            )
            if mode_name == "dynamic_int8":
                signature = signature + (
                    str(self.act_quant_method),
                    float(self.act_quant_percentile),
                )
            else:
                signature = signature + (
                    tuple(self.act_scale.detach().cpu().reshape(-1).tolist()),
                )
            self._cached_shared_int8_input_signature_key = cache_key
            self._cached_shared_int8_input_signature = signature
        return self._cached_shared_int8_input_signature

    def runtime_quantize_int8_input(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
        target_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        target_device = x.device
        x_local = x.to(device=target_device, dtype=target_dtype)
        if self._spin_enabled_runtime():
            x_local = apply_spin_transform(x_local, self.spin_signs.to(device=target_device))
        if self._pre_scale_active_runtime():
            x_local = _apply_pre_scale_to_input(x_local, self.pre_scale.to(device=target_device))

        mode_name = str(self.act_quant_mode).strip().lower()
        bits = int(self.act_quant_bits)
        if bits < 2 or bits > 8 or mode_name not in {"dynamic_int8", "static_int8"}:
            raise RuntimeError(
                "runtime_quantize_int8_input requires activation_quant in {dynamic_int8, static_int8} with bits in [2, 8]"
            )

        flat = x_local.reshape(-1, x_local.shape[-1])
        rows = int(flat.shape[0])
        if mode_name == "dynamic_int8":
            qx, row_scale = runtime_quantize_activation_int8_rowwise(
                x_local,
                scale=None,
                method=self.act_quant_method,
                percentile=float(self.act_quant_percentile),
                bits=bits,
            )
        else:
            scale = self.act_scale.to(device=target_device, dtype=torch.float32).reshape(-1)
            if scale.numel() == 1:
                row_scale = scale.expand(rows).contiguous()
            elif scale.numel() == rows:
                row_scale = scale.contiguous()
            else:
                raise ValueError("BitNet static_int8 act_scale must have 1 or rows elements")
            qx, row_scale = runtime_quantize_activation_int8_rowwise(
                x_local,
                scale=row_scale,
                method=self.act_quant_method,
                percentile=float(self.act_quant_percentile),
                bits=bits,
            )

        return qx.reshape_as(x_local), row_scale, target_dtype

    def runtime_linear_from_quantized_input(
        self,
        qx: torch.Tensor,
        row_scale: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_dtype = torch.float32 if out_dtype is None else out_dtype
        qweight, inv_scale = self._int8_backend_weight(device=qx.device)
        bias = self.runtime_bias(dtype=target_dtype, device=qx.device)
        return runtime_int8_linear_from_quantized_activation(
            qx,
            row_scale,
            qweight,
            inv_scale,
            bias,
            out_dtype=target_dtype,
        )

    @torch.no_grad()
    def from_float(
        self,
        module: nn.Linear,
        calibration: str = "absmax",
        percentile: float = 0.999,
        calibration_inputs: torch.Tensor | None = None,
        weight_opt: str = "none",
        activation_quant: str = "none",
        activation_quant_bits: int = 8,
        activation_quant_method: str = "absmax",
        activation_quant_percentile: float = 0.999,
        spin: bool = False,
        spin_random: bool = True,
        spin_seed: int = 0,
    ) -> "QuantizedLinearBitNet":
        assert module.in_features == self.in_features and module.out_features == self.out_features
        self.quant_calibration = str(calibration)
        self.quant_percentile = float(percentile)
        self.weight_opt = str(weight_opt)
        self.act_quant_mode = str(activation_quant)
        self.act_quant_bits = int(activation_quant_bits)
        self.act_quant_method = str(activation_quant_method)
        self.act_quant_percentile = float(activation_quant_percentile)
        _configure_spin_state(
            self.spin_enabled_flag,
            self.spin_signs,
            enabled=bool(spin),
            random_signs=bool(spin_random),
            seed=int(spin_seed),
        )
        self._cached_spin_enabled_key = None
        self._cached_spin_enabled_value = bool(spin)
        self._cached_pre_scale_active_key = None
        self._cached_pre_scale_active_value = None
        self._assign_float_state(module.weight.data, None if module.bias is None else module.bias.data, calibration_inputs)
        return self

    def runtime_linear(self, x: torch.Tensor, *, backend: str | None = None) -> torch.Tensor:
        mode_name = str(self.act_quant_mode).strip().lower()
        target_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        target_device = x.device

        if mode_name in {"", "none", "off"}:
            x_local = x.to(device=target_device, dtype=target_dtype)
            spin_enabled = self._spin_enabled_runtime()
            pre_scale_active = self._pre_scale_active_runtime()
            if not spin_enabled and not pre_scale_active:
                backend_name = str(backend or "auto").strip().lower()
                if backend_name == "bitnet":
                    compute_words, compute_scales = self._compute_backend_weight(device=target_device)
                    decode_nz, decode_sign, decode_scales = self._decode_backend_weight(device=target_device)
                    return runtime_bitnet_linear_compute_packed(
                        x_local,
                        self.packed_weight,
                        self.scale_values,
                        self.layout_header,
                        self.segment_offsets,
                        compute_words,
                        compute_scales,
                        decode_nz,
                        decode_sign,
                        decode_scales,
                        self.runtime_bias(dtype=target_dtype, device=target_device),
                        out_dtype=target_dtype,
                    )
                weight = self._dequantized_weight(target_dtype, device=target_device)
                bias = None if self.bias is None else self.bias.to(device=target_device, dtype=target_dtype)
                if backend_name in {"", "auto", "aten"}:
                    return torch.nn.functional.linear(x_local, weight, bias)
                return runtime_linear(x_local, weight, bias, backend=backend)
            bias = self.runtime_bias(dtype=target_dtype, device=target_device)
            weight = self.runtime_weight(dtype=target_dtype, device=target_device)
            if backend is None:
                return runtime_linear(x_local, weight, bias)
            return runtime_linear(x_local, weight, bias, backend=backend)

        if self._uses_int8_packed_backend():
            backend_name = str(backend or "auto").strip().lower()
            if (
                mode_name == "dynamic_int8"
                and backend_name in {"", "auto", "aten"}
                and not self._spin_enabled_runtime()
                and int(self.act_quant_bits) == 8
            ):
                if self._prefer_dynamic_int8_dense_decode_fallback(x) or self._prefer_dynamic_int8_dense_hopper_prefill_fallback(x):
                    x_local = x.to(device=target_device, dtype=target_dtype)
                    weight = self._dequantized_int8_backend_weight(target_dtype, device=target_device)
                    bias = self.runtime_bias(dtype=target_dtype, device=target_device)
                    return torch.nn.functional.linear(x_local, weight, bias)
            qweight, inv_scale = self._int8_backend_weight(device=target_device)
            pre_scale_active = self._pre_scale_active_runtime()
            pre_scale = self.pre_scale.to(device=target_device) if pre_scale_active else None
            act_scale = self.act_scale.to(device=target_device) if mode_name == "static_int8" else None
            return runtime_bitnet_int8_linear_from_float(
                x.to(device=target_device, dtype=target_dtype),
                qweight,
                inv_scale,
                self.runtime_bias(dtype=target_dtype, device=target_device),
                pre_scale=pre_scale,
                act_quant_mode=self.act_quant_mode,
                act_scale=act_scale,
                act_quant_bits=int(self.act_quant_bits),
                act_quant_method=self.act_quant_method,
                act_quant_percentile=float(self.act_quant_percentile),
            )

        spin_enabled = self._spin_enabled_runtime()
        spin_signs = self.spin_signs.to(device=target_device) if spin_enabled else None
        pre_scale = self.pre_scale.to(device=target_device) if self._pre_scale_active_runtime() else None
        act_scale = self.act_scale.to(device=target_device) if mode_name == "static_int8" else None
        x_local = runtime_bitnet_transform_input(
            x.to(device=target_device, dtype=target_dtype),
            spin_enabled=spin_enabled,
            spin_signs=spin_signs,
            pre_scale=pre_scale,
            act_quant_mode=self.act_quant_mode,
            act_scale=act_scale,
            act_quant_bits=int(self.act_quant_bits),
            act_quant_method=self.act_quant_method,
            act_quant_percentile=float(self.act_quant_percentile),
        )
        return runtime_bitnet_linear(
            x_local,
            self.packed_weight,
            self.scale_values,
            self.layout_header,
            self.segment_offsets,
            bias=self.runtime_bias(dtype=target_dtype, device=target_device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.runtime_linear(x)


class TrainableBitNetLinear(nn.Module):
    """Trainable runtime-row BitNet linear with a float shadow weight.

    The forward path applies a straight-through ternary projection to the
    trainable weight. ``to_quantized`` exports the same row scales and ternary
    codes into ``QuantizedLinearBitNet`` packed runtime state.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        scale_layout: str = "runtime_row",
        group_size: int = 64,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        layout = str(scale_layout).strip().lower()
        if layout not in {"runtime_row", "row", "per_row"}:
            raise ValueError("TrainableBitNetLinear currently supports only runtime_row scaling")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale_layout = "runtime_row"
        self.group_size = int(group_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def from_float(self, module: nn.Linear) -> "TrainableBitNetLinear":
        if module.in_features != self.in_features or module.out_features != self.out_features:
            raise ValueError(
                "source Linear shape does not match TrainableBitNetLinear: "
                f"expected ({self.out_features}, {self.in_features}), got {tuple(module.weight.shape)}"
            )
        self.weight.copy_(module.weight.detach().to(device=self.weight.device, dtype=self.weight.dtype))
        if self.bias is not None:
            if module.bias is None:
                self.bias.zero_()
            else:
                self.bias.copy_(module.bias.detach().to(device=self.bias.device, dtype=self.bias.dtype))
        return self

    def ternary_weight(self) -> torch.Tensor:
        return _bitnet_runtime_row_ste_weight(self.weight, eps=self.eps)

    @torch.no_grad()
    def quantized_codes_and_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _bitnet_runtime_row_codes_and_scale(self.weight.detach(), eps=self.eps)

    @torch.no_grad()
    def to_quantized(
        self,
        *,
        activation_quant: str = "none",
        activation_quant_bits: int = 8,
        activation_quant_method: str = "absmax",
        activation_quant_percentile: float = 0.999,
    ) -> "QuantizedLinearBitNet":
        qweight, row_scale = self.quantized_codes_and_scales()
        packed_weight, scale_values, layout_header, segment_offsets = _pack_bitnet_quantized(
            qweight,
            scale_values=row_scale,
            scale_granularity=2,
            scale_group_size=1,
        )
        layer = QuantizedLinearBitNet(self.in_features, self.out_features, bias=self.bias is not None).to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        layer.quant_calibration = "runtime_row"
        layer.act_quant_mode = str(activation_quant)
        layer.act_quant_bits = int(activation_quant_bits)
        layer.act_quant_method = str(activation_quant_method)
        layer.act_quant_percentile = float(activation_quant_percentile)
        layer._assign_packed_state(
            packed_weight.to(device=layer.packed_weight.device),
            scale_values.to(device=layer.scale_values.device),
            layout_header.to(device=layer.layout_header.device),
            segment_offsets.to(device=layer.segment_offsets.device),
            None if self.bias is None else self.bias.detach(),
        )
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.ternary_weight().to(dtype=x.dtype if x.dtype.is_floating_point else self.weight.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=weight.dtype)
        return F.linear(x.to(dtype=weight.dtype), weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, scale_layout={self.scale_layout!r}"
        )


class QuantizedLinearFP8(nn.Module):
    """Weight-only fake-FP8 linear routed through runtime quantized inference helpers."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, fp8_dtype: str = "e4m3") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_dtype = str(fp8_dtype)
        self.quant_calibration = "absmax"
        self.quant_percentile = 0.999
        self.weight_opt = "none"
        self.act_quant_mode = "none"
        self.act_quant_method = "absmax"
        self.act_quant_percentile = 0.999
        self.act_quant_bits = 8
        self.register_buffer("weight_fp8", torch.empty(out_features, in_features, dtype=torch.float32))
        self.register_buffer("weight_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("amax_observed", torch.zeros((), dtype=torch.float32))
        self.register_buffer("spin_signs", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("spin_enabled_flag", torch.zeros((), dtype=torch.uint8))
        self.register_buffer("pre_scale", torch.ones(in_features, dtype=torch.float32))
        self.register_buffer("act_scale", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._amax_tracker = _AmaxTracker(self)
        self._cached_weight_key: tuple[object, ...] | None = None
        self._cached_weight: torch.Tensor | None = None

    def _invalidate_weight_cache(self) -> None:
        self._cached_weight_key = None
        self._cached_weight = None

    def _assign_quantized_state(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        amax_observed: torch.Tensor | None = None,
    ) -> None:
        self.weight_fp8.copy_(weight_fp8.to(device=self.weight_fp8.device, dtype=self.weight_fp8.dtype))
        self.weight_scale.copy_(weight_scale.reshape(()).to(device=self.weight_scale.device, dtype=self.weight_scale.dtype))
        observed = self.weight_fp8.detach().abs().amax() if amax_observed is None else amax_observed
        self.amax_observed.copy_(observed.to(device=self.amax_observed.device, dtype=self.amax_observed.dtype))
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
        self._invalidate_weight_cache()

    def _assign_float_state(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        calibration_inputs: torch.Tensor | None = None,
    ) -> None:
        weight_local = weight.float()
        if _spin_enabled(self.spin_enabled_flag):
            weight_local = apply_spin_transform(weight_local, self.spin_signs.to(device=weight.device))
        flat_inputs = _flatten_calibration_inputs(calibration_inputs, in_features=self.in_features)
        if str(self.weight_opt).strip().lower() in {"awq", "gptq"} and flat_inputs is not None:
            pre_scale = awq_optimize_pre_scale(
                weight_local,
                flat_inputs.to(device=weight_local.device, dtype=torch.float32),
                quantize_dequantize=lambda w: _fp8_quantize_dequantize(
                    w,
                    calibration=self.quant_calibration,
                    percentile=float(self.quant_percentile),
                    fp8_dtype=self.fp8_dtype,
                ),
            )
        else:
            pre_scale = torch.ones(self.in_features, device=weight_local.device, dtype=torch.float32)
        self.pre_scale.copy_(pre_scale.to(device=self.pre_scale.device, dtype=self.pre_scale.dtype))
        weight_local = _apply_pre_scale_to_weight(weight_local, pre_scale)
        scale = calibrate_fp8_scale(
            weight_local,
            method=self.quant_calibration,
            p=float(self.quant_percentile),
        )
        quantized = fake_quantize_fp8(weight_local / scale, dtype=self.fp8_dtype)
        self._assign_quantized_state(quantized, scale, bias)
        if self.act_quant_mode == "static_int8" and flat_inputs is not None:
            act_inputs = flat_inputs.to(device=weight_local.device, dtype=torch.float32)
            if _spin_enabled(self.spin_enabled_flag):
                act_inputs = apply_spin_transform(act_inputs, self.spin_signs.to(device=weight_local.device))
            if _pre_scale_active(pre_scale):
                act_inputs = _apply_pre_scale_to_input(act_inputs, pre_scale)
            self.act_scale.copy_(
                calibrate_activation_scale(
                    act_inputs,
                    method=self.act_quant_method,
                    bits=int(self.act_quant_bits),
                    p=float(self.act_quant_percentile),
                ).to(device=self.act_scale.device, dtype=self.act_scale.dtype)
            )
        else:
            self.act_scale.fill_(1.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        fp8_weight_key = prefix + "weight_fp8"
        fp8_scale_key = prefix + "weight_scale"
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        if fp8_weight_key in state_dict and fp8_scale_key in state_dict:
            inserted = _inject_missing_state_defaults(
                state_dict,
                {
                    prefix + "spin_signs": self.spin_signs,
                    prefix + "spin_enabled_flag": self.spin_enabled_flag,
                    prefix + "pre_scale": self.pre_scale,
                    prefix + "act_scale": self.act_scale,
                },
            )
            try:
                super()._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
            finally:
                _remove_injected_state_defaults(state_dict, inserted)
            self._invalidate_weight_cache()
            return
        if weight_key in state_dict:
            try:
                weight = state_dict.pop(weight_key)
                bias = state_dict.pop(bias_key, None)
                self._assign_float_state(weight, bias)
            except Exception as exc:
                error_msgs.append(f"While quantizing checkpoint tensor for {prefix[:-1] or prefix}: {exc}")
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._invalidate_weight_cache()

    def _dequantized_weight(self, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.weight_fp8.device if device is None else torch.device(device)
        key = (
            str(target_device),
            str(dtype),
            str(self.fp8_dtype),
            tuple(self.weight_fp8.shape),
            int(getattr(self.weight_fp8, "_version", 0)),
            int(getattr(self.weight_scale, "_version", 0)),
            int(self.weight_fp8.data_ptr()),
            int(self.weight_scale.data_ptr()),
        )
        if self._cached_weight_key != key or self._cached_weight is None:
            weight_fp8 = self.weight_fp8 if self.weight_fp8.device == target_device else self.weight_fp8.to(device=target_device)
            scale = self.weight_scale if self.weight_scale.device == target_device else self.weight_scale.to(device=target_device)
            self._cached_weight = (weight_fp8.to(dtype=torch.float32) * scale.to(dtype=torch.float32)).to(dtype=dtype).contiguous()
            self._cached_weight_key = key
        return self._cached_weight

    @property
    def weight(self) -> torch.Tensor:
        return self.runtime_weight()

    def runtime_weight(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target_dtype = torch.float32 if dtype is None else dtype
        target_device = self.weight_fp8.device if device is None else torch.device(device)
        weight = self._dequantized_weight(target_dtype, device=target_device)
        if _pre_scale_active(self.pre_scale):
            weight = _undo_pre_scale_from_weight(weight, self.pre_scale.to(device=target_device))
        if _spin_enabled(self.spin_enabled_flag):
            return undo_spin_transform(weight, self.spin_signs.to(device=target_device))
        return weight

    def runtime_bias(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias.to(
            device=self.bias.device if device is None else torch.device(device),
            dtype=self.bias.dtype if dtype is None else dtype,
        )

    def runtime_signature(self):
        return (
            "fp8_fake",
            str(self.fp8_dtype),
            _parameter_signature(self.weight_fp8),
            _parameter_signature(self.weight_scale),
            _parameter_signature(self.amax_observed),
            str(self.weight_opt),
            int(self.spin_enabled_flag.item()),
            _parameter_signature(self.spin_signs),
            _parameter_signature(self.pre_scale),
            str(self.act_quant_mode),
            str(self.act_quant_method),
            int(self.act_quant_bits),
            _parameter_signature(self.act_scale),
            _parameter_signature(self.bias),
        )

    def runtime_supports_packed_backend(self, backend: str) -> bool:
        del backend
        return False

    @torch.no_grad()
    def from_float(
        self,
        module: nn.Linear,
        calibration: str = "absmax",
        percentile: float = 0.999,
        fp8_dtype: str | None = None,
        calibration_inputs: torch.Tensor | None = None,
        weight_opt: str = "none",
        activation_quant: str = "none",
        activation_quant_bits: int = 8,
        activation_quant_method: str = "absmax",
        activation_quant_percentile: float = 0.999,
        spin: bool = False,
        spin_random: bool = True,
        spin_seed: int = 0,
    ) -> "QuantizedLinearFP8":
        assert module.in_features == self.in_features and module.out_features == self.out_features
        if fp8_dtype is not None:
            self.fp8_dtype = str(fp8_dtype)
        self.quant_calibration = str(calibration)
        self.quant_percentile = float(percentile)
        self.weight_opt = str(weight_opt)
        self.act_quant_mode = str(activation_quant)
        self.act_quant_bits = int(activation_quant_bits)
        self.act_quant_method = str(activation_quant_method)
        self.act_quant_percentile = float(activation_quant_percentile)
        _configure_spin_state(
            self.spin_enabled_flag,
            self.spin_signs,
            enabled=bool(spin),
            random_signs=bool(spin_random),
            seed=int(spin_seed),
        )
        self._assign_float_state(module.weight.data, None if module.bias is None else module.bias.data, calibration_inputs)
        return self

    def runtime_linear(self, x: torch.Tensor, *, backend: str | None = None) -> torch.Tensor:
        del backend
        target_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        target_device = x.device
        x_local = x.to(device=target_device, dtype=target_dtype)
        if _spin_enabled(self.spin_enabled_flag):
            x_local = apply_spin_transform(x_local, self.spin_signs.to(device=target_device))
        if _pre_scale_active(self.pre_scale):
            x_local = _apply_pre_scale_to_input(x_local, self.pre_scale.to(device=target_device))
        x_local = _apply_activation_quantization(
            x_local,
            mode=self.act_quant_mode,
            act_scale=self.act_scale.to(device=target_device),
            bits=int(self.act_quant_bits),
            method=self.act_quant_method,
            percentile=float(self.act_quant_percentile),
        )
        bias = self.runtime_bias(dtype=target_dtype, device=target_device)
        return runtime_fp8_linear(
            x_local,
            self.weight_fp8,
            self._amax_tracker,
            float(self.weight_scale.item()),
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.runtime_linear(x)


def prepare_bitnet_qat_linear_modules(
    model: nn.Module,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    *,
    scale_layout: str = "runtime_row",
    group_size: int = 64,
    eps: float = 1e-8,
) -> Dict[str, TrainableBitNetLinear]:
    """Replace ``nn.Linear`` modules with trainable runtime-row BitNet QAT layers."""
    replacements: Dict[str, TrainableBitNetLinear] = {}
    include_list = None if include is None else list(include)
    exclude_list = None if exclude is None else list(exclude)
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if include_list is not None and len(include_list) > 0 and not any(p in name for p in include_list):
            continue
        if exclude_list is not None and any(p in name for p in exclude_list):
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        attr_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        qat = TrainableBitNetLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            scale_layout=scale_layout,
            group_size=group_size,
            eps=eps,
        ).to(device=module.weight.device, dtype=module.weight.dtype)
        qat.from_float(module)
        setattr(parent, attr_name, qat)
        replacements[name] = qat
    return replacements


def quantize_linear_modules(
    model: nn.Module,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    *,
    scheme: str = "int8",
    calibration: str = "absmax",
    percentile: float = 0.999,
    calibration_inputs: Optional[Dict[str, torch.Tensor]] = None,
    weight_opt: str = "none",
    activation_quant: str = "none",
    activation_quant_bits: int = 8,
    activation_quant_method: str = "absmax",
    activation_quant_percentile: float = 0.999,
    spin: bool = False,
    spin_random: bool = True,
    spin_seed: int = 0,
) -> Dict[str, nn.Module]:
    """Replace nn.Linear modules with quantized linear wrappers in-place.

    Returns a mapping of module-name -> quantized module.
    """
    replacements: Dict[str, nn.Module] = {}
    include_list = None if include is None else list(include)
    exclude_list = None if exclude is None else list(exclude)
    quant_scheme = str(scheme).strip().lower() or "int8"
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if include_list is not None and len(include_list) > 0 and not any(p in name for p in include_list):
            continue
        if exclude_list is not None and any(p in name for p in exclude_list):
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        attr_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        module_spin_seed = int(spin_seed) + len(replacements)
        module_calibration_inputs = None if calibration_inputs is None else calibration_inputs.get(name)
        if quant_scheme == "int8":
            ql = QuantizedLinearInt8(module.in_features, module.out_features, bias=(module.bias is not None))
            ql.from_float(
                module,
                calibration=calibration,
                percentile=percentile,
                calibration_inputs=module_calibration_inputs,
                weight_opt=weight_opt,
                activation_quant=activation_quant,
                activation_quant_bits=activation_quant_bits,
                activation_quant_method=activation_quant_method,
                activation_quant_percentile=activation_quant_percentile,
                spin=spin,
                spin_random=spin_random,
                spin_seed=module_spin_seed,
            )
        elif quant_scheme == "bitnet":
            ql = QuantizedLinearBitNet(module.in_features, module.out_features, bias=(module.bias is not None))
            ql.from_float(
                module,
                calibration=calibration,
                percentile=percentile,
                calibration_inputs=module_calibration_inputs,
                weight_opt=weight_opt,
                activation_quant=activation_quant,
                activation_quant_bits=activation_quant_bits,
                activation_quant_method=activation_quant_method,
                activation_quant_percentile=activation_quant_percentile,
                spin=spin,
                spin_random=spin_random,
                spin_seed=module_spin_seed,
            )
        elif quant_scheme in {"bitnet_qat", "qat_bitnet", "trainable_bitnet"}:
            ql = TrainableBitNetLinear(module.in_features, module.out_features, bias=(module.bias is not None)).to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            ql.from_float(module)
        elif quant_scheme == "int4":
            ql = QuantizedLinearInt4(module.in_features, module.out_features, bias=(module.bias is not None))
            ql.from_float(
                module,
                calibration=calibration,
                percentile=percentile,
                calibration_inputs=module_calibration_inputs,
                weight_opt=weight_opt,
                activation_quant=activation_quant,
                activation_quant_bits=activation_quant_bits,
                activation_quant_method=activation_quant_method,
                activation_quant_percentile=activation_quant_percentile,
                spin=spin,
                spin_random=spin_random,
                spin_seed=module_spin_seed,
            )
        elif quant_scheme == "nf4":
            ql = QuantizedLinearNF4(module.in_features, module.out_features, bias=(module.bias is not None))
            ql.from_float(
                module,
                calibration=calibration,
                percentile=percentile,
                calibration_inputs=module_calibration_inputs,
                weight_opt=weight_opt,
                activation_quant=activation_quant,
                activation_quant_bits=activation_quant_bits,
                activation_quant_method=activation_quant_method,
                activation_quant_percentile=activation_quant_percentile,
                spin=spin,
                spin_random=spin_random,
                spin_seed=module_spin_seed,
            )
        elif quant_scheme == "fp8":
            ql = QuantizedLinearFP8(module.in_features, module.out_features, bias=(module.bias is not None))
            ql.from_float(
                module,
                calibration=calibration,
                percentile=percentile,
                calibration_inputs=module_calibration_inputs,
                weight_opt=weight_opt,
                activation_quant=activation_quant,
                activation_quant_bits=activation_quant_bits,
                activation_quant_method=activation_quant_method,
                activation_quant_percentile=activation_quant_percentile,
                spin=spin,
                spin_random=spin_random,
                spin_seed=module_spin_seed,
            )
        else:
            raise ValueError(f"Unknown quantization scheme: {scheme}")
        setattr(parent, attr_name, ql)
        replacements[name] = ql
    return replacements


__all__ = [
    "_symmetric_per_channel_weight_quantize_int8",
    "_symmetric_per_channel_weight_quantize_int4",
    "_dequantize_int8_per_channel",
    "_dequantize_int4_per_channel",
    "fake_quantize_fp8",
    "fake_quantize_fp4",
    "apply_spin_transform",
    "undo_spin_transform",
    "fake_quantize_activation",
    "calibrate_activation_scale",
    "awq_optimize_pre_scale",
    "collect_linear_calibration_inputs",
    "calibrate_fp8_scale",
    "calibrate_int4_scales",
    "calibrate_nf4_scales",
    "QuantizedLinearInt8",
    "QuantizedLinearBitNet",
    "TrainableBitNetLinear",
    "QuantizedLinearInt4",
    "QuantizedLinearNF4",
    "QuantizedLinearFP8",
    "prepare_bitnet_qat_linear_modules",
    "quantize_linear_modules",
    "calibrate_int8_scales",
]
