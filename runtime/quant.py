from __future__ import annotations

from typing import Optional

import torch

from runtime.attention import scaled_dot_product_attention as runtime_scaled_dot_product_attention
from runtime.attention import select_attention_backend
from runtime.hardware import prefer_hopper_library_attention
from runtime.native import has_native_op, native_module
from runtime.ops import attention as runtime_attention
from runtime.ops import bitnet_transform_input as runtime_bitnet_transform_input
from runtime.ops import linear as runtime_linear
from tensor.numerics import mse_scale, percentile_scale


_NF4_CODEBOOK_VALUES = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def _should_use_eager_autograd_fallback(*tensors: torch.Tensor | None) -> bool:
    if not torch.is_grad_enabled():
        return False
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            return True
    return False


def _torch_compiler_is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    return bool(callable(is_compiling) and is_compiling())


def _register_bitnet_int8_linear_from_float_compile_op():
    custom_op = getattr(torch.library, "custom_op", None)
    if custom_op is None:
        return None
    try:
        @custom_op("model_stack::bitnet_int8_linear_from_float", mutates_args=())
        def op(
            x: torch.Tensor,
            qweight: torch.Tensor,
            inv_scale: torch.Tensor,
            bias: Optional[torch.Tensor],
            pre_scale: Optional[torch.Tensor],
            act_quant_mode: str,
            act_quant_method: str,
            act_quant_bits: int,
            act_quant_percentile: float,
            act_scale: Optional[torch.Tensor],
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            module = native_module()
            if module is None or not hasattr(module, "bitnet_int8_linear_from_float_forward"):
                raise RuntimeError("bitnet_int8_linear_from_float compile op requires the native backend")
            return module.bitnet_int8_linear_from_float_forward(
                x,
                qweight,
                inv_scale,
                bias,
                pre_scale,
                act_quant_mode,
                act_quant_method,
                int(act_quant_bits),
                float(act_quant_percentile),
                act_scale,
                out_dtype,
            )

        @op.register_fake
        def _(
            x: torch.Tensor,
            qweight: torch.Tensor,
            inv_scale: torch.Tensor,
            bias: Optional[torch.Tensor],
            pre_scale: Optional[torch.Tensor],
            act_quant_mode: str,
            act_quant_method: str,
            act_quant_bits: int,
            act_quant_percentile: float,
            act_scale: Optional[torch.Tensor],
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            return x.new_empty((*x.shape[:-1], qweight.shape[0]), dtype=out_dtype)

        return op
    except Exception:
        try:
            return torch.ops.model_stack.bitnet_int8_linear_from_float
        except Exception:
            return None


_BITNET_INT8_LINEAR_FROM_FLOAT_COMPILE_OP = _register_bitnet_int8_linear_from_float_compile_op()


def _register_int8_quantize_activation_transpose_compile_op():
    custom_op = getattr(torch.library, "custom_op", None)
    if custom_op is None:
        return None
    try:
        @custom_op("model_stack::int8_quantize_activation_transpose", mutates_args=())
        def op(
            x: torch.Tensor,
            provided_scale: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            module = native_module()
            if module is None or not hasattr(module, "int8_quantize_activation_transpose_forward"):
                raise RuntimeError("int8_quantize_activation_transpose compile op requires the native backend")
            return module.int8_quantize_activation_transpose_forward(x, provided_scale)

        @op.register_fake
        def _(
            x: torch.Tensor,
            provided_scale: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del provided_scale
            cols = x.shape[-1]
            rows = x.numel() // cols
            return (
                x.new_empty((cols, rows), dtype=torch.int8),
                x.new_empty((cols,), dtype=torch.float32),
            )

        return op
    except Exception:
        try:
            return torch.ops.model_stack.int8_quantize_activation_transpose
        except Exception:
            return None


def _register_int8_linear_compile_op():
    custom_op = getattr(torch.library, "custom_op", None)
    if custom_op is None:
        return None
    try:
        @custom_op("model_stack::int8_linear", mutates_args=())
        def op(
            qx: torch.Tensor,
            x_scale: torch.Tensor,
            qweight: torch.Tensor,
            inv_scale: torch.Tensor,
            bias: Optional[torch.Tensor],
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            module = native_module()
            if module is None or not hasattr(module, "int8_linear_forward"):
                raise RuntimeError("int8_linear compile op requires the native backend")
            return module.int8_linear_forward(qx, x_scale, qweight, inv_scale, bias, out_dtype)

        @op.register_fake
        def _(
            qx: torch.Tensor,
            x_scale: torch.Tensor,
            qweight: torch.Tensor,
            inv_scale: torch.Tensor,
            bias: Optional[torch.Tensor],
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            del x_scale, inv_scale, bias
            return qx.new_empty((*qx.shape[:-1], qweight.shape[0]), dtype=out_dtype)

        return op
    except Exception:
        try:
            return torch.ops.model_stack.int8_linear
        except Exception:
            return None


def _register_int8_linear_grad_weight_from_float_compile_op():
    custom_op = getattr(torch.library, "custom_op", None)
    if custom_op is None:
        return None
    try:
        @custom_op("model_stack::int8_linear_grad_weight_from_float", mutates_args=())
        def op(
            grad_out: torch.Tensor,
            x: torch.Tensor,
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            module = native_module()
            if module is None or not hasattr(module, "int8_linear_grad_weight_from_float_forward"):
                raise RuntimeError("int8_linear_grad_weight_from_float compile op requires the native backend")
            return module.int8_linear_grad_weight_from_float_forward(grad_out, x, out_dtype)

        @op.register_fake
        def _(
            grad_out: torch.Tensor,
            x: torch.Tensor,
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            return x.new_empty((grad_out.shape[-1], x.shape[-1]), dtype=out_dtype)

        return op
    except Exception:
        try:
            return torch.ops.model_stack.int8_linear_grad_weight_from_float
        except Exception:
            return None


_INT8_QUANTIZE_ACTIVATION_TRANSPOSE_COMPILE_OP = _register_int8_quantize_activation_transpose_compile_op()
_INT8_LINEAR_COMPILE_OP = _register_int8_linear_compile_op()
_INT8_LINEAR_GRAD_WEIGHT_FROM_FLOAT_COMPILE_OP = _register_int8_linear_grad_weight_from_float_compile_op()


def _linear_shape_signature(x: torch.Tensor, out_features: int) -> tuple[int, int, int]:
    in_features = int(x.shape[-1])
    rows = int(x.numel() // max(in_features, 1))
    return rows, int(out_features), in_features


def _broadcast_scale(scale: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    scale_t = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, device=like.device)
    scale_t = scale_t.to(device=like.device, dtype=torch.float32)
    while scale_t.ndim < like.ndim:
        scale_t = scale_t.unsqueeze(-1)
    return scale_t


def _dequantize_to_dtype(qx: torch.Tensor, scale: torch.Tensor | float, *, dtype: torch.dtype) -> torch.Tensor:
    scale_t = _broadcast_scale(scale, qx)
    return qx.to(device=scale_t.device, dtype=torch.float32).mul(scale_t).to(dtype=dtype)


def _unpack_packed_int4(qweight_packed: torch.Tensor, *, original_last_dim: int) -> torch.Tensor:
    packed = qweight_packed.to(dtype=torch.int16)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)[..., :original_last_dim]
    return (unpacked - 8).to(dtype=torch.int8)


def _dequantize_packed_int4_weight(
    qweight_packed: torch.Tensor,
    inv_scale: torch.Tensor,
    *,
    original_last_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    qweight = _unpack_packed_int4(qweight_packed, original_last_dim=original_last_dim)
    scale = _broadcast_scale(inv_scale, qweight)
    return qweight.to(device=scale.device, dtype=torch.float32).mul(scale).to(dtype=dtype)


def _nf4_codebook(*, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(_NF4_CODEBOOK_VALUES, device=device, dtype=dtype)


def _pack_packed_uint4(qcodes: torch.Tensor) -> torch.Tensor:
    if qcodes.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("NF4 code tensor must use an integer dtype")
    q = qcodes.to(dtype=torch.int16)
    if bool(((q < 0) | (q > 15)).any()):
        raise ValueError("NF4 codes must be in [0, 15]")
    if q.size(-1) % 2 != 0:
        pad = torch.zeros((*q.shape[:-1], 1), dtype=q.dtype, device=q.device)
        q = torch.cat([q, pad], dim=-1)
    low = q[..., 0::2]
    high = q[..., 1::2]
    return (low | (high << 4)).to(dtype=torch.uint8)


def _unpack_packed_uint4(qweight_packed: torch.Tensor, *, original_last_dim: int) -> torch.Tensor:
    if qweight_packed.dtype != torch.uint8:
        raise TypeError("packed NF4 tensor must use uint8 storage")
    packed = qweight_packed.to(dtype=torch.int16)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).flatten(-2)[..., :original_last_dim].to(dtype=torch.uint8)


def _dequantize_nf4_codes(
    qcodes: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    codebook = _nf4_codebook(device=qcodes.device, dtype=torch.float32)
    decoded = codebook[qcodes.to(dtype=torch.long)]
    scale_t = _broadcast_scale(scale, decoded)
    return decoded.mul(scale_t).to(dtype=dtype)


def _dequantize_packed_nf4_weight(
    qweight_packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    original_last_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    qcodes = _unpack_packed_uint4(qweight_packed, original_last_dim=original_last_dim)
    return _dequantize_nf4_codes(qcodes, scale, dtype=dtype)


_BITNET_LAYOUT_HEADER_LEN = 13
_BITNET_IDX_FORMAT_VERSION = 0
_BITNET_IDX_TILE_N = 1
_BITNET_IDX_TILE_K = 2
_BITNET_IDX_LOGICAL_OUT = 3
_BITNET_IDX_LOGICAL_IN = 4
_BITNET_IDX_PADDED_OUT = 5
_BITNET_IDX_PADDED_IN = 6
_BITNET_IDX_SCALE_GRANULARITY = 7
_BITNET_IDX_SCALE_GROUP_SIZE = 8
_BITNET_IDX_INTERLEAVE_MODE = 9
_BITNET_IDX_ARCH_MIN = 10
_BITNET_IDX_SEGMENT_COUNT = 11
_BITNET_IDX_FLAGS = 12


def _unpack_packed_bitnet(qweight_packed: torch.Tensor, *, original_last_dim: int) -> torch.Tensor:
    if qweight_packed.dtype != torch.uint8:
        raise TypeError("packed BitNet weight tensor must use uint8 storage")
    packed = qweight_packed.to(dtype=torch.int16)
    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03
    return torch.stack((c0, c1, c2, c3), dim=-1).flatten(-2)[..., :original_last_dim].to(dtype=torch.int8)


def _bitnet_row_scales(
    scale_values: torch.Tensor,
    *,
    logical_out_features: int,
    scale_granularity: int,
    scale_group_size: int,
    segment_offsets: torch.Tensor,
) -> torch.Tensor:
    row_scales = torch.zeros(logical_out_features, device=scale_values.device, dtype=torch.float32)
    values = scale_values.flatten()
    if scale_granularity == 0:
        if values.numel() < 1:
            raise ValueError("BitNet per-matrix scaling requires at least one scale value")
        row_scales.fill_(float(values[0].item()))
        return row_scales
    if scale_granularity == 1:
        segment_count = segment_offsets.numel() - 1
        if values.numel() != segment_count:
            raise ValueError(
                f"BitNet per-segment scaling requires {segment_count} scale values, got {values.numel()}"
            )
        for idx in range(segment_count):
            start = int(segment_offsets[idx].item())
            end = int(segment_offsets[idx + 1].item())
            row_scales[start:end] = float(values[idx].item())
        return row_scales
    if scale_granularity == 2:
        if scale_group_size <= 0:
            raise ValueError("BitNet per-output-group scaling requires a positive scale_group_size")
        expected_groups = (logical_out_features + scale_group_size - 1) // scale_group_size
        if values.numel() != expected_groups:
            raise ValueError(
                f"BitNet per-output-group scaling requires {expected_groups} scale values, got {values.numel()}"
            )
        for idx in range(expected_groups):
            start = idx * scale_group_size
            end = min(logical_out_features, start + scale_group_size)
            row_scales[start:end] = float(values[idx].item())
        return row_scales
    raise ValueError(f"Unsupported BitNet scale granularity: {scale_granularity}")


def _normalize_bitnet_layout_inputs(
    packed_weight: torch.Tensor,
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
    *,
    target_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    packed_cast = packed_weight.to(device=target_device, dtype=torch.uint8).contiguous()
    scale_cast = scale_values.to(device=target_device, dtype=torch.float32).contiguous()
    header_cast = layout_header.to(device=target_device, dtype=torch.int32).contiguous()
    offsets_cast = segment_offsets.to(device=target_device, dtype=torch.int32).contiguous()
    if packed_cast.ndim != 2:
        raise ValueError("BitNet packed_weight must be rank-2")
    if scale_cast.ndim != 1:
        raise ValueError("BitNet scale_values must be rank-1")
    if header_cast.ndim != 1 or int(header_cast.numel()) < _BITNET_LAYOUT_HEADER_LEN:
        raise ValueError(f"BitNet layout_header must be rank-1 with at least {_BITNET_LAYOUT_HEADER_LEN} entries")
    if offsets_cast.ndim != 1 or int(offsets_cast.numel()) < 2:
        raise ValueError("BitNet segment_offsets must be rank-1 with at least two entries")
    meta = {
        "format_version": int(header_cast[_BITNET_IDX_FORMAT_VERSION].item()),
        "tile_n": int(header_cast[_BITNET_IDX_TILE_N].item()),
        "tile_k": int(header_cast[_BITNET_IDX_TILE_K].item()),
        "logical_out_features": int(header_cast[_BITNET_IDX_LOGICAL_OUT].item()),
        "logical_in_features": int(header_cast[_BITNET_IDX_LOGICAL_IN].item()),
        "padded_out_features": int(header_cast[_BITNET_IDX_PADDED_OUT].item()),
        "padded_in_features": int(header_cast[_BITNET_IDX_PADDED_IN].item()),
        "scale_granularity": int(header_cast[_BITNET_IDX_SCALE_GRANULARITY].item()),
        "scale_group_size": int(header_cast[_BITNET_IDX_SCALE_GROUP_SIZE].item()),
        "interleave_mode": int(header_cast[_BITNET_IDX_INTERLEAVE_MODE].item()),
        "arch_min": int(header_cast[_BITNET_IDX_ARCH_MIN].item()),
        "segment_count": int(header_cast[_BITNET_IDX_SEGMENT_COUNT].item()),
        "flags": int(header_cast[_BITNET_IDX_FLAGS].item()),
    }
    if meta["format_version"] != 1:
        raise ValueError(f"Unsupported BitNet format_version: {meta['format_version']}")
    if meta["logical_out_features"] <= 0 or meta["logical_in_features"] <= 0:
        raise ValueError("BitNet logical dimensions must be positive")
    if meta["padded_out_features"] < meta["logical_out_features"] or meta["padded_in_features"] < meta["logical_in_features"]:
        raise ValueError("BitNet padded dimensions must be at least the logical dimensions")
    if meta["interleave_mode"] != 1:
        raise ValueError(f"Unsupported BitNet interleave_mode: {meta['interleave_mode']}")
    expected_cols = (meta["padded_in_features"] + 3) // 4
    if int(packed_cast.shape[0]) != meta["padded_out_features"] or int(packed_cast.shape[1]) != expected_cols:
        raise ValueError(
            "BitNet packed_weight shape mismatch: "
            f"expected ({meta['padded_out_features']}, {expected_cols}), got {tuple(packed_cast.shape)}"
        )
    if offsets_cast.numel() != meta["segment_count"] + 1:
        raise ValueError(
            "BitNet segment_offsets length mismatch: "
            f"expected {meta['segment_count'] + 1}, got {int(offsets_cast.numel())}"
        )
    if int(offsets_cast[0].item()) != 0 or int(offsets_cast[-1].item()) != meta["logical_out_features"]:
        raise ValueError("BitNet segment_offsets must start at 0 and end at logical_out_features")
    if bool((offsets_cast[1:] < offsets_cast[:-1]).any()):
        raise ValueError("BitNet segment_offsets must be non-decreasing")
    return packed_cast, scale_cast, header_cast, offsets_cast, meta


def _dequantize_packed_bitnet_weight(
    packed_weight: torch.Tensor,
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    _, scale_cast, _, offsets_cast, meta = _normalize_bitnet_layout_inputs(
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        target_device=packed_weight.device,
    )
    unpacked = _unpack_packed_bitnet(
        packed_weight,
        original_last_dim=meta["logical_in_features"],
    )[: meta["logical_out_features"]]
    code_to_value = torch.tensor([-1.0, 0.0, 1.0, 0.0], device=unpacked.device, dtype=torch.float32)
    row_scales = _bitnet_row_scales(
        scale_cast,
        logical_out_features=meta["logical_out_features"],
        scale_granularity=meta["scale_granularity"],
        scale_group_size=meta["scale_group_size"],
        segment_offsets=offsets_cast,
    )
    decoded = code_to_value[unpacked.to(dtype=torch.long)]
    return decoded.mul(row_scales.unsqueeze(-1)).to(dtype=dtype)


def per_channel_absmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    return x.abs().amax(dim=axis, keepdim=True)


def nf4_quantize(x: torch.Tensor, scale: torch.Tensor):
    meta = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, device=x.device, dtype=torch.float32)
    meta = meta.to(device=x.device, dtype=torch.float32)
    scale_t = _broadcast_scale(meta, x).clamp_min(1e-8)
    x_norm = x.to(device=scale_t.device, dtype=torch.float32).div(scale_t)
    codebook = _nf4_codebook(device=x_norm.device, dtype=torch.float32)
    x_norm = x_norm.clamp(min=float(codebook[0].item()), max=float(codebook[-1].item()))
    distances = (x_norm.unsqueeze(-1) - codebook).abs()
    qcodes = torch.argmin(distances, dim=-1).to(dtype=torch.uint8)
    return qcodes, meta


def nf4_dequantize(qx: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    target_dtype = meta.dtype if meta.dtype.is_floating_point else torch.float32
    return _dequantize_nf4_codes(qx.to(dtype=torch.uint8), meta, dtype=target_dtype)


def quantize_activation_int8_rowwise(
    x: torch.Tensor,
    *,
    scale: torch.Tensor | float | None = None,
    method: str = "absmax",
    percentile: float = 0.999,
    bits: int = 8,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = float(_bitnet_quant_max(int(bits)))
    x_float = x.float()
    flat = x_float.reshape(-1, x_float.shape[-1])
    if flat.numel() == 0:
        qx = torch.empty_like(x, dtype=torch.int8)
        rows = flat.shape[0]
        return qx, torch.ones(rows, device=x.device, dtype=torch.float32)

    if scale is None:
        method_name = str(method).strip().lower()
        if method_name == "absmax":
            clip = flat.abs().amax(dim=-1)
        elif method_name == "percentile":
            clip = torch.stack(
                [torch.quantile(row.abs(), float(percentile)) for row in flat],
                dim=0,
            )
        elif method_name == "mse":
            candidates = torch.tensor(
                [1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8],
                device=flat.device,
                dtype=torch.float32,
            )
            per_row_absmax = flat.abs().amax(dim=-1).clamp_min(eps)
            losses = []
            for ratio in candidates:
                clip_candidate = per_row_absmax * ratio
                scale_candidate = (clip_candidate / qmax).clamp_min(eps)
                q_candidate = torch.round(flat / scale_candidate.unsqueeze(-1)).clamp_(-qmax, qmax)
                recon = q_candidate * scale_candidate.unsqueeze(-1)
                losses.append(((recon - flat) ** 2).mean(dim=-1))
            loss_tensor = torch.stack(losses, dim=0)
            best = torch.argmin(loss_tensor, dim=0)
            clip = per_row_absmax * candidates[best]
        else:
            raise ValueError(f"Unknown activation quantization method: {method}")
        scale_row = (clip.clamp_min(eps) / qmax).to(dtype=torch.float32)
    else:
        scale_row = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, device=x.device, dtype=torch.float32)
        scale_row = scale_row.to(device=x.device, dtype=torch.float32)
        if scale_row.ndim == 0:
            scale_row = scale_row.expand(flat.shape[0])
        else:
            scale_row = scale_row.reshape(-1)
            if scale_row.numel() == 1:
                scale_row = scale_row.expand(flat.shape[0])
        if scale_row.numel() != flat.shape[0]:
            raise ValueError(
                "Activation scale shape mismatch: "
                f"expected {flat.shape[0]} row scales, got {scale_row.numel()}"
            )
        scale_row = scale_row.clamp_min(eps)

    q_flat = torch.round(flat / scale_row.unsqueeze(-1)).clamp_(-qmax, qmax).to(dtype=torch.int8)
    return q_flat.view_as(x), scale_row.contiguous()


def _dequantize_int8_activation(
    qx: torch.Tensor,
    scale_row: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    flat = qx.reshape(-1, qx.shape[-1]).to(dtype=torch.float32)
    dequant = flat * scale_row.to(device=qx.device, dtype=torch.float32).reshape(-1, 1)
    return dequant.view_as(qx).to(dtype=dtype)


def _dequantize_int8_weight(
    qweight: torch.Tensor,
    inv_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    scale = _broadcast_scale(inv_scale, qweight)
    return qweight.to(device=scale.device, dtype=torch.float32).mul(scale).to(dtype=dtype)


def _coerce_optional_row_scale(
    scale: torch.Tensor | float | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if scale is None:
        return None
    scale_t = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, device=device, dtype=torch.float32)
    return scale_t.to(device=device, dtype=torch.float32).reshape(-1).contiguous()


def _bitnet_quant_max(bits: int) -> int:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return (1 << (bits - 1)) - 1


def _calibrate_bitnet_activation_scale(
    x: torch.Tensor,
    *,
    method: str,
    bits: int,
    percentile: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    qmax = float(_bitnet_quant_max(bits))
    x_float = x.detach().float()
    method_name = str(method).strip().lower()
    if method_name == "absmax":
        clip = x_float.abs().amax()
    elif method_name == "percentile":
        clip = percentile_scale(x_float, p=float(percentile))
    elif method_name == "mse":
        clip = mse_scale(x_float)
    else:
        raise ValueError(f"Unknown BitNet activation calibration method: {method}")
    return (clip.clamp_min(eps) / qmax).to(device=x.device, dtype=torch.float32)


def int8_linear_from_quantized_activation(
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    qweight: torch.Tensor,
    inv_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dtype = torch.float32 if out_dtype is None else out_dtype
    target_device = qweight.device
    qx_cast = qx.to(device=target_device, dtype=torch.int8).contiguous()
    row_scale_cast = x_scale.to(device=target_device, dtype=torch.float32).reshape(-1).contiguous()
    qweight_cast = qweight.to(device=target_device, dtype=torch.int8).contiguous()
    scale_cast = inv_scale.to(device=target_device, dtype=torch.float32).contiguous()
    bias_cast = None if bias is None else bias.to(device=target_device, dtype=target_dtype)
    if qx_cast.is_cuda and _torch_compiler_is_compiling() and _INT8_LINEAR_COMPILE_OP is not None:
        return _INT8_LINEAR_COMPILE_OP(
            qx_cast,
            row_scale_cast,
            qweight_cast,
            scale_cast,
            bias_cast,
            target_dtype,
        )
    if qx_cast.is_cuda and not _should_use_eager_autograd_fallback(bias_cast):
        module = native_module()
        if (
            has_native_op("int8_linear")
            and module is not None
            and hasattr(module, "int8_linear_forward")
        ):
            return module.int8_linear_forward(
                qx_cast,
                row_scale_cast,
                qweight_cast,
                scale_cast,
                bias_cast,
                target_dtype,
            )

    x_dequant = _dequantize_int8_activation(qx_cast, row_scale_cast, dtype=target_dtype)
    weight = _dequantize_int8_weight(qweight_cast, scale_cast, dtype=target_dtype)
    return runtime_linear(x_dequant, weight, bias_cast)


def int8_quantize_activation_transpose(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_cast = x if x.dtype.is_floating_point else x.float()
    scale_cast = None if scale is None else scale.to(device=x_cast.device, dtype=torch.float32).reshape(-1).contiguous()
    if (
        x_cast.is_cuda
        and _torch_compiler_is_compiling()
        and _INT8_QUANTIZE_ACTIVATION_TRANSPOSE_COMPILE_OP is not None
    ):
        return _INT8_QUANTIZE_ACTIVATION_TRANSPOSE_COMPILE_OP(x_cast, scale_cast)
    if x_cast.is_cuda:
        module = native_module()
        if (
            has_native_op("int8_quantize_activation_transpose")
            and module is not None
            and hasattr(module, "int8_quantize_activation_transpose_forward")
        ):
            return module.int8_quantize_activation_transpose_forward(x_cast, scale_cast)
    rows = x_cast.numel() // max(int(x_cast.shape[-1]), 1)
    x_2d = x_cast.reshape(rows, x_cast.shape[-1])
    qx, row_scale = quantize_activation_int8_rowwise(x_2d.t().contiguous(), scale=scale_cast)
    return qx.contiguous(), row_scale.contiguous()


def int8_linear_grad_weight_from_float(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dtype = x.dtype if out_dtype is None else out_dtype
    go_cast = grad_out if grad_out.dtype.is_floating_point else grad_out.float()
    x_cast = x if x.dtype.is_floating_point else x.float()
    if (
        go_cast.is_cuda
        and x_cast.is_cuda
        and _torch_compiler_is_compiling()
        and _INT8_LINEAR_GRAD_WEIGHT_FROM_FLOAT_COMPILE_OP is not None
    ):
        return _INT8_LINEAR_GRAD_WEIGHT_FROM_FLOAT_COMPILE_OP(go_cast, x_cast, target_dtype)
    if go_cast.is_cuda and x_cast.is_cuda:
        module = native_module()
        if (
            has_native_op("int8_linear_grad_weight_from_float")
            and module is not None
            and hasattr(module, "int8_linear_grad_weight_from_float_forward")
        ):
            return module.int8_linear_grad_weight_from_float_forward(go_cast, x_cast, target_dtype)
    qx_t, x_t_scale = int8_quantize_activation_transpose(x_cast)
    qgo_t, go_t_scale = int8_quantize_activation_transpose(go_cast)
    return int8_linear_from_quantized_activation(
        qgo_t,
        go_t_scale,
        qx_t,
        x_t_scale,
        None,
        out_dtype=target_dtype,
    )


def int8_matmul_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scales: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    compute_dtype = (
        out_dtype
        if out_dtype is not None
        else (torch.float16 if q.is_cuda else torch.float32)
    )
    q_cast = q.to(dtype=torch.int8).contiguous()
    k_cast = k.to(device=q_cast.device, dtype=torch.int8).contiguous()
    v_cast = v.to(device=q_cast.device, dtype=torch.int8).contiguous()
    q_scales_raw = q_scales.to(device=q_cast.device, dtype=torch.float32)
    k_scales_raw = k_scales.to(device=q_cast.device, dtype=torch.float32)
    v_scales_raw = v_scales.to(device=q_cast.device, dtype=torch.float32)
    attn_mask_cast = None if attn_mask is None else attn_mask.to(device=q_cast.device)
    native_attn_mask = None
    native_mask_ok = attn_mask_cast is None
    if attn_mask_cast is not None:
        native_mask_ok = (
            attn_mask_cast.is_cuda
            and attn_mask_cast.ndim == 4
            and attn_mask_cast.shape[0] == q_cast.shape[0]
            and attn_mask_cast.shape[2] == q_cast.shape[2]
            and attn_mask_cast.shape[3] == k_cast.shape[2]
            and attn_mask_cast.shape[1] in (1, q_cast.shape[1])
        )
        if native_mask_ok:
            native_attn_mask = attn_mask_cast.contiguous()
            if native_attn_mask.dtype != torch.bool:
                native_attn_mask = native_attn_mask.to(dtype=torch.float32).contiguous()
    prefer_library = prefer_hopper_library_attention(
        device=q_cast.device,
        dtype=compute_dtype,
        q_seq=int(q_cast.shape[2]),
        kv_seq=int(k_cast.shape[2]),
    )
    if q_cast.is_cuda and native_mask_ok and not _should_use_eager_autograd_fallback():
        module = native_module()
        if has_native_op("int8_attention") and module is not None and hasattr(module, "int8_attention_forward"):
            return module.int8_attention_forward(
                q_cast,
                q_scales_raw.reshape(-1).contiguous(),
                k_cast,
                k_scales_raw.reshape(-1).contiguous(),
                v_cast,
                v_scales_raw.reshape(-1).contiguous(),
                native_attn_mask,
                bool(is_causal),
                scale,
                compute_dtype,
            )
    qf = _dequantize_int8_activation(q_cast, q_scales_raw.reshape(-1), dtype=compute_dtype)
    kf = _dequantize_int8_activation(k_cast, k_scales_raw.reshape(-1), dtype=compute_dtype)
    vf = _dequantize_int8_activation(v_cast, v_scales_raw.reshape(-1), dtype=compute_dtype)
    backend = None
    if prefer_library:
        backend = select_attention_backend(
            is_causal=bool(is_causal),
            dtype=compute_dtype,
            seq=int(q_cast.shape[2]),
            heads=int(q_cast.shape[1]),
            device=q_cast.device,
            kv_seq=int(k_cast.shape[2]),
        )
    if backend is None:
        return runtime_attention(qf, kf, vf, attn_mask=attn_mask_cast, is_causal=is_causal, scale=scale)
    return runtime_scaled_dot_product_attention(
        qf,
        kf,
        vf,
        attn_mask=attn_mask_cast,
        dropout_p=0.0,
        backend=backend,
        is_causal=is_causal,
        scale=scale,
    )


def int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    out_dtype: torch.dtype | None = None,
    q_scale: torch.Tensor | float | None = None,
    k_scale: torch.Tensor | float | None = None,
    v_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    compute_dtype = out_dtype if out_dtype is not None else (q.dtype if q.dtype.is_floating_point else torch.float32)
    target_device = q.device
    q_cast = q.to(device=target_device, dtype=compute_dtype)
    k_cast = k.to(device=target_device, dtype=compute_dtype)
    v_cast = v.to(device=target_device, dtype=compute_dtype)
    attn_mask_cast = None if attn_mask is None else attn_mask.to(device=target_device)
    q_scale_cast = _coerce_optional_row_scale(q_scale, device=target_device)
    k_scale_cast = _coerce_optional_row_scale(k_scale, device=target_device)
    v_scale_cast = _coerce_optional_row_scale(v_scale, device=target_device)
    if q_cast.is_cuda and not _should_use_eager_autograd_fallback(q_cast, k_cast, v_cast):
        module = native_module()
        if (
            has_native_op("int8_attention_from_float")
            and module is not None
            and hasattr(module, "int8_attention_from_float_forward")
        ):
            return module.int8_attention_from_float_forward(
                q_cast,
                k_cast,
                v_cast,
                attn_mask_cast,
                bool(is_causal),
                scale,
                compute_dtype,
                q_scale_cast,
                k_scale_cast,
                v_scale_cast,
            )

    qq, q_scale_row = quantize_activation_int8_rowwise(q_cast, scale=q_scale_cast)
    kk, k_scale_row = quantize_activation_int8_rowwise(k_cast, scale=k_scale_cast)
    vv, v_scale_row = quantize_activation_int8_rowwise(v_cast, scale=v_scale_cast)
    return int8_matmul_qkv(
        qq,
        kk,
        vv,
        q_scale_row,
        k_scale_row,
        v_scale_row,
        attn_mask_cast,
        is_causal=is_causal,
        scale=scale,
        out_dtype=compute_dtype,
    )


def int8_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    inv_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    act_scale: torch.Tensor | float | None = None,
    act_method: str = "absmax",
    act_percentile: float = 0.999,
) -> torch.Tensor:
    compute_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    target_device = qweight.device
    x_cast = x.to(device=target_device, dtype=compute_dtype)
    bias_cast = None if bias is None else bias.to(device=target_device, dtype=compute_dtype)
    if _should_use_eager_autograd_fallback(x_cast, bias):
        weight = _dequantize_int8_weight(
            qweight.to(device=target_device, dtype=torch.int8),
            inv_scale.to(device=target_device, dtype=torch.float32),
            dtype=compute_dtype,
        )
        return runtime_linear(x_cast, weight, bias_cast)
    act_scale_cast = _coerce_optional_row_scale(act_scale, device=target_device)
    if x_cast.is_cuda:
        module = native_module()
        if (
            has_native_op("int8_linear_from_float")
            and module is not None
            and hasattr(module, "int8_linear_from_float_forward")
        ):
            return module.int8_linear_from_float_forward(
                x_cast,
                qweight.to(device=target_device, dtype=torch.int8).contiguous(),
                inv_scale.to(device=target_device, dtype=torch.float32).contiguous(),
                bias_cast,
                act_scale_cast,
                compute_dtype,
            )

    qx_cast, row_scale = quantize_activation_int8_rowwise(
        x_cast,
        scale=act_scale_cast,
        method=act_method,
        percentile=act_percentile,
    )
    return int8_linear_from_quantized_activation(
        qx_cast,
        row_scale,
        qweight,
        inv_scale,
        bias_cast,
        out_dtype=compute_dtype,
    )


def bitnet_int8_linear_from_float(
    x: torch.Tensor,
    qweight: torch.Tensor,
    inv_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    pre_scale: torch.Tensor | None = None,
    act_quant_mode: str = "dynamic_int8",
    act_scale: torch.Tensor | float | None = None,
    act_quant_bits: int = 8,
    act_quant_method: str = "absmax",
    act_quant_percentile: float = 0.999,
) -> torch.Tensor:
    compute_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    target_device = qweight.device
    x_cast = x.to(device=target_device, dtype=compute_dtype)
    qweight_cast = qweight.to(device=target_device, dtype=torch.int8).contiguous()
    inv_scale_cast = inv_scale.to(device=target_device, dtype=torch.float32).contiguous()
    bias_cast = None if bias is None else bias.to(device=target_device, dtype=compute_dtype)
    pre_scale_cast = None
    if pre_scale is not None:
        pre_scale_cast = pre_scale.to(device=target_device, dtype=compute_dtype).reshape(-1).contiguous()
    act_scale_cast = _coerce_optional_row_scale(act_scale, device=target_device)

    if (
        x_cast.is_cuda
        and _torch_compiler_is_compiling()
        and _BITNET_INT8_LINEAR_FROM_FLOAT_COMPILE_OP is not None
    ):
        return _BITNET_INT8_LINEAR_FROM_FLOAT_COMPILE_OP(
            x_cast,
            qweight_cast,
            inv_scale_cast,
            bias_cast,
            pre_scale_cast,
            str(act_quant_mode),
            str(act_quant_method),
            int(act_quant_bits),
            float(act_quant_percentile),
            act_scale_cast,
            compute_dtype,
        )

    if not _should_use_eager_autograd_fallback(x_cast, bias_cast):
        if x_cast.is_cuda:
            module = native_module()
            if (
                has_native_op("bitnet_int8_linear_from_float")
                and module is not None
                and hasattr(module, "bitnet_int8_linear_from_float_forward")
            ):
                return module.bitnet_int8_linear_from_float_forward(
                    x_cast,
                    qweight_cast,
                    inv_scale_cast,
                    bias_cast,
                    pre_scale_cast,
                    str(act_quant_mode),
                    str(act_quant_method),
                    int(act_quant_bits),
                    float(act_quant_percentile),
                    act_scale_cast,
                    compute_dtype,
                )

    x_local = x_cast
    if pre_scale_cast is not None:
        x_local = x_local / pre_scale_cast.view(*([1] * (x_local.ndim - 1)), -1)

    mode_name = str(act_quant_mode).strip().lower()
    if mode_name == "static_int8":
        if act_scale_cast is None:
            raise ValueError("BitNet static_int8 requires act_scale")
    elif mode_name != "dynamic_int8":
        raise ValueError("BitNet int8 from-float requires act_quant_mode in {dynamic_int8, static_int8}")

    qx_cast, row_scale = quantize_activation_int8_rowwise(
        x_local,
        scale=act_scale_cast if mode_name == "static_int8" else None,
        method=str(act_quant_method),
        percentile=float(act_quant_percentile),
        bits=int(act_quant_bits),
    )
    return int8_linear_from_quantized_activation(
        qx_cast,
        row_scale,
        qweight_cast,
        inv_scale_cast,
        bias_cast,
        out_dtype=compute_dtype,
    )


def fp8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    amax_tracker,
    scale: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    amax_tracker.update(weight_fp8)
    compute_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    target_device = weight_fp8.device
    x_cast = x.to(device=target_device, dtype=compute_dtype)
    bias_cast = None if bias is None else bias.to(device=target_device, dtype=compute_dtype)
    if x_cast.is_cuda and not _should_use_eager_autograd_fallback(x_cast, bias_cast):
        module = native_module()
        if (
            has_native_op("fp8_linear")
            and module is not None
            and hasattr(module, "fp8_linear_forward")
        ):
            return module.fp8_linear_forward(
                x_cast,
                weight_fp8.to(device=target_device).contiguous(),
                float(scale),
                bias_cast,
                compute_dtype,
            )
    weight = _dequantize_to_dtype(weight_fp8, float(scale), dtype=compute_dtype)
    return runtime_linear(x_cast, weight, bias_cast)


def nf4_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    compute_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    target_device = weight_packed.device
    x_cast = x.to(device=target_device, dtype=compute_dtype)
    packed_cast = weight_packed.to(device=target_device, dtype=torch.uint8)
    scale_cast = weight_scale.to(device=target_device, dtype=torch.float32)
    bias_cast = None if bias is None else bias.to(device=target_device, dtype=compute_dtype)
    if packed_cast.is_cuda and not _should_use_eager_autograd_fallback(x_cast, bias_cast):
        module = native_module()
        if (
            has_native_op("nf4_linear")
            and module is not None
            and hasattr(module, "nf4_linear_forward")
        ):
            return module.nf4_linear_forward(
                x_cast,
                packed_cast.contiguous(),
                scale_cast.contiguous(),
                bias_cast,
            )
    weight = _dequantize_packed_nf4_weight(
        packed_cast,
        scale_cast,
        original_last_dim=int(x_cast.shape[-1]),
        dtype=compute_dtype,
    )
    return runtime_linear(x_cast, weight, bias_cast)


def int4_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    inv_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    compute_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    target_device = weight_packed.device
    x_cast = x.to(device=target_device, dtype=compute_dtype)
    packed_cast = weight_packed.to(device=target_device, dtype=torch.uint8)
    scale_cast = inv_scale.to(device=target_device, dtype=torch.float32)
    bias_cast = None if bias is None else bias.to(device=target_device, dtype=compute_dtype)
    if packed_cast.is_cuda and not _should_use_eager_autograd_fallback(x_cast, bias_cast):
        module = native_module()
        if (
            has_native_op("int4_linear")
            and module is not None
            and hasattr(module, "int4_linear_forward")
        ):
            return module.int4_linear_forward(x_cast, packed_cast, scale_cast, bias_cast)
    weight = _dequantize_packed_int4_weight(
        packed_cast,
        scale_cast,
        original_last_dim=int(x_cast.shape[-1]),
        dtype=compute_dtype,
    )
    return runtime_linear(x_cast, weight, bias_cast)


def bitnet_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    compute_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    target_device = packed_weight.device
    x_cast = x.to(device=target_device, dtype=compute_dtype)
    packed_cast, scales_cast, header_cast, offsets_cast, meta = _normalize_bitnet_layout_inputs(
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        target_device=target_device,
    )
    if int(x_cast.shape[-1]) != meta["logical_in_features"]:
        raise ValueError(
            "BitNet input feature mismatch: "
            f"expected {meta['logical_in_features']}, got {int(x_cast.shape[-1])}"
        )
    bias_cast = None if bias is None else bias.to(device=target_device, dtype=compute_dtype)
    if bias_cast is not None and int(bias_cast.shape[0]) != meta["logical_out_features"]:
        raise ValueError(
            "BitNet bias size mismatch: "
            f"expected {meta['logical_out_features']}, got {int(bias_cast.shape[0])}"
        )
    if packed_cast.is_cuda and not _should_use_eager_autograd_fallback(x_cast, bias_cast):
        module = native_module()
        if has_native_op("bitnet_linear") and module is not None and hasattr(module, "bitnet_linear_forward"):
            return module.bitnet_linear_forward(
                x_cast,
                packed_cast,
                scales_cast,
                header_cast,
                offsets_cast,
                bias_cast,
            )
    weight = _dequantize_packed_bitnet_weight(
        packed_cast,
        scales_cast,
        header_cast,
        offsets_cast,
        dtype=compute_dtype,
    )
    return runtime_linear(x_cast, weight, bias_cast)


def bitnet_linear_compute_packed(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
    compute_packed_words: torch.Tensor,
    compute_row_scales: torch.Tensor,
    decode_nz_masks: torch.Tensor,
    decode_sign_masks: torch.Tensor,
    decode_row_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    compute_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    target_device = compute_packed_words.device
    x_cast = x.to(device=target_device, dtype=compute_dtype)
    packed_cast, scales_cast, header_cast, offsets_cast, meta = _normalize_bitnet_layout_inputs(
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        target_device=target_device,
    )
    if int(x_cast.shape[-1]) != meta["logical_in_features"]:
        raise ValueError(
            "BitNet input feature mismatch: "
            f"expected {meta['logical_in_features']}, got {int(x_cast.shape[-1])}"
        )
    bias_cast = None if bias is None else bias.to(device=target_device, dtype=compute_dtype)
    if x_cast.is_cuda and not _should_use_eager_autograd_fallback(x_cast, bias_cast):
        module = native_module()
        if (
            has_native_op("bitnet_linear_compute_packed")
            and module is not None
            and hasattr(module, "bitnet_linear_compute_packed_forward")
        ):
            return module.bitnet_linear_compute_packed_forward(
                x_cast,
                packed_cast,
                scales_cast,
                header_cast,
                offsets_cast,
                compute_packed_words.to(device=target_device).contiguous(),
                compute_row_scales.to(device=target_device, dtype=torch.float32).contiguous(),
                decode_nz_masks.to(device=target_device).contiguous(),
                decode_sign_masks.to(device=target_device).contiguous(),
                decode_row_scales.to(device=target_device, dtype=torch.float32).contiguous(),
                bias_cast,
                out_dtype,
            )
    return bitnet_linear(
        x_cast,
        packed_cast,
        scales_cast,
        header_cast,
        offsets_cast,
        bias_cast,
        out_dtype=out_dtype,
    )


def bitnet_linear_from_float(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    spin_enabled: bool = False,
    spin_signs: torch.Tensor | None = None,
    pre_scale: torch.Tensor | None = None,
    act_quant_mode: str = "none",
    act_scale: torch.Tensor | float | None = None,
    act_quant_bits: int = 8,
    act_quant_method: str = "absmax",
    act_quant_percentile: float = 0.999,
) -> torch.Tensor:
    compute_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    target_device = packed_weight.device
    x_cast = x.to(device=target_device, dtype=compute_dtype)
    packed_cast, scales_cast, header_cast, offsets_cast, meta = _normalize_bitnet_layout_inputs(
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        target_device=target_device,
    )
    if int(x_cast.shape[-1]) != meta["logical_in_features"]:
        raise ValueError(
            "BitNet input feature mismatch: "
            f"expected {meta['logical_in_features']}, got {int(x_cast.shape[-1])}"
        )
    bias_cast = None if bias is None else bias.to(device=target_device, dtype=compute_dtype)
    if bias_cast is not None and int(bias_cast.shape[0]) != meta["logical_out_features"]:
        raise ValueError(
            "BitNet bias size mismatch: "
            f"expected {meta['logical_out_features']}, got {int(bias_cast.shape[0])}"
        )
    spin_signs_cast = None
    if spin_signs is not None:
        spin_signs_cast = spin_signs.to(device=target_device, dtype=torch.float32).reshape(-1).contiguous()
    pre_scale_cast = None
    if pre_scale is not None:
        pre_scale_cast = pre_scale.to(device=target_device, dtype=compute_dtype).reshape(-1).contiguous()
    act_scale_cast = _coerce_optional_row_scale(act_scale, device=target_device)
    if x_cast.is_cuda and not _should_use_eager_autograd_fallback(x_cast, bias_cast):
        module = native_module()
        if (
            has_native_op("bitnet_linear_from_float")
            and module is not None
            and hasattr(module, "bitnet_linear_from_float_forward")
        ):
            return module.bitnet_linear_from_float_forward(
                x_cast,
                packed_cast,
                scales_cast,
                header_cast,
                offsets_cast,
                bias_cast,
                bool(spin_enabled),
                spin_signs_cast,
                pre_scale_cast,
                str(act_quant_mode),
                str(act_quant_method),
                int(act_quant_bits),
                float(act_quant_percentile),
                act_scale_cast,
                None,
            )

    x_local = runtime_bitnet_transform_input(
        x_cast,
        spin_enabled=bool(spin_enabled),
        spin_signs=spin_signs_cast,
        pre_scale=pre_scale_cast,
        act_quant_mode=str(act_quant_mode),
        act_scale=act_scale_cast,
        act_quant_bits=int(act_quant_bits),
        act_quant_method=str(act_quant_method),
        act_quant_percentile=float(act_quant_percentile),
    )
    return bitnet_linear(
        x_local,
        packed_cast,
        scales_cast,
        header_cast,
        offsets_cast,
        bias_cast,
    )


__all__ = [
    "per_channel_absmax",
    "nf4_quantize",
    "nf4_dequantize",
    "quantize_activation_int8_rowwise",
    "int8_quantize_activation_transpose",
    "int8_attention",
    "int8_matmul_qkv",
    "int8_linear_from_quantized_activation",
    "int8_linear_grad_weight_from_float",
    "int8_linear",
    "bitnet_linear",
    "bitnet_linear_compute_packed",
    "bitnet_linear_from_float",
    "fp8_linear",
    "nf4_linear",
    "int4_linear",
]
