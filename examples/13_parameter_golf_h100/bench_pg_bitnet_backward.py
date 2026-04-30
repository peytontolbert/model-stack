from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.quantization import _pack_bitnet_quantized
from runtime.native import cuda_kernel_ops, has_native_op, native_module
from runtime.ops import bitnet_runtime_row_quantize
from runtime.quant import bitnet_int8_linear_from_float, bitnet_linear_from_float


@dataclass(frozen=True)
class LinearShape:
    name: str
    in_features: int
    out_features: int


PRESETS: dict[str, list[LinearShape]] = {
    "runtime_row_1024x7_relu2_mlp3": [
        LinearShape("attn_qkv_fused", 1024, 1536),
        LinearShape("attn_out", 1024, 1024),
        LinearShape("mlp_up_relu2", 1024, 3072),
        LinearShape("mlp_down_relu2", 3072, 1024),
        LinearShape("lm_head", 1024, 1024),
    ],
    "runtime_row_1024x7_swiglu_mlp2": [
        LinearShape("attn_qkv_fused", 1024, 1536),
        LinearShape("attn_out", 1024, 1024),
        LinearShape("swiglu_gate_up", 1024, 4096),
        LinearShape("swiglu_down", 2048, 1024),
        LinearShape("lm_head", 1024, 1024),
    ],
}


def _dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _parse_shapes(values: list[str]) -> list[LinearShape]:
    shapes: list[LinearShape] = []
    for idx, value in enumerate(values):
        parts = [part.strip() for part in value.split(":")]
        if len(parts) == 2:
            name = f"custom_{idx}"
            in_features, out_features = parts
        elif len(parts) == 3:
            name, in_features, out_features = parts
        else:
            raise ValueError("--shape must be IN:OUT or NAME:IN:OUT")
        shapes.append(LinearShape(name, int(in_features), int(out_features)))
    return shapes


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _consume(value: torch.Tensor, sink: torch.Tensor | None) -> None:
    if sink is not None:
        sink.add_(value.reshape(-1)[0].float())


def _time_once(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
    consume_output: bool,
) -> float:
    sink = torch.zeros((), device=device, dtype=torch.float32) if consume_output else None
    for _ in range(warmup):
        _consume(fn(), sink)
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _consume(fn(), sink)
        end.record()
        _sync(device)
        return float(start.elapsed_time(end) / max(iters, 1))
    t0 = time.perf_counter()
    for _ in range(iters):
        _consume(fn(), sink)
    return float((time.perf_counter() - t0) * 1000.0 / max(iters, 1))


def _time_median(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    device: torch.device,
    consume_output: bool,
) -> float:
    values = [
        _time_once(fn, warmup=warmup, iters=iters, device=device, consume_output=consume_output)
        for _ in range(max(1, repeats))
    ]
    return float(statistics.median(values))


def _time_median_optional(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    device: torch.device,
    consume_output: bool,
) -> tuple[float | None, str | None]:
    try:
        value = _time_median(
            fn,
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            device=device,
            consume_output=consume_output,
        )
        return value, None
    except Exception as exc:  # pragma: no cover - depends on GPU backend support.
        return None, f"{type(exc).__name__}: {exc}"


def _bench_shape(
    shape: LinearShape,
    *,
    rows: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    repeats: int,
    consume_output: bool,
    include_packed_bitnet: bool,
    include_int8_grad_weight: bool,
    include_int8_grad_weight_upper_bound: bool,
    act_quant_bits: int,
) -> dict[str, object]:
    weight = torch.randn(shape.out_features, shape.in_features, device=device, dtype=dtype).mul_(0.125)
    x = torch.randn(rows, shape.in_features, device=device, dtype=dtype)
    grad_out = torch.randn(rows, shape.out_features, device=device, dtype=dtype)
    qweight, row_scale = bitnet_runtime_row_quantize(weight)
    dequant_weight = qweight.to(dtype=dtype).mul(row_scale.to(dtype=dtype).unsqueeze(-1)).contiguous()

    qweight_t = qweight.t().contiguous()
    ones_scale = torch.ones(shape.in_features, device=device, dtype=torch.float32)
    inv_pre_scale = row_scale.clamp_min(1e-8).reciprocal().contiguous()
    row_scale_cast = row_scale.to(dtype=dtype).view(1, -1)

    def dense_grad_input() -> torch.Tensor:
        return grad_out.matmul(dequant_weight)

    def dense_grad_weight() -> torch.Tensor:
        return grad_out.t().matmul(x)

    def int8_grad_input_explicit_scale() -> torch.Tensor:
        return bitnet_int8_linear_from_float(
            grad_out * row_scale_cast,
            qweight_t,
            ones_scale,
            None,
            pre_scale=None,
            act_quant_mode="dynamic_int8",
            act_scale=None,
            act_quant_bits=int(act_quant_bits),
            act_quant_method="absmax",
            act_quant_percentile=0.999,
        )

    def int8_grad_input_pre_scale() -> torch.Tensor:
        return bitnet_int8_linear_from_float(
            grad_out,
            qweight_t,
            ones_scale,
            None,
            pre_scale=inv_pre_scale,
            act_quant_mode="dynamic_int8",
            act_scale=None,
            act_quant_bits=int(act_quant_bits),
            act_quant_method="absmax",
            act_quant_percentile=0.999,
        )

    timer_kwargs = {
        "warmup": warmup,
        "iters": iters,
        "repeats": repeats,
        "device": device,
        "consume_output": consume_output,
    }
    dense_gi_ms = _time_median(dense_grad_input, **timer_kwargs)
    dense_gw_ms = _time_median(dense_grad_weight, **timer_kwargs)
    int8_explicit_ms = _time_median(int8_grad_input_explicit_scale, **timer_kwargs)
    int8_pre_scale_ms = _time_median(int8_grad_input_pre_scale, **timer_kwargs)

    dense_gi = dense_grad_input()
    int8_explicit = int8_grad_input_explicit_scale()
    int8_pre_scale = int8_grad_input_pre_scale()
    result: dict[str, object] = {
        "shape": shape.name,
        "rows": int(rows),
        "in_features": int(shape.in_features),
        "out_features": int(shape.out_features),
        "dtype": str(dtype),
        "device": str(device),
        "act_quant_bits": int(act_quant_bits),
        "dense_grad_input_ms": dense_gi_ms,
        "dense_grad_weight_ms": dense_gw_ms,
        "int8_grad_input_explicit_scale_ms": int8_explicit_ms,
        "int8_grad_input_explicit_scale_speedup_vs_dense": dense_gi_ms / int8_explicit_ms,
        "int8_grad_input_explicit_scale_max_abs_err": float((int8_explicit - dense_gi).abs().max().item()),
        "int8_grad_input_pre_scale_ms": int8_pre_scale_ms,
        "int8_grad_input_pre_scale_speedup_vs_dense": dense_gi_ms / int8_pre_scale_ms,
        "int8_grad_input_pre_scale_max_abs_err": float((int8_pre_scale - dense_gi).abs().max().item()),
    }
    if include_packed_bitnet:
        packed, scales, header, offsets = _pack_bitnet_quantized(
            qweight_t,
            scale_values=ones_scale,
            scale_granularity=2,
            scale_group_size=1,
        )
        grad_scaled = grad_out.mul(row_scale_cast).contiguous()

        def packed_bitnet_grad_input() -> torch.Tensor:
            return bitnet_linear_from_float(grad_scaled, packed, scales, header, offsets)

        packed_ms = _time_median(packed_bitnet_grad_input, **timer_kwargs)
        packed_out = packed_bitnet_grad_input()
        result["packed_bitnet_grad_input_ms"] = packed_ms
        result["packed_bitnet_grad_input_speedup_vs_dense"] = dense_gi_ms / packed_ms
        result["packed_bitnet_grad_input_max_abs_err"] = float((packed_out - dense_gi).abs().max().item())
    if include_int8_grad_weight:
        module = native_module()
        if module is None or not hasattr(module, "int8_quantize_activation_forward") or not hasattr(module, "int8_linear_forward"):
            raise RuntimeError("int8 grad_weight benchmark requires native int8 quantize and linear ops")

        def quantize_x_transposed() -> tuple[torch.Tensor, torch.Tensor]:
            return module.int8_quantize_activation_forward(x.t().contiguous(), None)

        def quantize_grad_out_transposed() -> tuple[torch.Tensor, torch.Tensor]:
            return module.int8_quantize_activation_forward(grad_out.t().contiguous(), None)

        qx_t, x_t_scale = quantize_x_transposed()
        qgo_t, go_t_scale = quantize_grad_out_transposed()

        def transpose_x() -> torch.Tensor:
            return x.t().contiguous()

        def transpose_grad_out() -> torch.Tensor:
            return grad_out.t().contiguous()

        def quantize_x_transposed_tensor() -> torch.Tensor:
            return quantize_x_transposed()[0]

        def quantize_grad_out_transposed_tensor() -> torch.Tensor:
            return quantize_grad_out_transposed()[0]

        def int8_grad_weight_prequant() -> torch.Tensor:
            return module.int8_linear_forward(qgo_t, go_t_scale, qx_t, x_t_scale, None, dtype)

        def int8_grad_weight_full() -> torch.Tensor:
            qx_t_local, x_t_scale_local = quantize_x_transposed()
            qgo_t_local, go_t_scale_local = quantize_grad_out_transposed()
            return module.int8_linear_forward(qgo_t_local, go_t_scale_local, qx_t_local, x_t_scale_local, None, dtype)

        dense_gw = dense_grad_weight()
        int8_gw = int8_grad_weight_prequant()
        transpose_x_ms = _time_median(transpose_x, **timer_kwargs)
        transpose_go_ms = _time_median(transpose_grad_out, **timer_kwargs)
        quant_x_t_ms = _time_median(quantize_x_transposed_tensor, **timer_kwargs)
        quant_go_t_ms = _time_median(quantize_grad_out_transposed_tensor, **timer_kwargs)
        int8_gw_prequant_ms = _time_median(int8_grad_weight_prequant, **timer_kwargs)
        int8_gw_full_ms = _time_median(int8_grad_weight_full, **timer_kwargs)
        result["transpose_x_ms"] = transpose_x_ms
        result["transpose_grad_out_ms"] = transpose_go_ms
        result["int8_quantize_x_transposed_ms"] = quant_x_t_ms
        result["int8_quantize_grad_out_transposed_ms"] = quant_go_t_ms
        result["int8_grad_weight_prequant_ms"] = int8_gw_prequant_ms
        result["int8_grad_weight_prequant_speedup_vs_dense"] = dense_gw_ms / int8_gw_prequant_ms
        result["int8_grad_weight_full_ms"] = int8_gw_full_ms
        result["int8_grad_weight_full_speedup_vs_dense"] = dense_gw_ms / int8_gw_full_ms
        result["int8_grad_weight_prequant_max_abs_err"] = float((int8_gw - dense_gw).abs().max().item())
        if hasattr(module, "int8_linear_grad_weight_from_float_forward"):

            def int8_grad_weight_native_composed() -> torch.Tensor:
                return module.int8_linear_grad_weight_from_float_forward(grad_out, x, dtype)

            native_composed = int8_grad_weight_native_composed()
            native_composed_ms = _time_median(int8_grad_weight_native_composed, **timer_kwargs)
            result["int8_grad_weight_native_composed_ms"] = native_composed_ms
            result["int8_grad_weight_native_composed_speedup_vs_dense"] = dense_gw_ms / native_composed_ms
            result["int8_grad_weight_native_composed_max_abs_err"] = float(
                (native_composed - dense_gw).abs().max().item()
            )
        if hasattr(module, "int8_quantize_activation_transpose_forward"):

            def quantize_x_transpose_fused() -> tuple[torch.Tensor, torch.Tensor]:
                return module.int8_quantize_activation_transpose_forward(x)

            def quantize_grad_out_transpose_fused() -> tuple[torch.Tensor, torch.Tensor]:
                return module.int8_quantize_activation_transpose_forward(grad_out)

            qx_t_fused, x_t_scale_fused = quantize_x_transpose_fused()
            qgo_t_fused, go_t_scale_fused = quantize_grad_out_transpose_fused()

            def quantize_x_transpose_fused_provided_scale() -> tuple[torch.Tensor, torch.Tensor]:
                return module.int8_quantize_activation_transpose_forward(x, x_t_scale_fused)

            def quantize_grad_out_transpose_fused_provided_scale() -> tuple[torch.Tensor, torch.Tensor]:
                return module.int8_quantize_activation_transpose_forward(grad_out, go_t_scale_fused)

            def quantize_x_transpose_fused_tensor() -> torch.Tensor:
                return quantize_x_transpose_fused()[0]

            def quantize_grad_out_transpose_fused_tensor() -> torch.Tensor:
                return quantize_grad_out_transpose_fused()[0]

            def quantize_x_transpose_fused_provided_scale_tensor() -> torch.Tensor:
                return quantize_x_transpose_fused_provided_scale()[0]

            def quantize_grad_out_transpose_fused_provided_scale_tensor() -> torch.Tensor:
                return quantize_grad_out_transpose_fused_provided_scale()[0]

            def int8_grad_weight_fused_prequant() -> torch.Tensor:
                return module.int8_linear_forward(
                    qgo_t_fused,
                    go_t_scale_fused,
                    qx_t_fused,
                    x_t_scale_fused,
                    None,
                    dtype,
                )

            def int8_grad_weight_fused_full() -> torch.Tensor:
                qx_t_local, x_t_scale_local = quantize_x_transpose_fused()
                qgo_t_local, go_t_scale_local = quantize_grad_out_transpose_fused()
                return module.int8_linear_forward(
                    qgo_t_local,
                    go_t_scale_local,
                    qx_t_local,
                    x_t_scale_local,
                    None,
                    dtype,
                )

            def int8_grad_weight_fused_delayed_scale_full() -> torch.Tensor:
                qx_t_local, x_t_scale_local = quantize_x_transpose_fused_provided_scale()
                qgo_t_local, go_t_scale_local = quantize_grad_out_transpose_fused_provided_scale()
                return module.int8_linear_forward(
                    qgo_t_local,
                    go_t_scale_local,
                    qx_t_local,
                    x_t_scale_local,
                    None,
                    dtype,
                )

            int8_gw_fused = int8_grad_weight_fused_prequant()
            fused_quant_x_ms = _time_median(quantize_x_transpose_fused_tensor, **timer_kwargs)
            fused_quant_go_ms = _time_median(quantize_grad_out_transpose_fused_tensor, **timer_kwargs)
            fused_quant_x_provided_ms = _time_median(
                quantize_x_transpose_fused_provided_scale_tensor,
                **timer_kwargs,
            )
            fused_quant_go_provided_ms = _time_median(
                quantize_grad_out_transpose_fused_provided_scale_tensor,
                **timer_kwargs,
            )
            fused_prequant_ms = _time_median(int8_grad_weight_fused_prequant, **timer_kwargs)
            fused_full_ms = _time_median(int8_grad_weight_fused_full, **timer_kwargs)
            fused_delayed_scale_full_ms = _time_median(int8_grad_weight_fused_delayed_scale_full, **timer_kwargs)
            result["int8_quantize_x_transpose_fused_ms"] = fused_quant_x_ms
            result["int8_quantize_grad_out_transpose_fused_ms"] = fused_quant_go_ms
            result["int8_quantize_x_transpose_fused_provided_scale_ms"] = fused_quant_x_provided_ms
            result["int8_quantize_grad_out_transpose_fused_provided_scale_ms"] = fused_quant_go_provided_ms
            result["int8_grad_weight_fused_prequant_ms"] = fused_prequant_ms
            result["int8_grad_weight_fused_prequant_speedup_vs_dense"] = dense_gw_ms / fused_prequant_ms
            result["int8_grad_weight_fused_full_ms"] = fused_full_ms
            result["int8_grad_weight_fused_full_speedup_vs_dense"] = dense_gw_ms / fused_full_ms
            result["int8_grad_weight_fused_delayed_scale_full_ms"] = fused_delayed_scale_full_ms
            result["int8_grad_weight_fused_delayed_scale_full_speedup_vs_dense"] = (
                dense_gw_ms / fused_delayed_scale_full_ms
            )
            result["int8_grad_weight_fused_prequant_max_abs_err"] = float((int8_gw_fused - dense_gw).abs().max().item())
        if hasattr(module, "int8_quantize_activation_columnwise_forward"):

            def quantize_x_columnwise() -> tuple[torch.Tensor, torch.Tensor]:
                return module.int8_quantize_activation_columnwise_forward(x, None)

            def quantize_grad_out_columnwise() -> tuple[torch.Tensor, torch.Tensor]:
                return module.int8_quantize_activation_columnwise_forward(grad_out, None)

            qx_col, x_col_scale = quantize_x_columnwise()
            qgo_col, go_col_scale = quantize_grad_out_columnwise()

            def quantize_x_columnwise_provided_scale() -> tuple[torch.Tensor, torch.Tensor]:
                return module.int8_quantize_activation_columnwise_forward(x, x_col_scale)

            def quantize_grad_out_columnwise_provided_scale() -> tuple[torch.Tensor, torch.Tensor]:
                return module.int8_quantize_activation_columnwise_forward(grad_out, go_col_scale)

            def quantize_x_columnwise_tensor() -> torch.Tensor:
                return quantize_x_columnwise()[0]

            def quantize_grad_out_columnwise_tensor() -> torch.Tensor:
                return quantize_grad_out_columnwise()[0]

            def quantize_x_columnwise_provided_scale_tensor() -> torch.Tensor:
                return quantize_x_columnwise_provided_scale()[0]

            def quantize_grad_out_columnwise_provided_scale_tensor() -> torch.Tensor:
                return quantize_grad_out_columnwise_provided_scale()[0]

            columnwise_quant_x_ms = _time_median(quantize_x_columnwise_tensor, **timer_kwargs)
            columnwise_quant_go_ms = _time_median(quantize_grad_out_columnwise_tensor, **timer_kwargs)
            columnwise_quant_x_provided_ms = _time_median(
                quantize_x_columnwise_provided_scale_tensor,
                **timer_kwargs,
            )
            columnwise_quant_go_provided_ms = _time_median(
                quantize_grad_out_columnwise_provided_scale_tensor,
                **timer_kwargs,
            )
            result["int8_quantize_x_columnwise_ms"] = columnwise_quant_x_ms
            result["int8_quantize_grad_out_columnwise_ms"] = columnwise_quant_go_ms
            result["int8_quantize_x_columnwise_provided_scale_ms"] = columnwise_quant_x_provided_ms
            result["int8_quantize_grad_out_columnwise_provided_scale_ms"] = columnwise_quant_go_provided_ms
            result["int8_quantize_x_columnwise_shape"] = list(qx_col.shape)
            result["int8_quantize_grad_out_columnwise_shape"] = list(qgo_col.shape)
    if include_int8_grad_weight_upper_bound:
        module = native_module()
        if module is None or not hasattr(module, "int8_quantize_activation_forward") or not hasattr(torch, "_int_mm"):
            raise RuntimeError("int8 grad_weight upper bound requires native int8 quantize and torch._int_mm")

        def quantize_x_original() -> tuple[torch.Tensor, torch.Tensor]:
            return module.int8_quantize_activation_forward(x, None)

        def quantize_grad_out_original() -> tuple[torch.Tensor, torch.Tensor]:
            return module.int8_quantize_activation_forward(grad_out, None)

        qx_original, _ = quantize_x_original()
        qgo_original, _ = quantize_grad_out_original()

        def quantize_x_original_tensor() -> torch.Tensor:
            return quantize_x_original()[0]

        def quantize_grad_out_original_tensor() -> torch.Tensor:
            return quantize_grad_out_original()[0]

        def raw_original_rowwise_int_mm_view() -> torch.Tensor:
            return torch._int_mm(qgo_original.t(), qx_original)

        def raw_original_rowwise_int_mm_contiguous_lhs() -> torch.Tensor:
            return torch._int_mm(qgo_original.t().contiguous(), qx_original)

        qx_orig_ms = _time_median(quantize_x_original_tensor, **timer_kwargs)
        qgo_orig_ms = _time_median(quantize_grad_out_original_tensor, **timer_kwargs)
        raw_view_ms, raw_view_err = _time_median_optional(raw_original_rowwise_int_mm_view, **timer_kwargs)
        raw_contig_ms, raw_contig_err = _time_median_optional(
            raw_original_rowwise_int_mm_contiguous_lhs,
            **timer_kwargs,
        )
        result["int8_quantize_x_original_rowwise_ms"] = qx_orig_ms
        result["int8_quantize_grad_out_original_rowwise_ms"] = qgo_orig_ms
        if raw_view_ms is None:
            result["raw_original_rowwise_int_mm_view_error"] = raw_view_err
        else:
            result["raw_original_rowwise_int_mm_view_ms"] = raw_view_ms
            result["raw_original_rowwise_int_mm_view_speedup_vs_dense_grad_weight"] = dense_gw_ms / raw_view_ms
        if raw_contig_ms is None:
            result["raw_original_rowwise_int_mm_contiguous_lhs_error"] = raw_contig_err
        else:
            result["raw_original_rowwise_int_mm_contiguous_lhs_ms"] = raw_contig_ms
            result["raw_original_rowwise_int_mm_contiguous_lhs_speedup_vs_dense_grad_weight"] = (
                dense_gw_ms / raw_contig_ms
            )
        result["raw_original_rowwise_int_mm_scale_correct"] = False
        result["raw_original_rowwise_int_mm_scale_issue"] = (
            "original row-wise activation scales multiply inside the reduction; a single output row/column "
            "scale cannot recover dense grad_weight"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BitNet trainable backward component candidates.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="runtime_row_1024x7_relu2_mlp3")
    parser.add_argument("--shape", action="append", default=[], help="Custom shape NAME:IN:OUT or IN:OUT")
    parser.add_argument("--no-preset-shapes", action="store_true", help="Benchmark only --shape entries.")
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--act-quant-bits", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--consume-output", action="store_true")
    parser.add_argument("--include-packed-bitnet", action="store_true")
    parser.add_argument("--include-int8-grad-weight", action="store_true")
    parser.add_argument("--include-int8-grad-weight-upper-bound", action="store_true")
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    shapes = ([] if args.no_preset_shapes else list(PRESETS[args.preset])) + _parse_shapes(args.shape)
    if not shapes:
        raise ValueError("no shapes selected; use a preset or pass --shape")
    header = {
        "header": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "dtype": str(dtype),
            "preset": args.preset,
            "has_native_runtime_row_quantize": has_native_op("bitnet_runtime_row_quantize"),
            "has_cuda_runtime_row_quantize": "bitnet_runtime_row_quantize" in set(cuda_kernel_ops()),
        }
    }
    print(json.dumps(header, sort_keys=True) if args.jsonl else header)
    for shape in shapes:
        result = _bench_shape(
            shape,
            rows=args.rows,
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            consume_output=bool(args.consume_output),
            include_packed_bitnet=bool(args.include_packed_bitnet),
            include_int8_grad_weight=bool(args.include_int8_grad_weight),
            include_int8_grad_weight_upper_bound=bool(args.include_int8_grad_weight_upper_bound),
            act_quant_bits=int(args.act_quant_bits),
        )
        print(json.dumps(result, sort_keys=True) if args.jsonl else json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
