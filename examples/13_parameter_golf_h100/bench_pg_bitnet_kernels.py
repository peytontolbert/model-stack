from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.quantization import QuantizedLinearBitNet, TrainableBitNetLinear
from runtime.native import has_native_op, native_module
from runtime.quant import bitnet_int8_linear_from_float as runtime_bitnet_int8_linear_from_float
from runtime.quant import int8_linear as runtime_int8_linear
from runtime.quant import int8_linear_from_quantized_activation as runtime_int8_linear_from_quantized_activation


@dataclass(frozen=True)
class LinearShape:
    name: str
    in_features: int
    out_features: int


PRESETS: dict[str, list[LinearShape]] = {
    "runtime_row_1024x7_relu2_mlp2": [
        LinearShape("attn_q", 1024, 1024),
        LinearShape("attn_k", 1024, 256),
        LinearShape("attn_v", 1024, 256),
        LinearShape("attn_out", 1024, 1024),
        LinearShape("mlp_up", 1024, 2048),
        LinearShape("mlp_down", 2048, 1024),
    ],
    "runtime_row_1024x7_relu2_mlp3": [
        LinearShape("attn_qkv_fused", 1024, 1536),
        LinearShape("attn_out", 1024, 1024),
        LinearShape("mlp_up", 1024, 3072),
        LinearShape("mlp_down", 3072, 1024),
        LinearShape("lm_head", 1024, 1024),
    ],
    "runtime_row_1024": [
        LinearShape("attn_qkv_fused", 1024, 1536),
        LinearShape("attn_out", 1024, 1024),
        LinearShape("mlp_up", 1024, 2048),
        LinearShape("mlp_down", 2048, 1024),
        LinearShape("lm_head", 1024, 1024),
    ],
    "runtime_row_512": [
        LinearShape("attn_qkv_fused", 512, 1024),
        LinearShape("attn_out", 512, 512),
        LinearShape("mlp_up", 512, 1024),
        LinearShape("mlp_down", 1024, 512),
        LinearShape("lm_head", 512, 1024),
    ],
}


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[key]


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


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


def _normalize_activation_quant_mode_and_bits(mode: str, bits: int) -> tuple[str, int]:
    mode_name = str(mode).strip().lower()
    if mode_name in {"dynamic_int4", "dynamic_a4"}:
        return "dynamic_int8", 4
    if mode_name in {"static_int4", "static_a4"}:
        return "static_int8", 4
    return mode_name, int(bits)


def _consume(value: torch.Tensor, sink: torch.Tensor | None) -> None:
    if sink is None:
        return
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _consume(fn(), sink)
        end.record()
        torch.cuda.synchronize(device)
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


def _sync_wall_once(
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _consume(fn(), sink)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return float((time.perf_counter() - t0) * 1000.0 / max(iters, 1))


def _expected_plan(rows: int, in_features: int) -> str:
    if rows <= 8 and os.environ.get("MODEL_STACK_DISABLE_BITNET_PERSISTENT_DECODE", "0") in {"", "0"}:
        return "decode_persistent"
    if (
        rows >= 8
        and in_features >= 4096
        and os.environ.get("MODEL_STACK_DISABLE_BITNET_SPLITK", "0") in {"", "0"}
    ):
        return "prefill_splitk"
    return "prefill_tiled"


def _build_layer(
    shape: LinearShape,
    *,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
) -> tuple[torch.nn.Linear, QuantizedLinearBitNet, TrainableBitNetLinear]:
    linear = torch.nn.Linear(
        shape.in_features,
        shape.out_features,
        bias=bias,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.125)
        if linear.bias is not None:
            linear.bias.normal_(mean=0.0, std=0.01)
    bitnet = QuantizedLinearBitNet(shape.in_features, shape.out_features, bias=bias).from_float(
        linear,
        activation_quant=activation_quant,
        activation_quant_bits=activation_quant_bits,
        activation_quant_method=activation_quant_method,
        activation_quant_percentile=activation_quant_percentile,
    )
    bitnet = bitnet.to(device=device)
    qat = TrainableBitNetLinear(shape.in_features, shape.out_features, bias=bias).to(device=device, dtype=dtype)
    qat.from_float(linear)
    return linear, bitnet, qat


def _bench_one(
    shape: LinearShape,
    rows: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
    warmup: int,
    iters: int,
    repeats: int,
    sync_wall: bool,
    consume_output: bool,
    include_qat_forward: bool,
    include_int8_baselines: bool,
) -> dict[str, object]:
    requested_activation_quant = str(activation_quant)
    activation_quant, activation_quant_bits = _normalize_activation_quant_mode_and_bits(
        activation_quant,
        activation_quant_bits,
    )
    x = torch.randn(rows, shape.in_features, device=device, dtype=dtype)
    linear, bitnet, qat = _build_layer(
        shape,
        dtype=dtype,
        device=device,
        bias=bias,
        activation_quant=activation_quant,
        activation_quant_bits=activation_quant_bits,
        activation_quant_method=activation_quant_method,
        activation_quant_percentile=activation_quant_percentile,
    )

    with torch.inference_mode():
        mode_name = str(activation_quant).strip().lower()
        if mode_name in {"", "none", "off"}:
            bitnet._compute_backend_weight(device=device)
            bitnet._decode_backend_weight(device=device)
        elif mode_name in {"dynamic_int8", "static_int8"}:
            bitnet._int8_backend_weight(device=device)
        dequant_weight = bitnet.runtime_weight(dtype=dtype, device=device).contiguous()
        dequant_bias = bitnet.runtime_bias(dtype=dtype, device=device)
        dense_ref_out = F.linear(x, dequant_weight, dequant_bias)
        original_dense_out = F.linear(x, linear.weight, linear.bias)
        module = native_module()
        auto_out = bitnet.runtime_linear(x)
        forced_out = bitnet.runtime_linear(x, backend="bitnet")
        native_module_out = (
            module.linear_module_forward(x, bitnet, "auto")
            if module is not None and hasattr(module, "linear_module_forward")
            else None
        )

        def dense_ref() -> torch.Tensor:
            return F.linear(x, dequant_weight, dequant_bias)

        def dense_original() -> torch.Tensor:
            return F.linear(x, linear.weight, linear.bias)

        def model_stack_auto() -> torch.Tensor:
            return bitnet.runtime_linear(x)

        def forced_backend() -> torch.Tensor:
            return bitnet.runtime_linear(x, backend="bitnet")

        def native_module_auto() -> torch.Tensor:
            assert module is not None
            return module.linear_module_forward(x, bitnet, "auto")

        int8_baseline_items: dict[str, tuple[Callable[[], torch.Tensor], torch.Tensor]] = {}
        int8_kernel_metric_items: dict[str, Callable[[], torch.Tensor]] = {}
        if include_int8_baselines and mode_name in {"dynamic_int8", "static_int8"}:
            qweight, inv_scale = bitnet._int8_backend_weight(device=device)
            int8_bias = bitnet.runtime_bias(dtype=dtype, device=device)
            act_scale = bitnet.act_scale.to(device=device, dtype=torch.float32) if mode_name == "static_int8" else None

            def native_int8_from_float() -> torch.Tensor:
                return runtime_int8_linear(
                    x,
                    qweight,
                    inv_scale,
                    int8_bias,
                    act_scale=act_scale,
                    act_method=str(activation_quant_method),
                    act_percentile=float(activation_quant_percentile),
                )

            def direct_bitnet_int8_from_float() -> torch.Tensor:
                return runtime_bitnet_int8_linear_from_float(
                    x,
                    qweight,
                    inv_scale,
                    int8_bias,
                    pre_scale=None,
                    act_quant_mode=str(activation_quant),
                    act_scale=act_scale,
                    act_quant_bits=int(activation_quant_bits),
                    act_quant_method=str(activation_quant_method),
                    act_quant_percentile=float(activation_quant_percentile),
                )

            # Prequantized matmul isolates the int8 backend from activation quantization
            # cost. The one-time quantization here is deliberately outside the timed path.
            qx_once, row_scale_once, out_dtype = bitnet.runtime_quantize_int8_input(x)

            if (
                int(activation_quant_bits) == 8
                and module is not None
                and hasattr(module, "int8_quantize_activation_forward")
            ):
                def activation_int8_quantize() -> torch.Tensor:
                    qx, _row_scale = module.int8_quantize_activation_forward(x, act_scale)
                    return qx
            else:
                def activation_int8_quantize() -> torch.Tensor:
                    qx, _row_scale, _out_dtype = bitnet.runtime_quantize_int8_input(x)
                    return qx

            def prequant_int8_matmul() -> torch.Tensor:
                return runtime_int8_linear_from_quantized_activation(
                    qx_once,
                    row_scale_once,
                    qweight,
                    inv_scale,
                    int8_bias,
                    out_dtype=out_dtype,
                )

            if module is not None and hasattr(module, "int8_linear_accum_forward"):
                def prequant_int8_accum_only() -> torch.Tensor:
                    return module.int8_linear_accum_forward(qx_once, qweight)

                int8_kernel_metric_items["prequant_int8_accum_only"] = prequant_int8_accum_only

            int8_kernel_metric_items["activation_int8_quantize"] = activation_int8_quantize

            int8_baseline_items = {
                "direct_bitnet_int8_from_float": (direct_bitnet_int8_from_float, direct_bitnet_int8_from_float()),
                "prequant_int8_matmul": (prequant_int8_matmul, prequant_int8_matmul()),
            }
            if int(activation_quant_bits) == 8:
                int8_baseline_items["native_int8_from_float"] = (native_int8_from_float, native_int8_from_float())

        timer_kwargs = {
            "warmup": warmup,
            "iters": iters,
            "repeats": repeats,
            "device": device,
            "consume_output": consume_output,
        }
        dense_ref_ms = _time_median(dense_ref, **timer_kwargs)
        dense_original_ms = _time_median(dense_original, **timer_kwargs)
        auto_ms = _time_median(model_stack_auto, **timer_kwargs)
        forced_ms = _time_median(forced_backend, **timer_kwargs)
        native_module_ms = (
            _time_median(native_module_auto, **timer_kwargs)
            if module is not None and hasattr(module, "linear_module_forward")
            else None
        )
        int8_baseline_ms: dict[str, float] = {
            name: _time_median(fn, **timer_kwargs)
            for name, (fn, _out) in int8_baseline_items.items()
        }
        int8_kernel_metric_ms: dict[str, float] = {
            name: _time_median(fn, **timer_kwargs)
            for name, fn in int8_kernel_metric_items.items()
        }
        sync_kwargs = {
            "warmup": warmup,
            "iters": iters,
            "device": device,
            "consume_output": consume_output,
        }
        dense_ref_sync_ms = _sync_wall_once(dense_ref, **sync_kwargs) if sync_wall else None
        forced_sync_ms = _sync_wall_once(forced_backend, **sync_kwargs) if sync_wall else None

    result: dict[str, object] = {
        "shape": shape.name,
        "rows": int(rows),
        "in_features": int(shape.in_features),
        "out_features": int(shape.out_features),
        "expected_plan": _expected_plan(rows, shape.in_features),
        "activation_quant": requested_activation_quant,
        "canonical_activation_quant": str(activation_quant),
        "activation_quant_bits": int(activation_quant_bits),
        "dense_original_ms": dense_original_ms,
        "dense_bitnet_dequant_ms": dense_ref_ms,
        "model_stack_auto_ms": auto_ms,
        "model_stack_forced_backend_ms": forced_ms,
        "forced_speedup_vs_dequant": dense_ref_ms / forced_ms if forced_ms > 0 else float("inf"),
        "auto_speedup_vs_dequant": dense_ref_ms / auto_ms if auto_ms > 0 else float("inf"),
        "auto_max_abs_err_vs_dequant": float((auto_out - dense_ref_out).abs().max().item()),
        "forced_max_abs_err_vs_dequant": float((forced_out - dense_ref_out).abs().max().item()),
        "dequant_max_abs_err_vs_original_dense": float((dense_ref_out - original_dense_out).abs().max().item()),
    }
    if native_module_ms is not None and native_module_out is not None:
        result["native_module_auto_ms"] = native_module_ms
        result["native_module_auto_speedup_vs_dequant"] = (
            dense_ref_ms / native_module_ms if native_module_ms > 0 else float("inf")
        )
        result["native_module_auto_max_abs_err_vs_dequant"] = float(
            (native_module_out - dense_ref_out).abs().max().item()
        )
    for name, (_fn, out) in int8_baseline_items.items():
        ms = int8_baseline_ms[name]
        result[f"{name}_ms"] = ms
        result[f"{name}_speedup_vs_dequant"] = dense_ref_ms / ms if ms > 0 else float("inf")
        result[f"forced_speedup_vs_{name}"] = ms / forced_ms if forced_ms > 0 else float("inf")
        result[f"{name}_max_abs_err_vs_dequant"] = float((out - dense_ref_out).abs().max().item())
    for name, ms in int8_kernel_metric_ms.items():
        result[f"{name}_ms"] = ms
        result[f"{name}_speedup_vs_dequant"] = dense_ref_ms / ms if ms > 0 else float("inf")
    if "direct_bitnet_int8_from_float" in int8_baseline_ms and "native_int8_from_float" in int8_baseline_ms:
        direct_ms = int8_baseline_ms["direct_bitnet_int8_from_float"]
        native_ms = int8_baseline_ms["native_int8_from_float"]
        result["direct_bitnet_speedup_vs_native_int8_from_float"] = native_ms / direct_ms if direct_ms > 0 else float("inf")
    if dense_ref_sync_ms is not None and forced_sync_ms is not None:
        result["dense_bitnet_dequant_sync_wall_ms"] = dense_ref_sync_ms
        result["model_stack_forced_backend_sync_wall_ms"] = forced_sync_ms
        result["forced_sync_wall_speedup_vs_dequant"] = (
            dense_ref_sync_ms / forced_sync_ms if forced_sync_ms > 0 else float("inf")
        )
    if include_qat_forward:
        qat_x = x.detach()

        def qat_forward() -> torch.Tensor:
            return qat(qat_x)

        result["trainable_qat_forward_ms"] = _time_median(
            qat_forward,
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            device=device,
            consume_output=consume_output,
        )
    return result


def _set_env_variant(name: str, value: str | None) -> str | None:
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    return previous


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _env_variants(args: argparse.Namespace) -> list[tuple[str, dict[str, str]]]:
    int8_cutlass_fused_variant = (
        "int8_cutlass_fused",
        {
            "MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED": "1",
        },
    )
    int8_backend_variants = [
        (
            "int8_no_cublaslt",
            {
                "MODEL_STACK_DISABLE_INT8_LINEAR_CUBLASLT": "1",
                "MODEL_STACK_ENABLE_INT8_LINEAR_WGMMA": "0",
                "MODEL_STACK_DISABLE_INT8_LINEAR_WGMMA": "1",
            },
        ),
        (
            "int8_sm90a_wgmma",
            {
                "MODEL_STACK_DISABLE_INT8_LINEAR_CUBLASLT": "1",
                "MODEL_STACK_ENABLE_INT8_LINEAR_WGMMA": "1",
                "MODEL_STACK_DISABLE_INT8_LINEAR_WGMMA": "0",
            },
        ),
    ]
    if not args.tune_env:
        variants = [("current", {})]
        if args.include_int8_cutlass_fused_variant or args.include_int8_backend_variants:
            variants.append(int8_cutlass_fused_variant)
        if args.include_int8_backend_variants:
            variants.extend(int8_backend_variants)
        return variants
    variants: list[tuple[str, dict[str, str]]] = []
    decode_values = _parse_ints(args.decode_cta_multipliers)
    for value in decode_values:
        variants.append(
            (
                f"decode_cta_{value}",
                {
                    "MODEL_STACK_BITNET_DECODE_CTA_MULTIPLIER": str(value),
                    "MODEL_STACK_DISABLE_BITNET_PERSISTENT_DECODE": "0",
                },
            )
        )
    for disabled in _parse_ints(args.splitk_disabled_values):
        for max_slices in _parse_ints(args.splitk_max_slices_values):
            variants.append(
                (
                    f"splitk_disabled_{disabled}_max_{max_slices}",
                    {
                        "MODEL_STACK_DISABLE_BITNET_SPLITK": str(disabled),
                        "MODEL_STACK_BITNET_SPLITK_MAX_SLICES": str(max_slices),
                    },
                )
            )
    if args.include_int8_cutlass_fused_variant or args.include_int8_backend_variants:
        variants.append(int8_cutlass_fused_variant)
    if args.include_int8_backend_variants:
        variants.extend(int8_backend_variants)
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Model Stack BitNet kernels on Parameter Golf shapes.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="runtime_row_1024x7_relu2_mlp3")
    parser.add_argument("--shape", action="append", default=[], help="Extra/custom shape as IN:OUT or NAME:IN:OUT.")
    parser.add_argument("--no-preset-shapes", action="store_true", help="Benchmark only --shape entries.")
    parser.add_argument("--rows", default="1,16,4096,65536")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--activation-quant", default="none")
    parser.add_argument("--activation-quant-bits", type=int, default=8)
    parser.add_argument("--activation-quant-method", default="absmax")
    parser.add_argument("--activation-quant-percentile", type=float, default=0.999)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sync-wall", action="store_true")
    parser.add_argument("--consume-output", action="store_true")
    parser.add_argument("--include-qat-forward", action="store_true")
    parser.add_argument(
        "--no-int8-baselines",
        action="store_true",
        help="Skip native int8-from-float and prequantized int8 matmul comparisons.",
    )
    parser.add_argument("--tune-env", action="store_true")
    parser.add_argument(
        "--include-int8-cutlass-fused-variant",
        action="store_true",
        help="Also test the opt-in CUTLASS fused int8 backend variant.",
    )
    parser.add_argument(
        "--include-int8-backend-variants",
        action="store_true",
        help="Also test forced int8 backend variants: CUTLASS fused, no-cublasLt, and opt-in SM90a WGMMA.",
    )
    parser.add_argument("--decode-cta-multipliers", default="1,2,4,8")
    parser.add_argument("--splitk-disabled-values", default="0,1")
    parser.add_argument("--splitk-max-slices-values", default="2,4,8")
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dtype = _dtype(args.dtype)
    shapes = ([] if args.no_preset_shapes else list(PRESETS[args.preset])) + _parse_shapes(args.shape)
    if not shapes:
        raise ValueError("no shapes selected; use a preset or pass --shape")
    rows_values = _parse_ints(args.rows)
    module = native_module()
    header = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "preset": args.preset,
        "native_module": getattr(module, "__file__", None),
        "has_bitnet_linear": bool(has_native_op("bitnet_linear")),
        "has_bitnet_linear_compute_packed": bool(has_native_op("bitnet_linear_compute_packed")),
        "has_bitnet_linear_from_float": bool(has_native_op("bitnet_linear_from_float")),
        "has_bitnet_int8_linear_from_float": bool(has_native_op("bitnet_int8_linear_from_float")),
        "has_int8_linear": bool(has_native_op("int8_linear")),
        "has_int8_quantize_activation": bool(has_native_op("int8_quantize_activation")),
        "has_int8_quantize_activation_forward": bool(
            module is not None and hasattr(module, "int8_quantize_activation_forward")
        ),
        "has_linear_module_forward": bool(module is not None and hasattr(module, "linear_module_forward")),
        "has_int8_linear_from_float": bool(has_native_op("int8_linear_from_float")),
        "has_int8_linear_accum": bool(module is not None and hasattr(module, "int8_linear_accum_forward")),
    }
    print(json.dumps({"header": header}, sort_keys=True))

    for variant_name, env in _env_variants(args):
        previous = {key: _set_env_variant(key, value) for key, value in env.items()}
        try:
            for shape in shapes:
                for rows in rows_values:
                    result = _bench_one(
                        shape,
                        rows,
                        dtype=dtype,
                        device=device,
                        bias=bool(args.bias),
                        activation_quant=str(args.activation_quant),
                        activation_quant_bits=int(args.activation_quant_bits),
                        activation_quant_method=str(args.activation_quant_method),
                        activation_quant_percentile=float(args.activation_quant_percentile),
                        warmup=int(args.warmup),
                        iters=int(args.iters),
                        repeats=int(args.repeats),
                        sync_wall=bool(args.sync_wall),
                        consume_output=bool(args.consume_output),
                        include_qat_forward=bool(args.include_qat_forward),
                        include_int8_baselines=not bool(args.no_int8_baselines),
                    )
                    result["env_variant"] = variant_name
                    result["env"] = dict(env)
                    if args.jsonl:
                        print(json.dumps(result, sort_keys=True))
                    else:
                        line = (
                            "variant={env_variant} shape={shape} rows={rows} "
                            "plan={expected_plan} dense_dequant_ms={dense_bitnet_dequant_ms:.4f} "
                            "auto_ms={model_stack_auto_ms:.4f} forced_ms={model_stack_forced_backend_ms:.4f} "
                            "forced_speedup={forced_speedup_vs_dequant:.3f}x "
                            "auto_speedup={auto_speedup_vs_dequant:.3f}x "
                            "forced_err={forced_max_abs_err_vs_dequant:.6g}".format(**result)
                        )
                        if "native_int8_from_float_ms" in result:
                            line += (
                                " native_i8_ms={native_int8_from_float_ms:.4f} "
                                "forced_vs_native_i8={forced_speedup_vs_native_int8_from_float:.3f}x"
                            ).format(**result)
                        if "native_module_auto_ms" in result:
                            line += (
                                " native_module_auto_ms={native_module_auto_ms:.4f} "
                                "native_module_auto_speedup={native_module_auto_speedup_vs_dequant:.3f}x"
                            ).format(**result)
                        if "direct_bitnet_int8_from_float_ms" in result:
                            line += (
                                " direct_bitnet_i8_ms={direct_bitnet_int8_from_float_ms:.4f} "
                                "direct_bitnet_vs_native_i8={direct_bitnet_speedup_vs_native_int8_from_float:.3f}x"
                            ).format(**result)
                        if "prequant_int8_matmul_ms" in result:
                            line += (
                                " prequant_i8_mm_ms={prequant_int8_matmul_ms:.4f} "
                                "forced_vs_prequant_i8_mm={forced_speedup_vs_prequant_int8_matmul:.3f}x"
                            ).format(**result)
                        if "forced_sync_wall_speedup_vs_dequant" in result:
                            line += (
                                " dense_sync_ms={dense_bitnet_dequant_sync_wall_ms:.4f} "
                                "forced_sync_ms={model_stack_forced_backend_sync_wall_ms:.4f} "
                                "forced_sync_speedup={forced_sync_wall_speedup_vs_dequant:.3f}x"
                            ).format(**result)
                        print(line)
        finally:
            for key, value in previous.items():
                _restore_env(key, value)


if __name__ == "__main__":
    main()
