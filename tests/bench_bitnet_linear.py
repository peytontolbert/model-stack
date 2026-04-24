from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.quantization import QuantizedLinearBitNet
from runtime.ops import pack_bitnet_weight
from runtime.native import has_native_op, native_module
from runtime.quant import bitnet_linear
from runtime.quant import bitnet_linear_from_float
from runtime.quant import _dequantize_packed_bitnet_weight


def _timeit(fn, *, warmup: int, iters: int, use_cuda_events: bool) -> float:
    for _ in range(warmup):
        fn()
    if use_cuda_events:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / max(iters, 1)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return ((time.perf_counter() - t0) * 1000.0) / max(iters, 1)


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[normalized]


def _to_bitnet_linear(
    linear: torch.nn.Linear,
    *,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
    spin: bool,
) -> QuantizedLinearBitNet:
    return QuantizedLinearBitNet(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
    ).from_float(
        linear,
        activation_quant=activation_quant,
        activation_quant_bits=activation_quant_bits,
        activation_quant_method=activation_quant_method,
        activation_quant_percentile=activation_quant_percentile,
        spin=spin,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native BitNet linear against dense linear.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--activation-quant", type=str, default="dynamic_int8")
    parser.add_argument("--activation-quant-bits", type=int, default=8)
    parser.add_argument("--activation-quant-method", type=str, default="absmax")
    parser.add_argument("--activation-quant-percentile", type=float, default=0.999)
    parser.add_argument("--spin", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)
    use_cuda_events = device.type == "cuda"
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    x = torch.randn(args.batch, args.seq, args.in_features, device=device, dtype=dtype)
    weight = torch.randn(args.out_features, args.in_features, device=device, dtype=dtype) * 0.125
    bias = torch.randn(args.out_features, device=device, dtype=dtype) * 0.01
    dense_linear = torch.nn.Linear(args.in_features, args.out_features, bias=True, device=device, dtype=dtype)
    with torch.no_grad():
        dense_linear.weight.copy_(weight)
        dense_linear.bias.copy_(bias)
    bitnet_layer = _to_bitnet_linear(
        dense_linear,
        activation_quant=str(args.activation_quant),
        activation_quant_bits=int(args.activation_quant_bits),
        activation_quant_method=str(args.activation_quant_method),
        activation_quant_percentile=float(args.activation_quant_percentile),
        spin=bool(args.spin),
    ).to(device=device)

    packed_weight, scale_values, layout_header, segment_offsets = pack_bitnet_weight(weight)
    dequant_weight = _dequantize_packed_bitnet_weight(
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        dtype=dtype,
    )
    module = native_module()
    native_bitnet_available = (
        device.type == "cuda"
        and has_native_op("bitnet_linear_from_float")
        and module is not None
        and hasattr(module, "bitnet_linear_from_float_forward")
    )
    native_dense_available = (
        device.type == "cuda"
        and has_native_op("linear")
        and module is not None
        and hasattr(module, "linear_forward")
    )
    static_act_scale = bitnet_layer.act_scale if str(args.activation_quant).strip().lower() == "static_int8" else None
    spin_signs = bitnet_layer.spin_signs if bool(args.spin) else None

    with torch.no_grad():
        dense_out = torch.nn.functional.linear(x, weight, bias)
        dequant_ref_out = torch.nn.functional.linear(x, dequant_weight, bias)
        wrapper_out = bitnet_linear(
            x,
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
            bias=bias,
        )
        wrapper_from_float_out = bitnet_linear_from_float(
            x,
            bitnet_layer.packed_weight,
            bitnet_layer.scale_values,
            bitnet_layer.layout_header,
            bitnet_layer.segment_offsets,
            bias=bitnet_layer.bias,
            spin_enabled=bool(args.spin),
            spin_signs=spin_signs,
            pre_scale=bitnet_layer.pre_scale,
            act_quant_mode=str(args.activation_quant),
            act_scale=static_act_scale,
            act_quant_bits=int(args.activation_quant_bits),
            act_quant_method=str(args.activation_quant_method),
            act_quant_percentile=float(args.activation_quant_percentile),
        )
        module_out = bitnet_layer.runtime_linear(x)
        native_out = (
            module.bitnet_linear_from_float_forward(
                x,
                bitnet_layer.packed_weight,
                bitnet_layer.scale_values,
                bitnet_layer.layout_header,
                bitnet_layer.segment_offsets,
                bitnet_layer.bias,
                bool(args.spin),
                spin_signs,
                bitnet_layer.pre_scale,
                str(args.activation_quant),
                str(args.activation_quant_method),
                int(args.activation_quant_bits),
                float(args.activation_quant_percentile),
                static_act_scale,
                dtype,
            )
            if native_bitnet_available
            else wrapper_from_float_out
        )
    max_abs_err_vs_dense = float((wrapper_out - dense_out).abs().max().item())
    max_abs_err_vs_ref = float((wrapper_out - dequant_ref_out).abs().max().item())
    wrapper_from_float_max_abs_err_vs_ref = float((wrapper_from_float_out - dequant_ref_out).abs().max().item())
    module_max_abs_err_vs_ref = float((module_out - dequant_ref_out).abs().max().item())
    native_max_abs_err_vs_ref = float((native_out - dequant_ref_out).abs().max().item())

    with torch.inference_mode():
        pack_ms = _timeit(
            lambda: pack_bitnet_weight(weight),
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_events=use_cuda_events,
        )
        dense_ms = _timeit(
            lambda: torch.nn.functional.linear(x, weight, bias),
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_events=use_cuda_events,
        )
        dequant_ref_ms = _timeit(
            lambda: torch.nn.functional.linear(x, dequant_weight, bias),
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_events=use_cuda_events,
        )
        wrapper_ms = _timeit(
            lambda: bitnet_linear(
                x,
                packed_weight,
                scale_values,
                layout_header,
                segment_offsets,
                bias=bias,
            ),
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_events=use_cuda_events,
        )
        wrapper_from_float_ms = _timeit(
            lambda: bitnet_linear_from_float(
                x,
                bitnet_layer.packed_weight,
                bitnet_layer.scale_values,
                bitnet_layer.layout_header,
                bitnet_layer.segment_offsets,
                bias=bitnet_layer.bias,
                spin_enabled=bool(args.spin),
                spin_signs=spin_signs,
                pre_scale=bitnet_layer.pre_scale,
                act_quant_mode=str(args.activation_quant),
                act_scale=static_act_scale,
                act_quant_bits=int(args.activation_quant_bits),
                act_quant_method=str(args.activation_quant_method),
                act_quant_percentile=float(args.activation_quant_percentile),
            ),
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_events=use_cuda_events,
        )
        module_ms = _timeit(
            lambda: bitnet_layer.runtime_linear(x),
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_events=use_cuda_events,
        )
        native_bitnet_ms = (
            _timeit(
                lambda: module.bitnet_linear_from_float_forward(
                    x,
                    bitnet_layer.packed_weight,
                    bitnet_layer.scale_values,
                    bitnet_layer.layout_header,
                    bitnet_layer.segment_offsets,
                    bitnet_layer.bias,
                    bool(args.spin),
                    spin_signs,
                    bitnet_layer.pre_scale,
                    str(args.activation_quant),
                    str(args.activation_quant_method),
                    int(args.activation_quant_bits),
                    float(args.activation_quant_percentile),
                    static_act_scale,
                    dtype,
                ),
                warmup=args.warmup,
                iters=args.iters,
                use_cuda_events=use_cuda_events,
            )
            if native_bitnet_available
            else float("nan")
        )
        native_dense_ms = (
            _timeit(
                lambda: module.linear_forward(x, dequant_weight, bias, "auto"),
                warmup=args.warmup,
                iters=args.iters,
                use_cuda_events=use_cuda_events,
            )
            if native_dense_available
            else float("nan")
        )
    speedup_vs_ref = dequant_ref_ms / native_bitnet_ms if native_bitnet_available and native_bitnet_ms > 0 else float("nan")

    print(f"device={device} dtype={dtype} batch={args.batch} seq={args.seq}")
    print(f"in_features={args.in_features} out_features={args.out_features}")
    print(
        "activation_quant="
        f"{args.activation_quant} bits={args.activation_quant_bits} "
        f"method={args.activation_quant_method} percentile={args.activation_quant_percentile} "
        f"spin={bool(args.spin)}"
    )
    print(f"pack_ms={pack_ms:.3f}")
    print(f"dense_original_ms={dense_ms:.3f}")
    print(f"dense_bitnet_ref_ms={dequant_ref_ms:.3f}")
    if native_dense_available:
        print(f"native_dense_bitnet_ref_ms={native_dense_ms:.3f}")
    print(f"bitnet_wrapper_ms={wrapper_ms:.3f}")
    print(f"bitnet_wrapper_from_float_ms={wrapper_from_float_ms:.3f}")
    print(f"bitnet_module_ms={module_ms:.3f}")
    if native_bitnet_available:
        print(f"bitnet_native_ms={native_bitnet_ms:.3f}")
        print(f"speedup_vs_bitnet_ref={speedup_vs_ref:.3f}x")
    print(f"max_abs_err_vs_dense={max_abs_err_vs_dense:.6f}")
    print(f"max_abs_err_vs_bitnet_ref={max_abs_err_vs_ref:.6f}")
    print(f"wrapper_from_float_max_abs_err_vs_bitnet_ref={wrapper_from_float_max_abs_err_vs_ref:.6f}")
    print(f"module_max_abs_err_vs_bitnet_ref={module_max_abs_err_vs_ref:.6f}")
    print(f"native_max_abs_err_vs_bitnet_ref={native_max_abs_err_vs_ref:.6f}")


if __name__ == "__main__":
    main()
