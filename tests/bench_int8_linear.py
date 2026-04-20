from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.native import runtime_info
from runtime.quant import int8_linear_from_quantized_activation
from runtime.quant import quantize_activation_int8_rowwise


def _timeit(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return ((time.perf_counter() - start) * 1000.0) / max(iters, 1)


def _quantize_weight_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_f = weight.float()
    scale = weight_f.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    qweight = torch.round(weight_f / scale.unsqueeze(-1)).clamp_(-128.0, 127.0).to(torch.int8)
    return qweight, scale.to(dtype=torch.float32)


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark native int8 linear with the experimental SM90a WGMMA path against the non-WGMMA fallback."
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=384)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-cublaslt",
        action="store_true",
        help="Disable the large-shape cuBLASLt overlay to isolate WGMMA vs WMMA/tiled fallback routing.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_int8_linear.py")

    dtype_name = str(args.dtype).strip().lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype: {args.dtype}")

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda")
    dtype = dtype_map[dtype_name]

    x = torch.randn(args.batch, args.seq, args.in_features, device=device, dtype=dtype)
    weight = torch.randn(args.out_features, args.in_features, device=device, dtype=dtype) * 0.125
    bias = torch.randn(args.out_features, device=device, dtype=dtype) * 0.01

    qx, row_scale = quantize_activation_int8_rowwise(x)
    qweight, inv_scale = _quantize_weight_per_output_channel(weight)
    row_scale_view = row_scale.view(*x.shape[:-1], 1)
    quant_ref = torch.nn.functional.linear(
        qx.float() * row_scale_view,
        qweight.float() * inv_scale.unsqueeze(-1),
        bias.float(),
    ).view(args.batch, args.seq, args.out_features).to(dtype=dtype)

    info = runtime_info()
    print(f"kernel_family={info.get('int8_linear_kernel_family')}")
    print(f"wgmma_tile={info.get('int8_linear_wgmma_tile')}")
    print(f"wgmma_requires={info.get('int8_linear_wgmma_requires')}")
    print(f"wgmma_env={info.get('int8_linear_wgmma_env')}")
    print(f"wgmma_min_ops_env={info.get('int8_linear_wgmma_min_ops_env')}")
    print(f"wgmma_activation_strategy={info.get('int8_linear_wgmma_activation_strategy')}")

    if args.disable_cublaslt:
        _set_env("MODEL_STACK_DISABLE_INT8_LINEAR_CUBLASLT", "1")
    else:
        _set_env("MODEL_STACK_DISABLE_INT8_LINEAR_CUBLASLT", None)

    _set_env("MODEL_STACK_DISABLE_INT8_LINEAR_WGMMA", None)
    _set_env("MODEL_STACK_ENABLE_INT8_LINEAR_WGMMA", "1")
    out_wgmma = int8_linear_from_quantized_activation(
        qx,
        row_scale,
        qweight,
        inv_scale,
        bias=bias,
        out_dtype=dtype,
    )
    wgmma_ms = _timeit(
        lambda: int8_linear_from_quantized_activation(
            qx,
            row_scale,
            qweight,
            inv_scale,
            bias=bias,
            out_dtype=dtype,
        ),
        warmup=int(args.warmup),
        iters=int(args.iters),
    )

    _set_env("MODEL_STACK_DISABLE_INT8_LINEAR_WGMMA", "1")
    out_fallback = int8_linear_from_quantized_activation(
        qx,
        row_scale,
        qweight,
        inv_scale,
        bias=bias,
        out_dtype=dtype,
    )
    fallback_ms = _timeit(
        lambda: int8_linear_from_quantized_activation(
            qx,
            row_scale,
            qweight,
            inv_scale,
            bias=bias,
            out_dtype=dtype,
        ),
        warmup=int(args.warmup),
        iters=int(args.iters),
    )

    max_err_vs_fallback = float((out_wgmma - out_fallback).abs().max().item())
    max_err_vs_quant_ref = float((out_wgmma - quant_ref).abs().max().item())
    speedup = fallback_ms / wgmma_ms if wgmma_ms > 0 else float("inf")

    print(f"device={device} dtype={dtype} batch={args.batch} seq={args.seq}")
    print(f"in_features={args.in_features} out_features={args.out_features}")
    print(f"disable_cublaslt={bool(args.disable_cublaslt)}")
    print(f"wgmma_ms={wgmma_ms:.4f}")
    print(f"fallback_ms={fallback_ms:.4f}")
    print(f"speedup={speedup:.3f}x")
    print(f"max_abs_err_vs_fallback={max_err_vs_fallback:.6f}")
    print(f"max_abs_err_vs_quant_ref={max_err_vs_quant_ref:.6f}")


if __name__ == "__main__":
    main()
