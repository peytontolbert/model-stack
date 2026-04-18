from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.ops import pack_bitnet_weight
from runtime.quant import bitnet_linear


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native BitNet linear against dense linear.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    packed_weight, scale_values, layout_header, segment_offsets = pack_bitnet_weight(weight)

    with torch.no_grad():
        dense_out = torch.nn.functional.linear(x, weight, bias)
        bitnet_out = bitnet_linear(
            x,
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
            bias=bias,
        )
    max_abs_err = float((bitnet_out - dense_out).abs().max().item())

    dense_ms = _timeit(
        lambda: torch.nn.functional.linear(x, weight, bias),
        warmup=args.warmup,
        iters=args.iters,
        use_cuda_events=use_cuda_events,
    )
    bitnet_ms = _timeit(
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
    speedup = dense_ms / bitnet_ms if bitnet_ms > 0 else float("inf")

    print(f"device={device} dtype={dtype} batch={args.batch} seq={args.seq}")
    print(f"in_features={args.in_features} out_features={args.out_features}")
    print(f"dense_ms={dense_ms:.3f}")
    print(f"bitnet_ms={bitnet_ms:.3f}")
    print(f"speedup={speedup:.3f}x")
    print(f"max_abs_err={max_abs_err:.6f}")


if __name__ == "__main__":
    main()
