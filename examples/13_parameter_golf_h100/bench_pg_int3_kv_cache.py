import argparse
import json
import os
import time

import torch

from runtime.kv_cache import quantize_pack_int3_lastdim, unpack_dequantize_int3_lastdim


def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / max(iters, 1)


def _bench_one(x: torch.Tensor, warmup: int, iters: int, native: bool) -> dict[str, float | str]:
    if native:
        os.environ.pop("MODEL_STACK_DISABLE_NATIVE_INT3_KV", None)
        label = "native_cuda_pack_dequant"
    else:
        os.environ["MODEL_STACK_DISABLE_NATIVE_INT3_KV"] = "1"
        label = "python_reference_pack_dequant"

    def run():
        packed, scale, dim = quantize_pack_int3_lastdim(x)
        return unpack_dequantize_int3_lastdim(packed, scale, dim, dtype=x.dtype)

    ms = _time_cuda(run, warmup, iters)
    y = run()
    torch.cuda.synchronize()
    max_steps = ((x.float() - y.float()).abs() / x.float().abs().amax(dim=-1).clamp_min(1e-8).div(3.0).unsqueeze(-1)).max()
    return {
        "case": label,
        "ms": ms,
        "max_quant_steps": float(max_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    x = torch.randn(args.batch, args.heads, args.seq, args.head_dim, device=args.device, dtype=dtype)

    results = [
        _bench_one(x, args.warmup, args.iters, native=False),
        _bench_one(x, args.warmup, args.iters, native=True),
    ]
    ref_ms = next(r["ms"] for r in results if r["case"] == "python_reference_pack_dequant")
    native_ms = next(r["ms"] for r in results if r["case"] == "native_cuda_pack_dequant")
    out = {
        "shape": list(x.shape),
        "dtype": str(dtype).replace("torch.", ""),
        "results": results,
        "native_speedup": ref_ms / native_ms,
    }
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
