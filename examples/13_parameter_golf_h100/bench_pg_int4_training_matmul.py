from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable

import torch

from compress.quantization import QuantizedLinearInt4


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn: Callable[[], torch.Tensor], *, device: torch.device, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    out = fn()
    _sync(device)
    for _ in range(max(0, warmup)):
        out = fn()
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, iters)):
            out = fn()
        end.record()
        _sync(device)
        return float(start.elapsed_time(end) / max(1, iters)), out
    start_s = time.perf_counter()
    for _ in range(max(1, iters)):
        out = fn()
    _sync(device)
    return float((time.perf_counter() - start_s) * 1000.0 / max(1, iters)), out


def _make_layers(in_features: int, out_features: int, *, device: torch.device, dtype: torch.dtype):
    dense = torch.nn.Linear(in_features, out_features, bias=False).to(device=device, dtype=dtype)
    int4 = QuantizedLinearInt4(in_features, out_features, bias=False).to(device=device, dtype=dtype)
    int4.from_float(dense)
    return dense, int4


def _bench_variant(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    ms, out = _time_ms(fn, device=device, warmup=warmup, iters=iters)
    return {
        "variant": name,
        "ms": ms,
        "finite": bool(torch.isfinite(out.detach()).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native packed INT4 forward-under-grad on PG MLP shapes.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(1234)
    dense, int4 = _make_layers(int(args.in_features), int(args.out_features), device=device, dtype=dtype)
    x = torch.randn(int(args.rows), int(args.in_features), device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn(int(args.rows), int(args.out_features), device=device, dtype=dtype)

    def dense_step() -> torch.Tensor:
        dense.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        out = dense(x)
        out.backward(grad)
        return out

    def int4_default_step() -> torch.Tensor:
        if x.grad is not None:
            x.grad = None
        old = os.environ.pop("MODEL_STACK_INT4_NATIVE_FORWARD_UNDER_GRAD", None)
        try:
            out = int4(x)
            out.backward(grad)
            return out
        finally:
            if old is not None:
                os.environ["MODEL_STACK_INT4_NATIVE_FORWARD_UNDER_GRAD"] = old

    def int4_native_forward_step() -> torch.Tensor:
        if x.grad is not None:
            x.grad = None
        old = os.environ.get("MODEL_STACK_INT4_NATIVE_FORWARD_UNDER_GRAD")
        os.environ["MODEL_STACK_INT4_NATIVE_FORWARD_UNDER_GRAD"] = "1"
        try:
            out = int4(x)
            out.backward(grad)
            return out
        finally:
            if old is None:
                os.environ.pop("MODEL_STACK_INT4_NATIVE_FORWARD_UNDER_GRAD", None)
            else:
                os.environ["MODEL_STACK_INT4_NATIVE_FORWARD_UNDER_GRAD"] = old

    rows = [
        _bench_variant("dense_train", dense_step, device=device, warmup=int(args.warmup), iters=int(args.iters)),
        _bench_variant("int4_default_train", int4_default_step, device=device, warmup=int(args.warmup), iters=int(args.iters)),
        _bench_variant("int4_native_forward_dense_backward_train", int4_native_forward_step, device=device, warmup=int(args.warmup), iters=int(args.iters)),
    ]
    baseline = float(rows[0]["ms"])
    for row in rows:
        row.update(
            {
                "rows": int(args.rows),
                "in_features": int(args.in_features),
                "out_features": int(args.out_features),
                "dtype": str(dtype).replace("torch.", ""),
                "device": str(device),
                "speedup_vs_dense": baseline / float(row["ms"]) if float(row["ms"]) > 0.0 else float("nan"),
            }
        )
        if args.jsonl:
            print(json.dumps(row, sort_keys=True))
        else:
            print(f"{row['variant']}: {row['ms']:.4f} ms speedup_vs_dense={row['speedup_vs_dense']:.3f}")


if __name__ == "__main__":
    main()
