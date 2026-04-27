from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F


def _dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_once(fn: Callable[[], torch.Tensor], *, grad_out: torch.Tensor, warmup: int, iters: int, device: torch.device) -> float:
    params = tuple(t for t in (getattr(fn, "q"), getattr(fn, "k"), getattr(fn, "v")) if isinstance(t, torch.Tensor))
    for _ in range(warmup):
        for tensor in params:
            tensor.grad = None
        out = fn()
        out.backward(grad_out)
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            for tensor in params:
                tensor.grad = None
            out = fn()
            out.backward(grad_out)
        end.record()
        _sync(device)
        return float(start.elapsed_time(end) / max(iters, 1))
    t0 = time.perf_counter()
    for _ in range(iters):
        for tensor in params:
            tensor.grad = None
        out = fn()
        out.backward(grad_out)
    return float((time.perf_counter() - t0) * 1000.0 / max(iters, 1))


def _time_median(fn: Callable[[], torch.Tensor], *, grad_out: torch.Tensor, warmup: int, iters: int, repeats: int, device: torch.device) -> float:
    values = [
        _time_once(fn, grad_out=grad_out, warmup=warmup, iters=iters, device=device)
        for _ in range(max(1, repeats))
    ]
    return float(statistics.median(values))


class AttentionVariant(torch.nn.Module):
    def __init__(self, variant: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        super().__init__()
        self.variant = str(variant)
        self.q = torch.nn.Parameter(q)
        self.k = torch.nn.Parameter(k)
        self.v = torch.nn.Parameter(v)

    def forward(self) -> torch.Tensor:
        if self.variant == "gqa":
            return F.scaled_dot_product_attention(self.q, self.k, self.v, is_causal=True, enable_gqa=True)
        groups = self.q.size(1) // self.k.size(1)
        if self.variant == "repeat":
            k = self.k.repeat_interleave(groups, dim=1)
            v = self.v.repeat_interleave(groups, dim=1)
        elif self.variant == "repeat_contiguous":
            k = self.k.repeat_interleave(groups, dim=1).contiguous()
            v = self.v.repeat_interleave(groups, dim=1).contiguous()
        elif self.variant == "expand_reshape":
            bsz, kv_heads, seq_len, head_dim = self.k.shape
            k = self.k[:, :, None, :, :].expand(bsz, kv_heads, groups, seq_len, head_dim).reshape(
                bsz,
                kv_heads * groups,
                seq_len,
                head_dim,
            )
            v = self.v[:, :, None, :, :].expand(bsz, kv_heads, groups, seq_len, head_dim).reshape(
                bsz,
                kv_heads * groups,
                seq_len,
                head_dim,
            )
        else:
            raise ValueError(f"unknown attention variant: {self.variant}")
        return F.scaled_dot_product_attention(self.q, k, v, is_causal=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PG GQA attention implementation variants.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--compile-module", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)
    dtype = _dtype(args.dtype)
    torch.manual_seed(1337)
    q = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch_size, args.num_kv_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch_size, args.num_kv_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    grad_out = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)

    header = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "num_heads": int(args.num_heads),
        "num_kv_heads": int(args.num_kv_heads),
        "head_dim": int(args.head_dim),
        "compile_module": bool(args.compile_module),
    }
    print(json.dumps({"header": header}) if args.jsonl else header)

    baseline_ms = None
    for variant in ("gqa", "repeat", "repeat_contiguous", "expand_reshape"):
        module = AttentionVariant(variant, q.detach().clone(), k.detach().clone(), v.detach().clone()).to(device)
        runner = torch.compile(module, dynamic=False, fullgraph=True) if args.compile_module and device.type == "cuda" else module
        ms = _time_median(runner, grad_out=grad_out, warmup=args.warmup, iters=args.iters, repeats=args.repeats, device=device)
        if baseline_ms is None:
            baseline_ms = ms
        result = {
            "variant": variant,
            "train_step_ms": ms,
            "speedup_vs_gqa": baseline_ms / ms if baseline_ms and ms > 0 else 1.0,
        }
        print(json.dumps(result) if args.jsonl else result)


if __name__ == "__main__":
    main()
