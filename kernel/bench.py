from __future__ import annotations

import time
from typing import Iterable

import torch

from attn.backends import scaled_dot_product_attention as sdpa
from attn.backends import select_attention_backend
from kernel import available as kernel_available


def _timeit(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.time() - t0) / iters


def bench_attention_backends(
    batch_size: int = 2,
    heads: int = 16,
    seq: int = 512,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    backends: Iterable[str] | None = None,
):
    B, H, T, D = batch_size, heads, seq, head_dim
    xq = torch.randn(B, H, T, D, device=device, dtype=dtype)
    xk = torch.randn(B, H, T, D, device=device, dtype=dtype)
    xv = torch.randn(B, H, T, D, device=device, dtype=dtype)

    if backends is None:
        # Probe availability
        avail = kernel_available()
        candidates = ["torch"]
        if avail.get("attn.flash2", False):
            candidates.append("flash2")
        if avail.get("attn.triton", False):
            candidates.append("triton")
        # Try xformers regardless of registry too
        try:
            import xformers  # type: ignore  # noqa: F401
            candidates.append("xformers")
        except Exception:
            pass
        backends = candidates

    results: dict[str, float] = {}
    for be in backends:
        def run():
            sdpa(xq, xk, xv, attn_mask=None, dropout_p=0.0, backend=be, is_causal=True)

        try:
            ms = _timeit(run) * 1000.0
            results[be] = ms
        except Exception as e:
            results[be] = float("nan")
    return results


if __name__ == "__main__":
    res = bench_attention_backends()
    for k, v in res.items():
        print(f"{k}: {v:.2f} ms/iter")


