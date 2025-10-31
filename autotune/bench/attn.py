from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch


def benchmark_attention_backends(seq: int = 256, heads: int = 8, d_k: int = 64, dtype: torch.dtype = torch.float16, warmup: int = 10, iters: int = 50) -> List[Tuple[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 1
    q = torch.randn(B, heads, seq, d_k, device=device, dtype=dtype)
    k = torch.randn(B, heads, seq, d_k, device=device, dtype=dtype)
    v = torch.randn(B, heads, seq, d_k, device=device, dtype=dtype)

    from attn.backends import scaled_dot_product_attention

    backends = ["flash2", "xformers", "torch"]
    results: List[Tuple[str, float]] = []
    for be in backends:
        # warmup
        for _ in range(warmup):
            try:
                _ = scaled_dot_product_attention(q, k, v, backend=be, is_causal=True)
            except Exception:
                pass
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        n_ok = 0
        for _ in range(iters):
            try:
                _ = scaled_dot_product_attention(q, k, v, backend=be, is_causal=True)
                n_ok += 1
            except Exception:
                break
        torch.cuda.synchronize() if device.type == "cuda" else None
        if n_ok == 0:
            continue
        dt = (time.time() - t0) / n_ok
        results.append((be, dt))

    results.sort(key=lambda x: x[1])
    return results


def select_fastest_backend(seq: int = 256, heads: int = 8, d_k: int = 64, dtype: torch.dtype = torch.float16) -> str:
    res = benchmark_attention_backends(seq=seq, heads=heads, d_k=d_k, dtype=dtype)
    return res[0][0] if res else "torch"


