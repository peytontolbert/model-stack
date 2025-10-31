from __future__ import annotations

import time
from typing import Iterable, List, Tuple

import torch

from attn.kv_cache import PagedKVCache


def _sync_if_cuda() -> None:
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def _time_callable(fn, *args, warmup: int = 10, repeat: int = 50, **kwargs) -> float:
    for _ in range(max(0, int(warmup))):
        _ = fn(*args, **kwargs)
    _sync_if_cuda()
    t0 = time.time()
    for _ in range(max(1, int(repeat))):
        _ = fn(*args, **kwargs)
    _sync_if_cuda()
    return (time.time() - t0) / max(1, int(repeat))


def bench_kvcache(pagesizes: Iterable[int], batch: int, n_layers: int, n_kv_heads: int, head_dim: int, seq: int, dtype: torch.dtype = torch.bfloat16) -> List[Tuple[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: List[Tuple[str, float]] = []
    for ps in pagesizes:
        try:
            cache = PagedKVCache(batch, n_layers, n_kv_heads, head_dim, int(ps), dtype, device)
            # One layer microbench: append tokens then slice full range
            Hk, D = n_kv_heads, head_dim
            def step():
                for b in range(batch):
                    k = torch.randn(Hk, 1, D, device=device, dtype=dtype)
                    v = torch.randn(Hk, 1, D, device=device, dtype=dtype)
                    cache.append(0, b, k, v)
                _ = cache.slice(0, 0, cache.lengths[0][0])
                return 0
            dt = _time_callable(step, warmup=10, repeat=max(1, int(seq))) / max(1, int(seq))
            results.append((f"pagesize={ps}", dt))
        except Exception:
            continue
    results.sort(key=lambda x: x[1])
    return results



