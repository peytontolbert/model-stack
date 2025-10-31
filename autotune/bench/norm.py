from __future__ import annotations

import time
from typing import List, Tuple

import torch

from tensor.norms import RMSNorm, layer_norm as fn_layer_norm, rmsnorm as fn_rmsnorm


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


def bench_norms(hidden_size: int, dtype: torch.dtype = torch.bfloat16) -> List[Tuple[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T = 4, 256
    x = torch.randn(B, T, hidden_size, device=device, dtype=dtype)
    w = torch.ones(hidden_size, device=device, dtype=dtype)

    results: List[Tuple[str, float]] = []
    try:
        mod = RMSNorm(hidden_size).to(device=device, dtype=dtype)
        results.append(("RMSNorm(module)", _time_callable(mod, x)))
    except Exception:
        pass
    try:
        results.append(("rmsnorm(functional)", _time_callable(fn_rmsnorm, x, w)))
    except Exception:
        pass
    try:
        results.append(("layer_norm(functional)", _time_callable(fn_layer_norm, x, w)))
    except Exception:
        pass
    results.sort(key=lambda x: x[1])
    return results



