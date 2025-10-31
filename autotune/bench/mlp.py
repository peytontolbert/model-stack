from __future__ import annotations

import time
from typing import List, Tuple

import torch

from tensor.mlp import MLP


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


def bench_mlp(hidden_size: int, ff_size: int, activations: List[str] | None = None, dtype: torch.dtype = torch.bfloat16) -> List[Tuple[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T = 4, 256
    x = torch.randn(B, T, hidden_size, device=device, dtype=dtype)
    acts = activations or ["silu", "gelu", "swiglu", "geglu", "reglu"]
    results: List[Tuple[str, float]] = []
    for act in acts:
        try:
            mlp = MLP(hidden_size, ff_size, activation=act, dropout_p=0.0).to(device=device, dtype=dtype)
            dt = _time_callable(mlp, x)
            results.append((act, dt))
        except Exception:
            continue
    results.sort(key=lambda x: x[1])
    return results



