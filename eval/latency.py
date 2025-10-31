from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

from serve.engine import generate as decode_tokens, GenerationConfig


@dataclass
class LatencyDist:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples: int


def _percentiles_ms(times: List[float]) -> LatencyDist:
    if not times:
        return LatencyDist(0.0, 0.0, 0.0, 0)
    ts = sorted(times)
    def pct(p: float) -> float:
        idx = min(len(ts) - 1, max(0, int(round(p * (len(ts) - 1)))))
        return ts[idx] * 1000.0
    return LatencyDist(p50_ms=pct(0.50), p95_ms=pct(0.95), p99_ms=pct(0.99), samples=len(ts))


@torch.inference_mode()
def latency_forward(model: torch.nn.Module, *, repeats: int = 100, batch_size: int = 8, seq_len: int = 512, vocab_size: Optional[int] = None, device: Optional[str | torch.device] = None) -> LatencyDist:
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    V = vocab_size or getattr(model, "vocab_size", 32000)
    x = torch.randint(0, int(V), (int(batch_size), int(seq_len)), dtype=torch.long, device=dev)
    times: List[float] = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        t0 = time.perf_counter()
        _ = model(x, None, None)
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        times.append(time.perf_counter() - t0)
    return _percentiles_ms(times)


@torch.inference_mode()
def latency_generate(model: torch.nn.Module, *, repeats: int = 100, batch_size: int = 1, seq_len: int = 128, max_new_tokens: int = 1, vocab_size: Optional[int] = None, device: Optional[str | torch.device] = None) -> LatencyDist:
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    V = vocab_size or getattr(model, "vocab_size", 32000)
    x = torch.randint(0, int(V), (int(batch_size), int(seq_len)), dtype=torch.long, device=dev)
    cfg = GenerationConfig(max_new_tokens=int(max_new_tokens))
    times: List[float] = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        t0 = time.perf_counter()
        _ = decode_tokens(model, x, cache=None, config=cfg)
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        times.append(time.perf_counter() - t0)
    return _percentiles_ms(times)


