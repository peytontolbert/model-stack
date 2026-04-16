from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

from runtime.generation import build_generation_config as runtime_build_generation_config
from runtime.generation import resolve_generation_sampling_mode as runtime_resolve_generation_sampling_mode
from serve.engine import generate as decode_tokens


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
def latency_generate(
    model: torch.nn.Module,
    *,
    repeats: int = 100,
    batch_size: int = 1,
    seq_len: int = 128,
    max_new_tokens: int = 1,
    vocab_size: Optional[int] = None,
    device: Optional[str | torch.device] = None,
    kv_cache_factory=None,
    do_sample: bool | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    no_repeat_ngram: int = 0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    sliding_window: int | None = None,
    cache_backend: str | None = None,
) -> LatencyDist:
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    V = vocab_size or getattr(model, "vocab_size", 32000)
    x = torch.randint(0, int(V), (int(batch_size), int(seq_len)), dtype=torch.long, device=dev)
    resolved_do_sample = runtime_resolve_generation_sampling_mode(
        do_sample=do_sample,
        temperature=float(temperature),
        top_k=top_k,
        top_p=top_p,
    )
    cfg = runtime_build_generation_config(
        max_new_tokens=int(max_new_tokens),
        do_sample=resolved_do_sample,
        temperature=float(temperature),
        top_k=(int(top_k) if top_k is not None else None),
        top_p=(float(top_p) if top_p is not None else None),
        eos_id=(int(eos_id) if eos_id is not None else None),
        no_repeat_ngram=int(no_repeat_ngram),
        repetition_penalty=float(repetition_penalty),
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
        sliding_window=(int(sliding_window) if sliding_window is not None else None),
    )

    def _allocate_cache():
        if kv_cache_factory is None:
            return None
        try:
            return kv_cache_factory(batch_size=int(batch_size), backend=cache_backend)
        except TypeError:
            return kv_cache_factory(batch_size=int(batch_size))

    times: List[float] = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        t0 = time.perf_counter()
        _ = decode_tokens(model, x, cache=_allocate_cache(), config=cfg, cache_backend=cache_backend)
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        times.append(time.perf_counter() - t0)
    return _percentiles_ms(times)
