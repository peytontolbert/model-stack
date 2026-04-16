from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from runtime.generation import build_generation_config as runtime_build_generation_config
from runtime.generation import resolve_generation_sampling_mode as runtime_resolve_generation_sampling_mode
from serve.engine import generate as decode_tokens


@dataclass
class ThroughputResult:
    tokens_per_sec: float
    latency_ms: float
    total_tokens: int
    total_time_s: float


@torch.inference_mode()
def benchmark_forward(
    model: torch.nn.Module,
    *,
    batch_size: int = 8,
    seq_len: int = 512,
    vocab_size: int | None = None,
    warmup_steps: int = 5,
    steps: int = 20,
    device: Optional[str | torch.device] = None,
) -> ThroughputResult:
    """Measure forward-only tokens/sec for next-token LM.

    Generates random input_ids and runs model(input_ids, None, None).
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    V = vocab_size or getattr(model, "vocab_size", 32000)
    x = torch.randint(0, int(V), (int(batch_size), int(seq_len)), device=dev, dtype=torch.long)

    # Warmup
    for _ in range(int(warmup_steps)):
        _ = model(x, None, None)

    torch.cuda.synchronize(dev) if dev.type == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(int(steps)):
        _ = model(x, None, None)
    torch.cuda.synchronize(dev) if dev.type == "cuda" else None
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_tokens = int(batch_size) * int(seq_len) * int(steps)
    tps = total_tokens / max(total_time, 1e-9)
    latency_ms = (total_time / max(int(steps), 1)) * 1000.0
    return ThroughputResult(tokens_per_sec=tps, latency_ms=latency_ms, total_tokens=total_tokens, total_time_s=total_time)


@torch.inference_mode()
def benchmark_generate(
    model: torch.nn.Module,
    *,
    batch_size: int = 1,
    seq_len: int = 128,
    max_new_tokens: int = 128,
    vocab_size: int | None = None,
    warmup_steps: int = 2,
    repeats: int = 5,
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
) -> ThroughputResult:
    """Measure decode throughput (new tokens/sec) for incremental generation.

    Uses serve.engine.generate with an optional kv_cache created by kv_cache_factory(batch_size).
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    V = vocab_size or getattr(model, "vocab_size", 32000)
    x = torch.randint(0, int(V), (int(batch_size), int(seq_len)), device=dev, dtype=torch.long)

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

    # Warmup
    for _ in range(int(warmup_steps)):
        _ = decode_tokens(model, x, cache=_allocate_cache(), config=cfg, cache_backend=cache_backend)

    torch.cuda.synchronize(dev) if dev.type == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(int(repeats)):
        _ = decode_tokens(model, x, cache=_allocate_cache(), config=cfg, cache_backend=cache_backend)
    torch.cuda.synchronize(dev) if dev.type == "cuda" else None
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_tokens = int(batch_size) * int(max_new_tokens) * int(repeats)
    tps = total_tokens / max(total_time, 1e-9)
    latency_ms = (total_time / max(int(repeats), 1)) * 1000.0
    return ThroughputResult(tokens_per_sec=tps, latency_ms=latency_ms, total_tokens=total_tokens, total_time_s=total_time)
