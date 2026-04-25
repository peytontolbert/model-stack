from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.causal import CausalLM
from specs.config import ModelConfig
from tests.bench_bitnet_decode import (
    _build_generation_session,
    _clone_generation_session,
    _configure_attention_backend,
    _convert_model_to_bitnet,
    _dtype_from_name,
)


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
        }
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _bytes_to_mib(value: int) -> float:
    return float(value) / float(1024 ** 2)


def _delta_mib(after: dict[str, int], before: dict[str, int], key: str) -> float:
    return _bytes_to_mib(int(after[key]) - int(before[key]))


def _apply_cache_mode(mode: str) -> None:
    normalized = str(mode).strip().lower()
    if normalized == "full":
        os.environ.pop("MODEL_STACK_DISABLE_HOPPER_DENSE_DECODE_CACHE", None)
        os.environ["MODEL_STACK_HOPPER_DENSE_DECODE_CACHE_MODE"] = "full"
        return
    if normalized in {"", "qkv", "qkv_only"}:
        os.environ.pop("MODEL_STACK_DISABLE_HOPPER_DENSE_DECODE_CACHE", None)
        os.environ["MODEL_STACK_HOPPER_DENSE_DECODE_CACHE_MODE"] = "qkv_only"
        return
    if normalized in {"off", "disabled", "none"}:
        os.environ["MODEL_STACK_DISABLE_HOPPER_DENSE_DECODE_CACHE"] = "1"
        os.environ.pop("MODEL_STACK_HOPPER_DENSE_DECODE_CACHE_MODE", None)
        return
    raise ValueError(f"unsupported cache mode: {mode}")


def _time_fresh_clone_decode(
    session,
    *,
    warmup: int,
    iters: int,
    use_cuda_events: bool,
) -> float:
    total = int(warmup) + int(iters)
    measurements_ms: list[float] = []
    for idx in range(total):
        clone = _clone_generation_session(session, enabled=True)
        native = getattr(clone, "_native_session", None)
        if native is None:
            raise RuntimeError("native session unavailable for cloned serving benchmark")
        if use_cuda_events:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            native.decode_next_logits()
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
        else:
            import time
            t0 = time.perf_counter()
            native.decode_next_logits()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if idx >= int(warmup):
            measurements_ms.append(elapsed_ms)
        del clone
    return sum(measurements_ms) / max(len(measurements_ms), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serving-oriented H100 BitNet benchmark for decode-cache policy tradeoffs."
    )
    parser.add_argument("--cache-mode", choices=("full", "qkv_only", "off"), default="qkv_only")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=6912)
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--activation-quant", type=str, default="dynamic_int8")
    parser.add_argument("--activation-quant-bits", type=int, default=8)
    parser.add_argument("--packed-backend", choices=("auto", "disabled"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    _apply_cache_mode(args.cache_mode)
    torch.manual_seed(int(args.seed))

    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)
    use_cuda_events = device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)

    baseline = _memory_snapshot(device)

    cfg = ModelConfig(
        d_model=int(args.heads) * int(args.head_dim),
        n_heads=int(args.heads),
        n_layers=int(args.layers),
        d_ff=int(args.ff_dim),
        vocab_size=int(args.vocab_size),
        dtype=str(args.dtype),
    )
    model = CausalLM(cfg, block_variant="llama", tie_weights=False).to(device=device, dtype=dtype).eval()
    _convert_model_to_bitnet(
        model,
        activation_quant=str(args.activation_quant),
        activation_quant_bits=int(args.activation_quant_bits),
        activation_quant_method="absmax",
        activation_quant_percentile=0.999,
        spin=False,
    )
    _configure_attention_backend(model, packed_backend=str(args.packed_backend))
    after_model = _memory_snapshot(device)

    seq = torch.randint(0, cfg.vocab_size, (int(args.batch), int(args.prompt)), device=device)
    session, executor_kind = _build_generation_session(
        model,
        seq,
        enabled=True,
        cache_pagesize=max(int(args.prompt), 1),
    )
    if session is None or getattr(session, "native_executor_kind", "python") != "causal_lm":
        raise RuntimeError("native causal executor unavailable")
    after_session = _memory_snapshot(device)

    with torch.no_grad():
        prefill = session.prefill_next_logits()
    if prefill is None:
        raise RuntimeError("expected KV-backed prefill session")
    next_id = torch.argmax(prefill, dim=-1, keepdim=True)
    session.append(next_id)
    after_prefill = _memory_snapshot(device)

    decode_ms = _time_fresh_clone_decode(
        session,
        warmup=int(args.warmup),
        iters=int(args.iters),
        use_cuda_events=use_cuda_events,
    )
    after_bench = _memory_snapshot(device)

    payload: dict[str, Any] = {
        "cache_mode": str(args.cache_mode),
        "packed_backend": str(args.packed_backend),
        "activation_quant": str(args.activation_quant),
        "activation_quant_bits": int(args.activation_quant_bits),
        "batch": int(args.batch),
        "prompt": int(args.prompt),
        "layers": int(args.layers),
        "heads": int(args.heads),
        "head_dim": int(args.head_dim),
        "ff_dim": int(args.ff_dim),
        "vocab_size": int(args.vocab_size),
        "dtype": str(dtype),
        "device": str(device),
        "native_executor_kind": str(getattr(session, "native_executor_kind", executor_kind)),
        "decode_ms": float(decode_ms),
        "memory": {
            "baseline": baseline,
            "after_model": after_model,
            "after_session": after_session,
            "after_prefill": after_prefill,
            "after_bench": after_bench,
            "model_delta_mib": _delta_mib(after_model, baseline, "allocated_bytes"),
            "session_delta_mib": _delta_mib(after_session, after_model, "allocated_bytes"),
            "prefill_delta_mib": _delta_mib(after_prefill, after_session, "allocated_bytes"),
            "bench_delta_mib": _delta_mib(after_bench, after_prefill, "allocated_bytes"),
            "total_delta_mib": _delta_mib(after_bench, baseline, "allocated_bytes"),
            "peak_allocated_mib": _bytes_to_mib(after_bench["max_allocated_bytes"]),
            "peak_reserved_mib": _bytes_to_mib(after_bench["max_reserved_bytes"]),
        },
        "env": {
            "MODEL_STACK_DISABLE_HOPPER_DENSE_DECODE_CACHE": os.getenv(
                "MODEL_STACK_DISABLE_HOPPER_DENSE_DECODE_CACHE", ""
            ),
            "MODEL_STACK_HOPPER_DENSE_DECODE_CACHE_MODE": os.getenv(
                "MODEL_STACK_HOPPER_DENSE_DECODE_CACHE_MODE", ""
            ),
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"cache_mode={payload['cache_mode']}")
        print(f"decode_ms={payload['decode_ms']:.6f}")
        print(f"model_delta_mib={payload['memory']['model_delta_mib']:.2f}")
        print(f"session_delta_mib={payload['memory']['session_delta_mib']:.2f}")
        print(f"prefill_delta_mib={payload['memory']['prefill_delta_mib']:.2f}")
        print(f"total_delta_mib={payload['memory']['total_delta_mib']:.2f}")
        print(f"peak_allocated_mib={payload['memory']['peak_allocated_mib']:.2f}")
        print(f"peak_reserved_mib={payload['memory']['peak_reserved_mib']:.2f}")


if __name__ == "__main__":
    main()
