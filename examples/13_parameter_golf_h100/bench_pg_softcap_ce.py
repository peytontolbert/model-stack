from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from tensor.losses import softcapped_cross_entropy, softcapped_cross_entropy_manual


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn: Callable[[], torch.Tensor], *, device: torch.device, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    out = fn()
    _sync(device)
    for _ in range(max(0, warmup)):
        out = fn()
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, iters)):
            out = fn()
        end.record()
        _sync(device)
        return float(start.elapsed_time(end) / max(1, iters)), out
    start_s = time.perf_counter()
    for _ in range(max(1, iters)):
        out = fn()
    _sync(device)
    return float((time.perf_counter() - start_s) * 1000.0 / max(1, iters)), out


def _compiled(fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor], enabled: bool):
    if not enabled:
        return fn
    return torch.compile(fn, mode="reduce-overhead", fullgraph=True)


def _torch_ce(logits: torch.Tensor, targets: torch.Tensor, softcap: float) -> torch.Tensor:
    capped = float(softcap) * torch.tanh(logits / float(softcap))
    return F.cross_entropy(capped.float(), targets, reduction="mean")


def _helper_ce(logits: torch.Tensor, targets: torch.Tensor, softcap: float) -> torch.Tensor:
    return softcapped_cross_entropy(logits, targets, softcap=float(softcap), reduction="mean")


def _manual_logsumexp_ce(logits: torch.Tensor, targets: torch.Tensor, softcap: float) -> torch.Tensor:
    return softcapped_cross_entropy_manual(logits, targets, softcap=float(softcap), reduction="mean")


def _make_logits_targets(rows: int, vocab: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(rows, vocab, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, vocab, (rows,), device=device, dtype=torch.long)
    return logits, targets


def _bench_one(
    name: str,
    fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    softcap: float,
    device: torch.device,
    warmup: int,
    iters: int,
    backward: bool,
) -> dict[str, object]:
    def step() -> torch.Tensor:
        if logits.grad is not None:
            logits.grad = None
        loss = fn(logits, targets, softcap)
        if backward:
            loss.backward()
        return loss.detach()

    ms, loss = _time_ms(step, device=device, warmup=warmup, iters=iters)
    grad_norm = None
    if backward and logits.grad is not None:
        grad_norm = float(logits.grad.float().norm().item())
    return {
        "variant": name,
        "ms": ms,
        "loss": float(loss.float().item()),
        "grad_norm": grad_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Parameter Golf softcapped cross entropy variants.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--softcap", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(1234)
    logits, targets = _make_logits_targets(int(args.rows), int(args.vocab), device, dtype)

    compile_enabled = not bool(args.no_compile)
    variants = [
        ("torch_softcap_ce", _torch_ce),
        ("helper_softcap_ce", _helper_ce),
        ("manual_logsumexp_ce", _manual_logsumexp_ce),
    ]
    rows = []
    for name, fn in variants:
        bench_fn = _compiled(fn, compile_enabled and name != "helper_softcap_ce")
        result = _bench_one(
            name + ("_compiled" if compile_enabled and name != "helper_softcap_ce" else ""),
            bench_fn,
            logits,
            targets,
            softcap=float(args.softcap),
            device=device,
            warmup=int(args.warmup),
            iters=int(args.iters),
            backward=bool(args.backward),
        )
        result.update(
            {
                "rows": int(args.rows),
                "vocab": int(args.vocab),
                "dtype": str(dtype).replace("torch.", ""),
                "device": str(device),
                "backward": bool(args.backward),
                "softcap": float(args.softcap),
            }
        )
        rows.append(result)

    baseline = rows[0]["ms"]
    for row in rows:
        row["speedup_vs_torch"] = float(baseline) / float(row["ms"]) if float(row["ms"]) > 0.0 else math.nan
        if args.jsonl:
            print(json.dumps(row, sort_keys=True))
        else:
            print(
                f"{row['variant']}: {row['ms']:.4f} ms "
                f"speedup_vs_torch={row['speedup_vs_torch']:.3f} loss={row['loss']:.6f}"
            )


if __name__ == "__main__":
    main()
