from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch

from kernel import available as kernel_available, get as kernel_get


def _sync_if_cuda() -> None:
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def list_kernels(prefix: str | None = None) -> Dict[str, bool]:
    avail = kernel_available()
    if prefix is None:
        return avail
    prefix = prefix.lower().strip()
    return {k: v for k, v in avail.items() if k.startswith(prefix)}


def time_callable(fn: Callable[..., Any], *args, warmup: int = 10, repeat: int = 50, **kwargs) -> float:
    for _ in range(max(0, int(warmup))):
        _ = fn(*args, **kwargs)
    _sync_if_cuda()
    t0 = time.time()
    for _ in range(max(1, int(repeat))):
        _ = fn(*args, **kwargs)
    _sync_if_cuda()
    return (time.time() - t0) / max(1, int(repeat))


def bench_attn(seq: int, heads: int, d_k: int, dtype: torch.dtype = torch.bfloat16) -> List[Tuple[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 1
    q = torch.randn(B, heads, seq, d_k, device=device, dtype=dtype)
    k = torch.randn(B, heads, seq, d_k, device=device, dtype=dtype)
    v = torch.randn(B, heads, seq, d_k, device=device, dtype=dtype)

    names = [n for n in list_kernels("attn.") if n in ("attn.flash2", "attn.xformers")]
    names += ["attn.torch_sdpa"]  # synthetic to compare with torch SDPA
    results: List[Tuple[str, float]] = []
    for name in names:
        if name == "attn.torch_sdpa":
            fn = lambda q_, k_, v_: torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, is_causal=True)  # noqa: E731
        else:
            try:
                fn_k = kernel_get(name)
            except Exception:
                continue
            def fn(q_, k_, v_):  # noqa: ANN001
                try:
                    return fn_k(q_, k_, v_, is_causal=True)
                except TypeError:
                    return fn_k(q_, k_, v_)

        try:
            dt = time_callable(fn, q, k, v)
            results.append((name, dt))
        except Exception:
            continue

    results.sort(key=lambda x: x[1])
    return results


def bench_rope(seq: int, heads: int, d_k: int, base_theta: float = 1e6, dtype: torch.dtype = torch.bfloat16) -> List[Tuple[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 1
    q = torch.randn(B, heads, seq, d_k, device=device, dtype=dtype)
    k = torch.randn(B, heads, seq, d_k, device=device, dtype=dtype)
    from tensor.positional import build_rope_cache
    cos, sin = build_rope_cache(seq, d_k, device=device, base_theta=base_theta)
    names = [n for n in list_kernels("rope.")]
    if "rope.apply" not in names:
        names.append("rope.apply")
    results: List[Tuple[str, float]] = []
    for name in names:
        try:
            fn_k = kernel_get(name)
        except Exception:
            continue
        def fn(q_, k_):  # noqa: ANN001
            return fn_k(q_, k_, cos, sin)
        try:
            dt = time_callable(fn, q, k)
            results.append((name, dt))
        except Exception:
            continue
    results.sort(key=lambda x: x[1])
    return results


def select_fastest(results: Sequence[Tuple[str, float]]) -> str | None:
    return results[0][0] if results else None


