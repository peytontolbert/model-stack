from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.native import runtime_info
from runtime.quant import int8_matmul_qkv


def _timeit(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return ((time.perf_counter() - start) * 1000.0) / max(iters, 1)


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127.0
    q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.float32)


def _run_case(
    *,
    batch: int,
    heads: int,
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    use_mask: bool,
    opt_in_decode: bool,
) -> None:
    device = torch.device("cuda")
    qf = torch.randn(batch, heads, q_seq, head_dim, device=device, dtype=dtype)
    kf = torch.randn(batch, heads, kv_seq, head_dim, device=device, dtype=dtype)
    vf = torch.randn(batch, heads, kv_seq, head_dim, device=device, dtype=dtype)
    q, q_scale = _quantize_rows(qf.float())
    k, k_scale = _quantize_rows(kf.float())
    v, v_scale = _quantize_rows(vf.float())
    attn_mask = None
    if use_mask:
        attn_mask = torch.zeros(batch, heads, q_seq, kv_seq, device=device, dtype=torch.bool)
        attn_mask[..., kv_seq // 2 :] = True

    def run_once() -> torch.Tensor:
        return int8_matmul_qkv(q, k, v, q_scale, k_scale, v_scale, attn_mask, out_dtype=dtype)

    os.environ.pop("MODEL_STACK_DISABLE_INT8_ATTENTION_OPTIMIZED", None)
    if opt_in_decode:
        os.environ["MODEL_STACK_ENABLE_INT8_ATTENTION_DECODE_SPECIALIZED"] = "1"
    else:
        os.environ.pop("MODEL_STACK_ENABLE_INT8_ATTENTION_DECODE_SPECIALIZED", None)
    out_default = run_once()
    default_ms = _timeit(run_once, warmup=warmup, iters=iters)

    os.environ["MODEL_STACK_DISABLE_INT8_ATTENTION_OPTIMIZED"] = "1"
    os.environ.pop("MODEL_STACK_ENABLE_INT8_ATTENTION_DECODE_SPECIALIZED", None)
    out_generic = run_once()
    generic_ms = _timeit(run_once, warmup=warmup, iters=iters)

    max_err = float((out_default - out_generic).abs().max().item())
    speedup = generic_ms / default_ms if default_ms > 0 else float("inf")
    print(
        f"shape=batch={batch} heads={heads} q_seq={q_seq} kv_seq={kv_seq} head_dim={head_dim} "
        f"mask={use_mask} decode_opt_in={opt_in_decode}"
    )
    print(f"default_ms={default_ms:.4f}")
    print(f"generic_ms={generic_ms:.4f}")
    print(f"speedup={speedup:.3f}x")
    print(f"max_abs_err={max_err:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native int8 attention dispatch against the generic fallback.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--q-seq", type=int, default=128)
    parser.add_argument("--kv-seq", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mask", action="store_true", help="Use a boolean mask to benchmark masked prefill.")
    parser.add_argument(
        "--decode-opt-in",
        action="store_true",
        help="Enable the experimental q_len==1 decode-specialized kernel.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_int8_attention.py")

    dtype_name = str(args.dtype).strip().lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype: {args.dtype}")

    info = runtime_info()
    print(f"kernel_family={info.get('int8_attention_kernel_family')}")
    print(f"specializations={info.get('int8_attention_specializations')}")
    print(f"optimized_min_work_default={info.get('int8_attention_optimized_min_work_default')}")
    print(
        "optimized_small_seq_min_head_dim_default="
        f"{info.get('int8_attention_optimized_small_seq_min_head_dim_default')}"
    )
    print(f"decode_specialized_env={info.get('int8_attention_decode_specialized_env')}")

    _run_case(
        batch=int(args.batch),
        heads=int(args.heads),
        q_seq=int(args.q_seq),
        kv_seq=int(args.kv_seq),
        head_dim=int(args.head_dim),
        dtype=dtype_map[dtype_name],
        warmup=int(args.warmup),
        iters=int(args.iters),
        use_mask=bool(args.mask),
        opt_in_decode=bool(args.decode_opt_in),
    )


if __name__ == "__main__":
    main()
