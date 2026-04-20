from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FLASH3_HOPPER = ROOT / "other_repos" / "flash-attention" / "hopper"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FLASH3_HOPPER) not in sys.path:
    sys.path.insert(0, str(FLASH3_HOPPER))

from runtime.native import runtime_info
from runtime.quant import int8_matmul_qkv

try:
    from flash_attn_interface import flash_attn_func
except Exception as exc:  # pragma: no cover - runtime benchmark helper
    flash_attn_func = None
    FLASH3_IMPORT_ERROR = exc
else:
    FLASH3_IMPORT_ERROR = None


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


def _dequantize_rows(q: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return (q.float() * scale.unsqueeze(-1)).to(dtype=dtype)


def _set_native_env(*, use_wgmma: bool, use_persistent: bool) -> None:
    os.environ.pop("MODEL_STACK_DISABLE_INT8_ATTENTION_OPTIMIZED", None)
    if use_wgmma:
        os.environ["MODEL_STACK_ENABLE_INT8_ATTENTION_WGMMA"] = "1"
        os.environ.pop("MODEL_STACK_DISABLE_INT8_ATTENTION_WGMMA", None)
    else:
        os.environ["MODEL_STACK_DISABLE_INT8_ATTENTION_WGMMA"] = "1"
        os.environ.pop("MODEL_STACK_ENABLE_INT8_ATTENTION_WGMMA", None)
    if use_persistent:
        os.environ["MODEL_STACK_ENABLE_INT8_ATTENTION_PERSISTENT"] = "1"
        os.environ.pop("MODEL_STACK_DISABLE_INT8_ATTENTION_PERSISTENT", None)
    else:
        os.environ["MODEL_STACK_DISABLE_INT8_ATTENTION_PERSISTENT"] = "1"
        os.environ.pop("MODEL_STACK_ENABLE_INT8_ATTENTION_PERSISTENT", None)
    os.environ.pop("MODEL_STACK_ENABLE_INT8_ATTENTION_DECODE_SPECIALIZED", None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark native int8 Hopper attention against the local FlashAttention-3 Hopper implementation."
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--q-seq", type=int, default=128)
    parser.add_argument("--kv-seq", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--native-wgmma", action="store_true", help="Enable the experimental native SM90a WGMMA path.")
    parser.add_argument(
        "--native-persistent",
        action="store_true",
        help="Enable the experimental native persistent CTA scheduler.",
    )
    args = parser.parse_args()

    if FLASH3_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FlashAttention-3 Hopper import failed. Build or install the local Hopper extension first: "
            f"{FLASH3_IMPORT_ERROR}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_int8_attention_vs_flash3.py")

    dtype_name = str(args.dtype).strip().lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype: {args.dtype}")
    dtype = dtype_map[dtype_name]

    device = torch.device("cuda")
    qf = torch.randn(args.batch, args.heads, args.q_seq, args.head_dim, device=device, dtype=dtype)
    kf = torch.randn(args.batch, args.heads, args.kv_seq, args.head_dim, device=device, dtype=dtype)
    vf = torch.randn(args.batch, args.heads, args.kv_seq, args.head_dim, device=device, dtype=dtype)
    q, q_scale = _quantize_rows(qf.float())
    k, k_scale = _quantize_rows(kf.float())
    v, v_scale = _quantize_rows(vf.float())

    q_dq = _dequantize_rows(q, q_scale, dtype=dtype).permute(0, 2, 1, 3).contiguous()
    k_dq = _dequantize_rows(k, k_scale, dtype=dtype).permute(0, 2, 1, 3).contiguous()
    v_dq = _dequantize_rows(v, v_scale, dtype=dtype).permute(0, 2, 1, 3).contiguous()

    def run_native() -> torch.Tensor:
        return int8_matmul_qkv(q, k, v, q_scale, k_scale, v_scale, None, out_dtype=dtype)

    def run_flash3() -> torch.Tensor:
        return flash_attn_func(q_dq, k_dq, v_dq, causal=bool(args.causal))

    _set_native_env(use_wgmma=bool(args.native_wgmma), use_persistent=bool(args.native_persistent))
    out_native = run_native()
    native_ms = _timeit(run_native, warmup=int(args.warmup), iters=int(args.iters))

    out_flash3 = run_flash3()
    flash3_ms = _timeit(run_flash3, warmup=int(args.warmup), iters=int(args.iters))

    out_flash3_native_layout = out_flash3.permute(0, 2, 1, 3).contiguous()
    max_err = float((out_native - out_flash3_native_layout).abs().max().item())
    mean_err = float((out_native - out_flash3_native_layout).abs().mean().item())
    native_vs_flash3 = flash3_ms / native_ms if native_ms > 0 else float("inf")

    info = runtime_info()
    print(f"native_kernel_family={info.get('int8_attention_kernel_family')}")
    print(f"native_scheduler={info.get('int8_attention_scheduler')}")
    print(f"native_wgmma_tile={info.get('int8_attention_wgmma_tile')}")
    print(
        f"shape=batch={args.batch} heads={args.heads} q_seq={args.q_seq} "
        f"kv_seq={args.kv_seq} head_dim={args.head_dim} causal={bool(args.causal)} dtype={dtype_name}"
    )
    print(f"native_wgmma={bool(args.native_wgmma)}")
    print(f"native_persistent={bool(args.native_persistent)}")
    print(f"native_ms={native_ms:.4f}")
    print(f"flash3_ms={flash3_ms:.4f}")
    print(f"native_speedup_vs_flash3={native_vs_flash3:.3f}x")
    print(f"max_abs_err={max_err:.6f}")
    print(f"mean_abs_err={mean_err:.6f}")


if __name__ == "__main__":
    main()
