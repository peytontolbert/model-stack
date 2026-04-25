from __future__ import annotations

import argparse
import os
import sys
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model-stack SM80 causal prefill attention across doubling sequence lengths.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--min-seq", type=int, default=64)
    parser.add_argument("--max-seq", type=int, default=131072)
    parser.add_argument("--seqs", default=None, help="Comma-separated explicit sequence lengths to benchmark.")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--native-kernel", default="64x64_rf")
    parser.add_argument("--enable-flash-prefill", action="store_true")
    parser.add_argument("--disable-flash-prefill", action="store_true")
    parser.add_argument("--flash-min-seq", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _dtype_from_arg(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def _parse_seq_list(value: str | None, min_seq: int, max_seq: int) -> list[int]:
    if value:
        seqs: list[int] = []
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            seq = int(token)
            if seq <= 0:
                raise ValueError("explicit sequence lengths must be positive")
            seqs.append(seq)
        if not seqs:
            raise ValueError("--seqs did not contain any sequence lengths")
        return seqs
    if min_seq <= 0 or max_seq < min_seq:
        raise ValueError("sequence length bounds must be positive and ordered")
    if min_seq & (min_seq - 1) or max_seq & (max_seq - 1):
        raise ValueError("min-seq and max-seq must be powers of two unless --seqs is used")
    seqs = []
    seq_len = min_seq
    while seq_len <= max_seq:
        seqs.append(seq_len)
        seq_len *= 2
    return seqs


def _reps_for_seq(seq_len: int) -> int:
    if seq_len <= 2048:
        return 30
    if seq_len <= 8192:
        return 15
    if seq_len <= 32768:
        return 6
    if seq_len <= 65536:
        return 3
    return 2


def _warmups_for_seq(seq_len: int) -> int:
    if seq_len <= 2048:
        return 10
    if seq_len <= 8192:
        return 6
    if seq_len <= 32768:
        return 4
    return 2


def _time_once(fn) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_sm80_attention_scaling.py")
    seqs = _parse_seq_list(args.seqs, args.min_seq, args.max_seq)

    os.environ["MODEL_STACK_DISABLE_ATTENTION_PREFILL_PYTORCH_MEMEFF"] = "1"
    os.environ["MODEL_STACK_DISABLE_ATTENTION_PREFILL_CUTLASS"] = "1"
    os.environ["MODEL_STACK_DISABLE_ATTENTION_PREFILL_TENSORCORE"] = "1"
    if args.disable_flash_prefill:
        os.environ["MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_FLASH"] = "1"
    else:
        os.environ.pop("MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_FLASH", None)
    if args.enable_flash_prefill:
        os.environ["MODEL_STACK_ENABLE_ATTENTION_PREFILL_SM80_FLASH"] = "1"
    else:
        os.environ.pop("MODEL_STACK_ENABLE_ATTENTION_PREFILL_SM80_FLASH", None)
    if args.flash_min_seq is not None:
        os.environ["MODEL_STACK_SM80_FLASH_PREFILL_MIN_SEQ"] = str(args.flash_min_seq)
    os.environ["MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_INFERENCE"] = "0"
    os.environ["MODEL_STACK_SM80_INFERENCE_PREFILL_KERNEL"] = args.native_kernel

    from runtime import ops as runtime_ops
    from runtime.native import runtime_info

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = _dtype_from_arg(args.dtype)
    scale = args.head_dim ** -0.5
    module = runtime_ops.native_module()
    if module is None or not hasattr(module, "attention_forward"):
        raise RuntimeError("native attention_forward is unavailable")

    print(
        {
            "device": str(device),
            "batch": args.batch,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "native_kernel": args.native_kernel,
            "seqs": seqs,
            "runtime_info": {
                key: runtime_info().get(key)
                for key in (
                    "attention_sm80_inference_prefill_compiled",
                    "attention_pytorch_memeff_prefill_compiled",
                    "attention_cutlass_prefill_compiled",
                    "attention_sm80_flash_prefill_compiled",
                )
            },
        }
    )

    for seq_len in seqs:
        q = torch.randn(args.batch, args.heads, seq_len, args.head_dim, device=device, dtype=dtype)
        k = torch.randn(args.batch, args.heads, seq_len, args.head_dim, device=device, dtype=dtype)
        v = torch.randn(args.batch, args.heads, seq_len, args.head_dim, device=device, dtype=dtype)

        plan = runtime_ops.attention_plan_info(q, k, v, None, is_causal=True)
        warmups = _warmups_for_seq(seq_len)
        reps = _reps_for_seq(seq_len)

        for _ in range(warmups):
            native_out = module.attention_forward(q, k, v, None, True, scale)
            torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        torch.cuda.synchronize()

        native_times: list[float] = []
        torch_times: list[float] = []
        for i in range(reps):
            if i % 2 == 0:
                t, native_out = _time_once(lambda: module.attention_forward(q, k, v, None, True, scale))
                native_times.append(t)
                t, torch_out = _time_once(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale))
                torch_times.append(t)
            else:
                t, torch_out = _time_once(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale))
                torch_times.append(t)
                t, native_out = _time_once(lambda: module.attention_forward(q, k, v, None, True, scale))
                native_times.append(t)

        print(
            {
                "seq_len": seq_len,
                "reps": reps,
                "native_ms": statistics.median(native_times),
                "torch_ms": statistics.median(torch_times),
                "speedup": statistics.median(torch_times) / statistics.median(native_times),
                "max_abs_diff": (native_out.float() - torch_out.float()).abs().max().item(),
                "plan_kernel": plan.get("kernel"),
                "plan_row_threads": plan.get("row_reduce_threads"),
                "split_kv_splits": plan.get("split_kv_splits"),
                "split_kv_effective_sms": plan.get("split_kv_effective_sms"),
                "flash_selected": plan.get("sm80_flash_prefill_selected"),
                "flash_min_seq": plan.get("sm80_flash_prefill_min_seq"),
            }
        )

        del q, k, v, native_out, torch_out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
