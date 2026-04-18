from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.quantization import QuantizedLinearBitNet
from runtime.attention_modules import EagerAttention
from specs.config import ModelConfig


def _timeit(fn, *, warmup: int, iters: int, use_cuda_events: bool) -> float:
    for _ in range(warmup):
        fn()
    if use_cuda_events:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / max(iters, 1)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return ((time.perf_counter() - t0) * 1000.0) / max(iters, 1)


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[normalized]


def _to_bitnet_linear(linear: torch.nn.Linear) -> QuantizedLinearBitNet:
    layer = QuantizedLinearBitNet(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
    )
    return layer.from_float(linear)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BitNet attention projections against dense attention.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)
    use_cuda_events = device.type == "cuda"
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    cfg = ModelConfig(
        d_model=int(args.heads) * int(args.head_dim),
        n_heads=int(args.heads),
        n_layers=1,
        d_ff=int(args.heads) * int(args.head_dim) * 4,
        vocab_size=32000,
        dtype=str(args.dtype),
    )
    cfg.n_kv_heads = int(args.kv_heads)
    dense_attn = EagerAttention(cfg).to(device=device, dtype=dtype).eval()
    bitnet_attn = copy.deepcopy(dense_attn).eval()
    bitnet_attn.w_q = _to_bitnet_linear(bitnet_attn.w_q).to(device=device)
    bitnet_attn.w_k = _to_bitnet_linear(bitnet_attn.w_k).to(device=device)
    bitnet_attn.w_v = _to_bitnet_linear(bitnet_attn.w_v).to(device=device)
    bitnet_attn.w_o = _to_bitnet_linear(bitnet_attn.w_o).to(device=device)

    x = torch.randn(args.batch, args.seq, cfg.d_model, device=device, dtype=dtype)
    with torch.no_grad():
        dense_out = dense_attn(x, None, None, None)
        bitnet_out = bitnet_attn(x, None, None, None)
    max_abs_err = float((bitnet_out - dense_out).abs().max().item())
    packed_backend = bitnet_attn._packed_backend(x)

    dense_ms = _timeit(
        lambda: dense_attn(x, None, None, None),
        warmup=args.warmup,
        iters=args.iters,
        use_cuda_events=use_cuda_events,
    )
    bitnet_ms = _timeit(
        lambda: bitnet_attn(x, None, None, None),
        warmup=args.warmup,
        iters=args.iters,
        use_cuda_events=use_cuda_events,
    )
    speedup = dense_ms / bitnet_ms if bitnet_ms > 0 else float("inf")

    print(f"device={device} dtype={dtype} batch={args.batch} seq={args.seq}")
    print(f"heads={args.heads} kv_heads={args.kv_heads} head_dim={args.head_dim}")
    print(f"packed_backend={packed_backend}")
    print(f"dense_ms={dense_ms:.3f}")
    print(f"bitnet_ms={bitnet_ms:.3f}")
    print(f"speedup={speedup:.3f}x")
    print(f"max_abs_err={max_abs_err:.6f}")


if __name__ == "__main__":
    main()
