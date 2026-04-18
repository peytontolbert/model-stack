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
from runtime.causal import CausalLM
from runtime.native import create_native_model_session
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
    return QuantizedLinearBitNet(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
    ).from_float(linear)


def _convert_model_to_bitnet(model: CausalLM) -> None:
    for block in model.blocks:
        attn_device = block.attn.w_q.weight.device
        for name in ("w_q", "w_k", "w_v", "w_o"):
            setattr(block.attn, name, _to_bitnet_linear(getattr(block.attn, name)).to(device=attn_device))
        mlp_device = block.mlp.w_in.weight.device
        for name in ("w_in", "w_out"):
            setattr(block.mlp, name, _to_bitnet_linear(getattr(block.mlp, name)).to(device=mlp_device))
    model.lm_head = _to_bitnet_linear(model.lm_head).to(device=model.embed.weight.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BitNet decode using the native model session when available.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)
    use_cuda_events = device.type == "cuda"
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    d_model = int(args.heads) * int(args.head_dim)
    cfg = ModelConfig(
        d_model=d_model,
        n_heads=int(args.heads),
        n_layers=int(args.layers),
        d_ff=d_model * int(args.ff_mult),
        vocab_size=int(args.vocab_size),
        dtype=str(args.dtype),
    )
    dense_model = CausalLM(cfg, block_variant="llama", tie_weights=False).to(device=device, dtype=dtype).eval()
    bitnet_model = copy.deepcopy(dense_model).eval()
    _convert_model_to_bitnet(bitnet_model)
    seq = torch.randint(0, cfg.vocab_size, (int(args.batch), int(args.prompt)), device=device)

    native_session = create_native_model_session(bitnet_model, seq)
    native_executor_kind = getattr(native_session, "native_executor_kind", None) if native_session is not None else None

    with torch.no_grad():
        dense_logits = dense_model(seq, return_dict=False)[:, -1, :]
        if native_session is not None:
            bitnet_logits = native_session.full_next_logits()
        else:
            bitnet_logits = bitnet_model(seq, return_dict=False)[:, -1, :]
    max_abs_err = float((bitnet_logits - dense_logits).abs().max().item())

    dense_ms = _timeit(
        lambda: dense_model(seq, return_dict=False)[:, -1, :],
        warmup=args.warmup,
        iters=args.iters,
        use_cuda_events=use_cuda_events,
    )
    if native_session is not None:
        bitnet_fn = native_session.full_next_logits
    else:
        bitnet_fn = lambda: bitnet_model(seq, return_dict=False)[:, -1, :]
    bitnet_ms = _timeit(
        bitnet_fn,
        warmup=args.warmup,
        iters=args.iters,
        use_cuda_events=use_cuda_events,
    )
    speedup = dense_ms / bitnet_ms if bitnet_ms > 0 else float("inf")

    print(f"device={device} dtype={dtype} batch={args.batch} prompt={args.prompt}")
    print(f"layers={args.layers} heads={args.heads} head_dim={args.head_dim}")
    print(f"native_executor_kind={native_executor_kind}")
    print(f"dense_ms={dense_ms:.3f}")
    print(f"bitnet_ms={bitnet_ms:.3f}")
    print(f"speedup={speedup:.3f}x")
    print(f"max_abs_err={max_abs_err:.6f}")


if __name__ == "__main__":
    main()
