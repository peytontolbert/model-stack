from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn

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


def _to_bitnet_linear(
    linear: torch.nn.Linear,
    *,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
    spin: bool,
) -> QuantizedLinearBitNet:
    layer = QuantizedLinearBitNet(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
    )
    return layer.from_float(
        linear,
        activation_quant=activation_quant,
        activation_quant_bits=activation_quant_bits,
        activation_quant_method=activation_quant_method,
        activation_quant_percentile=activation_quant_percentile,
        spin=spin,
    )


def _to_dequantized_linear(layer: QuantizedLinearBitNet, *, dtype: torch.dtype, device: torch.device) -> nn.Linear:
    linear = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None, device=device, dtype=dtype)
    with torch.no_grad():
        linear.weight.copy_(layer.runtime_weight(dtype=dtype, device=device))
        if linear.bias is not None:
            bias = layer.runtime_bias(dtype=dtype, device=device)
            assert bias is not None
            linear.bias.copy_(bias)
    return linear


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BitNet attention projections against dense attention.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--activation-quant", type=str, default="dynamic_int8")
    parser.add_argument("--activation-quant-bits", type=int, default=8)
    parser.add_argument("--activation-quant-method", type=str, default="absmax")
    parser.add_argument("--activation-quant-percentile", type=float, default=0.999)
    parser.add_argument("--packed-backend", choices=("disabled", "auto"), default="auto")
    parser.add_argument("--spin", action="store_true")
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
    quant_kwargs = {
        "activation_quant": str(args.activation_quant),
        "activation_quant_bits": int(args.activation_quant_bits),
        "activation_quant_method": str(args.activation_quant_method),
        "activation_quant_percentile": float(args.activation_quant_percentile),
        "spin": bool(args.spin),
    }
    bitnet_attn.w_q = _to_bitnet_linear(bitnet_attn.w_q, **quant_kwargs).to(device=device)
    bitnet_attn.w_k = _to_bitnet_linear(bitnet_attn.w_k, **quant_kwargs).to(device=device)
    bitnet_attn.w_v = _to_bitnet_linear(bitnet_attn.w_v, **quant_kwargs).to(device=device)
    bitnet_attn.w_o = _to_bitnet_linear(bitnet_attn.w_o, **quant_kwargs).to(device=device)
    if str(args.packed_backend) == "disabled":
        bitnet_attn._packed_backend = lambda _x: None  # type: ignore[method-assign]
    dequant_attn = copy.deepcopy(bitnet_attn).eval()
    dequant_attn.w_q = _to_dequantized_linear(bitnet_attn.w_q, dtype=dtype, device=device)
    dequant_attn.w_k = _to_dequantized_linear(bitnet_attn.w_k, dtype=dtype, device=device)
    dequant_attn.w_v = _to_dequantized_linear(bitnet_attn.w_v, dtype=dtype, device=device)
    dequant_attn.w_o = _to_dequantized_linear(bitnet_attn.w_o, dtype=dtype, device=device)
    dequant_attn.eval()

    x = torch.randn(args.batch, args.seq, cfg.d_model, device=device, dtype=dtype)
    with torch.no_grad():
        dense_out = dense_attn(x, None, None, None)
        dequant_out = dequant_attn(x, None, None, None)
        bitnet_out = bitnet_attn(x, None, None, None)
    max_abs_err_vs_dense = float((bitnet_out - dense_out).abs().max().item())
    max_abs_err_vs_bitnet_ref = float((bitnet_out - dequant_out).abs().max().item())
    packed_backend = bitnet_attn._packed_backend(x)

    dense_ms = _timeit(
        lambda: dense_attn(x, None, None, None),
        warmup=args.warmup,
        iters=args.iters,
        use_cuda_events=use_cuda_events,
    )
    dequant_ms = _timeit(
        lambda: dequant_attn(x, None, None, None),
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
    speedup = dequant_ms / bitnet_ms if bitnet_ms > 0 else float("inf")

    print(f"device={device} dtype={dtype} batch={args.batch} seq={args.seq}")
    print(f"heads={args.heads} kv_heads={args.kv_heads} head_dim={args.head_dim}")
    print(
        "activation_quant="
        f"{args.activation_quant} bits={args.activation_quant_bits} "
        f"method={args.activation_quant_method} percentile={args.activation_quant_percentile} "
        f"spin={bool(args.spin)}"
    )
    print(f"packed_backend_mode={args.packed_backend}")
    print(f"packed_backend={packed_backend}")
    print(f"dense_original_ms={dense_ms:.3f}")
    print(f"dense_bitnet_ref_ms={dequant_ms:.3f}")
    print(f"bitnet_ms={bitnet_ms:.3f}")
    print(f"speedup_vs_bitnet_ref={speedup:.3f}x")
    print(f"max_abs_err_vs_dense={max_abs_err_vs_dense:.6f}")
    print(f"max_abs_err_vs_bitnet_ref={max_abs_err_vs_bitnet_ref:.6f}")


if __name__ == "__main__":
    main()
