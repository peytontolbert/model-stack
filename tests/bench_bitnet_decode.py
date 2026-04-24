from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.quantization import QuantizedLinearBitNet
from runtime.causal import CausalLM
from runtime.generation import RuntimeGenerationSession
from runtime.kv_cache import clone_kv_cache_rows
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


def _time_session_calls(
    sessions,
    *,
    method_name: str,
    warmup: int,
    iters: int,
    use_cuda_events: bool,
) -> float:
    total = warmup + iters
    if len(sessions) < total:
        raise ValueError(f"Need at least {total} prepared sessions, got {len(sessions)}")
    if use_cuda_events:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for idx in range(warmup):
            getattr(sessions[idx], method_name)()
        torch.cuda.synchronize()
        start.record()
        for idx in range(warmup, total):
            getattr(sessions[idx], method_name)()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / max(iters, 1)
    for idx in range(warmup):
        getattr(sessions[idx], method_name)()
    t0 = time.perf_counter()
    for idx in range(warmup, total):
        getattr(sessions[idx], method_name)()
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


def _build_generation_session(
    model,
    seq: torch.Tensor,
    *,
    enabled: bool,
    cache_pagesize: int,
):
    session = RuntimeGenerationSession.from_model(
        model,
        seq,
        cache_pagesize=int(cache_pagesize),
        cache_backend="native-paged",
    )
    executor_kind = session.native_executor_kind
    if not bool(enabled):
        session._native_session = None
        executor_kind = "python"
    return session, executor_kind


def _clone_generation_session(session: RuntimeGenerationSession, *, enabled: bool) -> RuntimeGenerationSession:
    seq = session.seq.clone()
    attention_mask = None if session.attention_mask is None else session.attention_mask.clone()
    cache = None
    if session.cache is not None:
        row_ids = torch.arange(session.batch_size, device=seq.device, dtype=torch.long)
        cache = clone_kv_cache_rows(session.cache, row_ids)
    cloned = RuntimeGenerationSession(
        model=session.model,
        seq=seq,
        attention_mask=attention_mask,
        cache=cache,
        trace=session.trace,
    )
    if not bool(enabled):
        cloned._native_session = None
    return cloned


def _prepare_decode_base_session(
    model,
    seq: torch.Tensor,
    *,
    enabled: bool,
    cache_pagesize: int,
    next_id: torch.Tensor | None = None,
):
    session, executor_kind = _build_generation_session(
        model,
        seq,
        enabled=enabled,
        cache_pagesize=cache_pagesize,
    )
    prefill = session.prefill_next_logits()
    if prefill is None:
        raise RuntimeError("decode benchmark requires a KV cache-backed generation session")
    append_id = torch.argmax(prefill, dim=-1, keepdim=True) if next_id is None else next_id.to(device=prefill.device, dtype=torch.long)
    session.append(append_id)
    return session, executor_kind, prefill


def _prepare_decode_sessions(
    base_session: RuntimeGenerationSession,
    *,
    enabled: bool,
    count: int,
):
    return [_clone_generation_session(base_session, enabled=enabled) for _ in range(int(count))]


def _prepare_native_decode_sessions(
    sessions: list[RuntimeGenerationSession],
) -> list[object]:
    native_sessions: list[object] = []
    for session in sessions:
        native = getattr(session, "_native_session", None)
        if native is None:
            raise RuntimeError("native-direct decode driver requires a live native session on every prepared session")
        native_sessions.append(native)
    return native_sessions


def _to_bitnet_linear(
    linear: torch.nn.Linear,
    *,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
    spin: bool,
) -> QuantizedLinearBitNet:
    return QuantizedLinearBitNet(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
    ).from_float(
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


def _convert_model_to_bitnet(
    model: CausalLM,
    *,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
    spin: bool,
) -> None:
    for block in model.blocks:
        attn_device = block.attn.w_q.weight.device
        for name in ("w_q", "w_k", "w_v", "w_o"):
            setattr(
                block.attn,
                name,
                _to_bitnet_linear(
                    getattr(block.attn, name),
                    activation_quant=activation_quant,
                    activation_quant_bits=activation_quant_bits,
                    activation_quant_method=activation_quant_method,
                    activation_quant_percentile=activation_quant_percentile,
                    spin=spin,
                ).to(device=attn_device),
            )
        mlp_device = block.mlp.w_in.weight.device
        for name in ("w_in", "w_out"):
            setattr(
                block.mlp,
                name,
                _to_bitnet_linear(
                    getattr(block.mlp, name),
                    activation_quant=activation_quant,
                    activation_quant_bits=activation_quant_bits,
                    activation_quant_method=activation_quant_method,
                    activation_quant_percentile=activation_quant_percentile,
                    spin=spin,
                ).to(device=mlp_device),
            )
    model.lm_head = _to_bitnet_linear(
        model.lm_head,
        activation_quant=activation_quant,
        activation_quant_bits=activation_quant_bits,
        activation_quant_method=activation_quant_method,
        activation_quant_percentile=activation_quant_percentile,
        spin=spin,
    ).to(device=model.embed.weight.device)


def _convert_model_to_dequantized_bitnet(model: CausalLM, *, dtype: torch.dtype, device: torch.device) -> None:
    for block in model.blocks:
        for name in ("w_q", "w_k", "w_v", "w_o"):
            layer = getattr(block.attn, name)
            if isinstance(layer, QuantizedLinearBitNet):
                setattr(block.attn, name, _to_dequantized_linear(layer, dtype=dtype, device=device))
        for name in ("w_in", "w_out"):
            layer = getattr(block.mlp, name)
            if isinstance(layer, QuantizedLinearBitNet):
                setattr(block.mlp, name, _to_dequantized_linear(layer, dtype=dtype, device=device))
    if isinstance(model.lm_head, QuantizedLinearBitNet):
        model.lm_head = _to_dequantized_linear(model.lm_head, dtype=dtype, device=device)
    model.eval()


def _configure_attention_backend(model: CausalLM, *, packed_backend: str) -> None:
    if str(packed_backend) != "disabled":
        return
    for block in model.blocks:
        block.attn._packed_backend = lambda _x: None  # type: ignore[method-assign]


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
    parser.add_argument("--activation-quant", type=str, default="dynamic_int8")
    parser.add_argument("--activation-quant-bits", type=int, default=8)
    parser.add_argument("--activation-quant-method", type=str, default="absmax")
    parser.add_argument("--activation-quant-percentile", type=float, default=0.999)
    parser.add_argument("--packed-backend", choices=("disabled", "auto"), default="auto")
    parser.add_argument("--native-session", dest="native_session", action="store_true")
    parser.add_argument("--no-native-session", dest="native_session", action="store_false")
    parser.add_argument("--spin", action="store_true")
    parser.add_argument(
        "--driver",
        choices=("generation_session", "native_direct"),
        default="generation_session",
        help="Benchmark through the RuntimeGenerationSession wrapper or directly through NativeModelSession.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the timed decode benchmark this many times and report summary stats. "
             "The legacy *_ms lines report the median when repeats > 1.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON object instead of the human-readable text report.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(native_session=True)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)
    use_cuda_events = device.type == "cuda"
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.driver == "native_direct" and not bool(args.native_session):
        raise RuntimeError("--driver native_direct requires --native-session")

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
    _convert_model_to_bitnet(
        bitnet_model,
        activation_quant=str(args.activation_quant),
        activation_quant_bits=int(args.activation_quant_bits),
        activation_quant_method=str(args.activation_quant_method),
        activation_quant_percentile=float(args.activation_quant_percentile),
        spin=bool(args.spin),
    )
    dequant_model = copy.deepcopy(bitnet_model).eval()
    _convert_model_to_dequantized_bitnet(dequant_model, dtype=dtype, device=device)
    _configure_attention_backend(dense_model, packed_backend=str(args.packed_backend))
    _configure_attention_backend(bitnet_model, packed_backend=str(args.packed_backend))
    _configure_attention_backend(dequant_model, packed_backend=str(args.packed_backend))
    seq = torch.randint(0, cfg.vocab_size, (int(args.batch), int(args.prompt)), device=device)

    prepared_count = int(args.warmup) + int(args.iters)
    dense_base, dense_executor_kind, dense_prefill = _prepare_decode_base_session(
        dense_model,
        seq,
        enabled=bool(args.native_session),
        cache_pagesize=max(int(args.prompt), 1),
    )
    next_id = torch.argmax(dense_prefill, dim=-1, keepdim=True)
    bitnet_base, native_executor_kind, _ = _prepare_decode_base_session(
        bitnet_model,
        seq,
        enabled=bool(args.native_session),
        cache_pagesize=max(int(args.prompt), 1),
        next_id=next_id,
    )
    dequant_base, dequant_executor_kind, _ = _prepare_decode_base_session(
        dequant_model,
        seq,
        enabled=bool(args.native_session),
        cache_pagesize=max(int(args.prompt), 1),
        next_id=next_id,
    )

    with torch.no_grad():
        dense_logits = dense_base.decode_next_logits()
        dequant_logits = dequant_base.decode_next_logits()
        bitnet_logits = bitnet_base.decode_next_logits()
    if dense_logits is None or dequant_logits is None or bitnet_logits is None:
        raise RuntimeError("decode benchmark expected non-null decode logits")
    max_abs_err_vs_dense = float((bitnet_logits - dense_logits).abs().max().item())
    max_abs_err_vs_bitnet_ref = float((bitnet_logits - dequant_logits).abs().max().item())

    repeat_count = max(int(args.repeats), 1)
    dense_runs: list[float] = []
    dequant_runs: list[float] = []
    bitnet_runs: list[float] = []

    for _ in range(repeat_count):
        dense_sessions = _prepare_decode_sessions(dense_base, enabled=bool(args.native_session), count=prepared_count)
        bitnet_sessions = _prepare_decode_sessions(bitnet_base, enabled=bool(args.native_session), count=prepared_count)
        dequant_sessions = _prepare_decode_sessions(dequant_base, enabled=bool(args.native_session), count=prepared_count)
        if args.driver == "native_direct":
            dense_bench_targets = _prepare_native_decode_sessions(dense_sessions)
            bitnet_bench_targets = _prepare_native_decode_sessions(bitnet_sessions)
            dequant_bench_targets = _prepare_native_decode_sessions(dequant_sessions)
        else:
            dense_bench_targets = dense_sessions
            bitnet_bench_targets = bitnet_sessions
            dequant_bench_targets = dequant_sessions

        dense_runs.append(
            _time_session_calls(
                dense_bench_targets,
                method_name="decode_next_logits",
                warmup=args.warmup,
                iters=args.iters,
                use_cuda_events=use_cuda_events,
            )
        )
        dequant_runs.append(
            _time_session_calls(
                dequant_bench_targets,
                method_name="decode_next_logits",
                warmup=args.warmup,
                iters=args.iters,
                use_cuda_events=use_cuda_events,
            )
        )
        bitnet_runs.append(
            _time_session_calls(
                bitnet_bench_targets,
                method_name="decode_next_logits",
                warmup=args.warmup,
                iters=args.iters,
                use_cuda_events=use_cuda_events,
            )
        )

    dense_ms = statistics.median(dense_runs)
    dequant_ms = statistics.median(dequant_runs)
    bitnet_ms = statistics.median(bitnet_runs)
    speedup = dequant_ms / bitnet_ms if bitnet_ms > 0 else float("inf")

    result = {
        "device": str(device),
        "dtype": str(dtype),
        "batch": int(args.batch),
        "prompt": int(args.prompt),
        "layers": int(args.layers),
        "heads": int(args.heads),
        "head_dim": int(args.head_dim),
        "activation_quant": str(args.activation_quant),
        "activation_quant_bits": int(args.activation_quant_bits),
        "activation_quant_method": str(args.activation_quant_method),
        "activation_quant_percentile": float(args.activation_quant_percentile),
        "spin": bool(args.spin),
        "bench_stage": "decode_cached",
        "bench_driver": str(args.driver),
        "repeat_count": int(repeat_count),
        "packed_backend_mode": str(args.packed_backend),
        "native_session_enabled": bool(args.native_session),
        "native_dense_executor_kind": str(dense_executor_kind),
        "native_executor_kind": str(native_executor_kind),
        "native_dequant_executor_kind": str(dequant_executor_kind),
        "dense_original_ms": float(dense_ms),
        "dense_bitnet_ref_ms": float(dequant_ms),
        "bitnet_ms": float(bitnet_ms),
        "speedup_vs_bitnet_ref": float(speedup),
        "max_abs_err_vs_dense": float(max_abs_err_vs_dense),
        "max_abs_err_vs_bitnet_ref": float(max_abs_err_vs_bitnet_ref),
        "dense_original_runs_ms": [float(v) for v in dense_runs],
        "dense_bitnet_ref_runs_ms": [float(v) for v in dequant_runs],
        "bitnet_runs_ms": [float(v) for v in bitnet_runs],
        "dense_original_mean_ms": float(statistics.mean(dense_runs)),
        "dense_bitnet_ref_mean_ms": float(statistics.mean(dequant_runs)),
        "bitnet_mean_ms": float(statistics.mean(bitnet_runs)),
    }

    if bool(args.json):
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"device={device} dtype={dtype} batch={args.batch} prompt={args.prompt}")
    print(f"layers={args.layers} heads={args.heads} head_dim={args.head_dim}")
    print(
        "activation_quant="
        f"{args.activation_quant} bits={args.activation_quant_bits} "
        f"method={args.activation_quant_method} percentile={args.activation_quant_percentile} "
        f"spin={bool(args.spin)}"
    )
    print("bench_stage=decode_cached")
    print(f"bench_driver={args.driver}")
    print(f"repeat_count={repeat_count}")
    print(f"packed_backend_mode={args.packed_backend}")
    print(f"native_session_enabled={bool(args.native_session)}")
    print(f"native_dense_executor_kind={dense_executor_kind}")
    print(f"native_executor_kind={native_executor_kind}")
    print(f"native_dequant_executor_kind={dequant_executor_kind}")
    print(f"dense_original_ms={dense_ms:.3f}")
    print(f"dense_bitnet_ref_ms={dequant_ms:.3f}")
    print(f"bitnet_ms={bitnet_ms:.3f}")
    print(f"speedup_vs_bitnet_ref={speedup:.3f}x")
    print(f"max_abs_err_vs_dense={max_abs_err_vs_dense:.6f}")
    print(f"max_abs_err_vs_bitnet_ref={max_abs_err_vs_bitnet_ref:.6f}")
    if repeat_count > 1:
        dense_runs_str = ",".join(f"{v:.3f}" for v in dense_runs)
        dequant_runs_str = ",".join(f"{v:.3f}" for v in dequant_runs)
        bitnet_runs_str = ",".join(f"{v:.3f}" for v in bitnet_runs)
        print(f"dense_original_runs_ms={dense_runs_str}")
        print(f"dense_bitnet_ref_runs_ms={dequant_runs_str}")
        print(f"bitnet_runs_ms={bitnet_runs_str}")
        print(f"dense_original_mean_ms={statistics.mean(dense_runs):.3f}")
        print(f"dense_bitnet_ref_mean_ms={statistics.mean(dequant_runs):.3f}")
        print(f"bitnet_mean_ms={statistics.mean(bitnet_runs):.3f}")


if __name__ == "__main__":
    main()
