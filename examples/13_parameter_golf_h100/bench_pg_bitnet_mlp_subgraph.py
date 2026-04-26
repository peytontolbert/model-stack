from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.quantization import QuantizedLinearBitNet
from runtime.native import has_native_op, native_module
from runtime.ops import mlp_module as runtime_mlp_module


@dataclass(frozen=True)
class MlpPreset:
    name: str
    model_dim: int
    hidden_features: int
    activation: str
    gated: bool

    @property
    def in_projection_features(self) -> int:
        return self.hidden_features * 2 if self.gated else self.hidden_features


PRESETS: dict[str, MlpPreset] = {
    "runtime_row_1024x7_relu2_mlp3": MlpPreset(
        name="runtime_row_1024x7_relu2_mlp3",
        model_dim=1024,
        hidden_features=3072,
        activation="relu2",
        gated=False,
    ),
    "runtime_row_1024_swiglu_mlp2": MlpPreset(
        name="runtime_row_1024_swiglu_mlp2",
        model_dim=1024,
        hidden_features=2048,
        activation="swiglu",
        gated=True,
    ),
    "runtime_row_512_swiglu_mlp2": MlpPreset(
        name="runtime_row_512_swiglu_mlp2",
        model_dim=512,
        hidden_features=1024,
        activation="swiglu",
        gated=True,
    ),
}


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[key]


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _consume(value: torch.Tensor, sink: torch.Tensor | None) -> None:
    if sink is not None:
        sink.add_(value.reshape(-1)[0].float())


def _time_once(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
    consume_output: bool,
) -> float:
    sink = torch.zeros((), device=device, dtype=torch.float32) if consume_output else None
    for _ in range(warmup):
        _consume(fn(), sink)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _consume(fn(), sink)
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / max(iters, 1))
    t0 = time.perf_counter()
    for _ in range(iters):
        _consume(fn(), sink)
    return float((time.perf_counter() - t0) * 1000.0 / max(iters, 1))


def _time_median(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    device: torch.device,
    consume_output: bool,
) -> float:
    values = [
        _time_once(fn, warmup=warmup, iters=iters, device=device, consume_output=consume_output)
        for _ in range(max(1, repeats))
    ]
    return float(statistics.median(values))


def _apply_mlp_activation(hidden: torch.Tensor, *, activation: str, gated: bool) -> torch.Tensor:
    act = str(activation).lower()
    if gated:
        a, b = hidden.chunk(2, dim=-1)
        if act in {"geglu", "gelu"}:
            return F.gelu(a) * b
        if act in {"reglu", "relu"}:
            return F.relu(a) * b
        if act in {"relu2", "squared_relu", "squared-relu"}:
            y = F.relu(a)
            return (y * y) * b
        if act in {
            "leaky_relu_0p5_squared",
            "leaky-relu-0p5-squared",
            "leaky_relu_0.5_squared",
            "leaky-relu-0.5-squared",
        }:
            y = F.leaky_relu(a, negative_slope=0.5)
            return (y * y) * b
        return F.silu(a) * b
    if act in {"silu", "swish", "swiglu", "gated-silu"}:
        return F.silu(hidden)
    if act in {"relu"}:
        return F.relu(hidden)
    if act in {"relu2", "squared_relu", "squared-relu"}:
        y = F.relu(hidden)
        return y * y
    if act in {
        "leaky_relu_0p5_squared",
        "leaky-relu-0p5-squared",
        "leaky_relu_0.5_squared",
        "leaky-relu-0.5-squared",
    }:
        y = F.leaky_relu(hidden, negative_slope=0.5)
        return y * y
    return F.gelu(hidden)


def _build_bitnet_linear(
    in_features: int,
    out_features: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
) -> tuple[torch.nn.Linear, QuantizedLinearBitNet]:
    linear = torch.nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.125)
        if linear.bias is not None:
            linear.bias.normal_(mean=0.0, std=0.01)
    bitnet = QuantizedLinearBitNet(in_features, out_features, bias=bias).from_float(
        linear,
        activation_quant=activation_quant,
        activation_quant_bits=activation_quant_bits,
        activation_quant_method=activation_quant_method,
        activation_quant_percentile=activation_quant_percentile,
    )
    return linear, bitnet.to(device=device)


def _prepare_backend_weight(bitnet: QuantizedLinearBitNet, *, activation_quant: str, device: torch.device) -> None:
    mode = str(activation_quant).strip().lower()
    if mode in {"", "none", "off"}:
        bitnet._compute_backend_weight(device=device)
        bitnet._decode_backend_weight(device=device)
    elif mode in {"dynamic_int8", "static_int8"}:
        bitnet._int8_backend_weight(device=device)


def _bench_one(
    preset: MlpPreset,
    rows: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
    warmup: int,
    iters: int,
    repeats: int,
    consume_output: bool,
) -> dict[str, object]:
    x = torch.randn(rows, preset.model_dim, device=device, dtype=dtype)
    _linear_in, bitnet_in = _build_bitnet_linear(
        preset.model_dim,
        preset.in_projection_features,
        dtype=dtype,
        device=device,
        bias=bias,
        activation_quant=activation_quant,
        activation_quant_bits=activation_quant_bits,
        activation_quant_method=activation_quant_method,
        activation_quant_percentile=activation_quant_percentile,
    )
    _linear_out, bitnet_out = _build_bitnet_linear(
        preset.hidden_features,
        preset.model_dim,
        dtype=dtype,
        device=device,
        bias=bias,
        activation_quant=activation_quant,
        activation_quant_bits=activation_quant_bits,
        activation_quant_method=activation_quant_method,
        activation_quant_percentile=activation_quant_percentile,
    )

    with torch.inference_mode():
        _prepare_backend_weight(bitnet_in, activation_quant=activation_quant, device=device)
        _prepare_backend_weight(bitnet_out, activation_quant=activation_quant, device=device)
        dense_in_weight = bitnet_in.runtime_weight(dtype=dtype, device=device).contiguous()
        dense_in_bias = bitnet_in.runtime_bias(dtype=dtype, device=device)
        dense_out_weight = bitnet_out.runtime_weight(dtype=dtype, device=device).contiguous()
        dense_out_bias = bitnet_out.runtime_bias(dtype=dtype, device=device)
        module = native_module()

        def dense_dequant_mlp() -> torch.Tensor:
            hidden = F.linear(x, dense_in_weight, dense_in_bias)
            hidden = _apply_mlp_activation(hidden, activation=preset.activation, gated=preset.gated)
            return F.linear(hidden, dense_out_weight, dense_out_bias)

        def model_stack_mlp_module() -> torch.Tensor:
            return runtime_mlp_module(
                x,
                bitnet_in,
                bitnet_out,
                activation=preset.activation,
                gated=preset.gated,
            )

        def native_linear_module_pair() -> torch.Tensor:
            assert module is not None
            hidden = module.linear_module_forward(x, bitnet_in, "auto")
            hidden = _apply_mlp_activation(hidden, activation=preset.activation, gated=preset.gated)
            return module.linear_module_forward(hidden, bitnet_out, "auto")

        dense_out = dense_dequant_mlp()
        auto_out = model_stack_mlp_module()
        native_pair_out = (
            native_linear_module_pair()
            if module is not None and hasattr(module, "linear_module_forward")
            else None
        )
        timer_kwargs = {
            "warmup": warmup,
            "iters": iters,
            "repeats": repeats,
            "device": device,
            "consume_output": consume_output,
        }
        dense_ms = _time_median(dense_dequant_mlp, **timer_kwargs)
        auto_ms = _time_median(model_stack_mlp_module, **timer_kwargs)
        native_pair_ms = (
            _time_median(native_linear_module_pair, **timer_kwargs)
            if module is not None and hasattr(module, "linear_module_forward")
            else None
        )

    result: dict[str, object] = {
        "subgraph": "mlp_pair",
        "preset": preset.name,
        "rows": int(rows),
        "model_dim": int(preset.model_dim),
        "hidden_features": int(preset.hidden_features),
        "in_projection_features": int(preset.in_projection_features),
        "activation": preset.activation,
        "gated": bool(preset.gated),
        "activation_quant": str(activation_quant),
        "dense_bitnet_dequant_mlp_ms": dense_ms,
        "model_stack_mlp_module_ms": auto_ms,
        "model_stack_mlp_module_speedup_vs_dequant": dense_ms / auto_ms if auto_ms > 0.0 else math.inf,
        "model_stack_mlp_module_max_abs_err_vs_dequant": float((auto_out - dense_out).abs().max().item()),
    }
    if native_pair_ms is not None and native_pair_out is not None:
        result["native_linear_module_pair_ms"] = native_pair_ms
        result["native_linear_module_pair_speedup_vs_dequant"] = (
            dense_ms / native_pair_ms if native_pair_ms > 0.0 else math.inf
        )
        result["native_linear_module_pair_max_abs_err_vs_dequant"] = float(
            (native_pair_out - dense_out).abs().max().item()
        )
    return result


def _set_env_variant(name: str, value: str | None) -> str | None:
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    return previous


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _env_variants(include_cutlass: bool) -> list[tuple[str, dict[str, str]]]:
    variants = [("current", {})]
    if include_cutlass:
        variants.append(("int8_cutlass_fused", {"MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED": "1"}))
    return variants


def _enforce_gate(
    results: list[dict[str, object]],
    *,
    min_cutlass_speedup_vs_current: float,
    min_cutlass_speedup_vs_dequant: float,
) -> None:
    if min_cutlass_speedup_vs_current <= 0.0 and min_cutlass_speedup_vs_dequant <= 0.0:
        return
    failures: list[str] = []
    current_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    cutlass_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    for item in results:
        key = (str(item["preset"]), int(item["rows"]), str(item["activation_quant"]))
        if item.get("env_variant") == "current":
            current_by_key[key] = item
        elif item.get("env_variant") == "int8_cutlass_fused":
            cutlass_by_key[key] = item
    for key, cutlass in cutlass_by_key.items():
        preset, rows, mode = key
        cutlass_vs_dequant = float(cutlass["model_stack_mlp_module_speedup_vs_dequant"])
        if min_cutlass_speedup_vs_dequant > 0.0 and cutlass_vs_dequant < min_cutlass_speedup_vs_dequant:
            failures.append(
                f"{preset} rows={rows} {mode}: CUTLASS MLP pair speedup vs dequant "
                f"{cutlass_vs_dequant:.3f} < {min_cutlass_speedup_vs_dequant:.3f}"
            )
        current = current_by_key.get(key)
        if current is None:
            continue
        current_ms = float(current["model_stack_mlp_module_ms"])
        cutlass_ms = float(cutlass["model_stack_mlp_module_ms"])
        speedup = current_ms / cutlass_ms if cutlass_ms > 0.0 else math.inf
        if min_cutlass_speedup_vs_current > 0.0 and speedup < min_cutlass_speedup_vs_current:
            failures.append(
                f"{preset} rows={rows} {mode}: CUTLASS MLP pair speedup vs current "
                f"{speedup:.3f} < {min_cutlass_speedup_vs_current:.3f} "
                f"(current_ms={current_ms:.6g}, cutlass_ms={cutlass_ms:.6g})"
            )
    if failures:
        print("MLP subgraph gate failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("MLP subgraph gate passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Parameter Golf BitNet MLP subgraphs on H100.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="runtime_row_1024x7_relu2_mlp3")
    parser.add_argument("--rows", default="65536")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--activation-quant", default="dynamic_int8")
    parser.add_argument("--activation-quant-bits", type=int, default=8)
    parser.add_argument("--activation-quant-method", default="absmax")
    parser.add_argument("--activation-quant-percentile", type=float, default=0.999)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--consume-output", action="store_true")
    parser.add_argument("--include-int8-cutlass-fused-variant", action="store_true")
    parser.add_argument("--min-cutlass-speedup-vs-current", type=float, default=0.0)
    parser.add_argument("--min-cutlass-speedup-vs-dequant", type=float, default=0.0)
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dtype = _dtype(args.dtype)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    module = native_module()
    header = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "native_module": getattr(module, "__file__", None),
        "has_linear_module_forward": bool(module is not None and hasattr(module, "linear_module_forward")),
        "has_int8_quantize_relu2_activation": bool(has_native_op("int8_quantize_relu2_activation")),
        "has_int8_quantize_relu2_activation_forward": bool(
            module is not None and hasattr(module, "int8_quantize_relu2_activation_forward")
        ),
    }
    print(json.dumps({"header": header}, sort_keys=True))

    preset = PRESETS[str(args.preset)]
    results: list[dict[str, object]] = []
    for variant_name, env in _env_variants(bool(args.include_int8_cutlass_fused_variant)):
        previous = {key: _set_env_variant(key, value) for key, value in env.items()}
        try:
            for rows in _parse_ints(args.rows):
                result = _bench_one(
                    preset,
                    rows,
                    dtype=dtype,
                    device=device,
                    bias=bool(args.bias),
                    activation_quant=str(args.activation_quant),
                    activation_quant_bits=int(args.activation_quant_bits),
                    activation_quant_method=str(args.activation_quant_method),
                    activation_quant_percentile=float(args.activation_quant_percentile),
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    repeats=int(args.repeats),
                    consume_output=bool(args.consume_output),
                )
                result["env_variant"] = variant_name
                result["env"] = dict(env)
                results.append(result)
                if args.jsonl:
                    print(json.dumps(result, sort_keys=True))
                else:
                    line = (
                        "variant={env_variant} preset={preset} rows={rows} "
                        "dense_mlp_ms={dense_bitnet_dequant_mlp_ms:.4f} "
                        "model_stack_mlp_ms={model_stack_mlp_module_ms:.4f} "
                        "model_stack_speedup={model_stack_mlp_module_speedup_vs_dequant:.3f}x "
                        "err={model_stack_mlp_module_max_abs_err_vs_dequant:.6g}"
                    ).format(**result)
                    if "native_linear_module_pair_ms" in result:
                        line += (
                            " native_pair_ms={native_linear_module_pair_ms:.4f} "
                            "native_pair_speedup={native_linear_module_pair_speedup_vs_dequant:.3f}x"
                        ).format(**result)
                    print(line)
        finally:
            for key, value in previous.items():
                _restore_env(key, value)

    _enforce_gate(
        results,
        min_cutlass_speedup_vs_current=float(args.min_cutlass_speedup_vs_current),
        min_cutlass_speedup_vs_dequant=float(args.min_cutlass_speedup_vs_dequant),
    )


if __name__ == "__main__":
    main()
