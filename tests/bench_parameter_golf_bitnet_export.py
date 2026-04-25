from __future__ import annotations

import argparse
import importlib.util
import io
import lzma
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.quantization import QuantizedLinearBitNet
from runtime.ops import pack_bitnet_weight


DEFAULT_MODULES = (
    "blocks.0.attn.c_qkv.weight,"
    "blocks.0.attn.proj.weight,"
    "blocks.0.mlp.fc.weight,"
    "blocks.0.mlp.proj.weight"
)


def _load_pg_module(path: Path):
    spec = importlib.util.spec_from_file_location("parameter_golf_train_gpt_cuda_ternary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import Parameter Golf script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_quantized_artifact(artifact: Path):
    with artifact.open("rb") as f:
        return torch.load(io.BytesIO(lzma.decompress(f.read())), map_location="cpu", weights_only=False)


def _load_dequantized_state(pg_module, quantized, *, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return pg_module.deq_sd(quantized, target_dtype=dtype)


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


def _timeit(fn, *, warmup: int, iters: int, use_cuda_events: bool) -> float:
    with torch.inference_mode():
        for _ in range(int(warmup)):
            fn()
        if use_cuda_events:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(int(iters)):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / max(int(iters), 1)
        t0 = time.perf_counter()
        for _ in range(int(iters)):
            fn()
        return 1000.0 * (time.perf_counter() - t0) / max(int(iters), 1)


def _to_bitnet_layer(
    weight: torch.Tensor,
    *,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
) -> QuantizedLinearBitNet:
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    dense = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        dense.weight.copy_(weight.detach().float().cpu())
    return QuantizedLinearBitNet(in_features, out_features, bias=False).from_float(
        dense,
        calibration="absmax",
        activation_quant=activation_quant,
        activation_quant_bits=activation_quant_bits,
        activation_quant_method=activation_quant_method,
        activation_quant_percentile=activation_quant_percentile,
        spin=False,
    )


def _to_exact_runtime_row_bitnet_layer(pg_module, entry: dict, *, dtype: torch.dtype) -> QuantizedLinearBitNet:
    if entry.get("scale_layout") not in {"runtime_row", "row", "per_row"}:
        raise ValueError("exact runtime-row export requires an artifact entry with scale_layout=runtime_row")
    shape = entry["shape"]
    if len(shape) != 2:
        raise ValueError(f"exact runtime-row export expects rank-2 shape, got {shape}")
    if entry["type"] == "ternary":
        q = pg_module.unpack_ternary(entry["packed"], entry["n_trits"])
    elif entry["type"] == "ternary_bitmask":
        q = pg_module.unpack_ternary_bitmask(entry["packed"], entry["n_trits"])
    else:
        raise ValueError(f"exact runtime-row export only supports ternary entries, got {entry['type']}")
    q = q.reshape(shape).to(dtype=torch.float32)
    scale = entry["scale"].float().reshape(-1)
    weight = q * scale.unsqueeze(-1)
    out_features, in_features = int(shape[0]), int(shape[1])
    layout_header = torch.tensor(
        [
            1,
            16,
            32,
            out_features,
            in_features,
            ((out_features + 15) // 16) * 16,
            ((in_features + 31) // 32) * 32,
            2,
            1,
            1,
            80,
            1,
            0,
        ],
        dtype=torch.int32,
    )
    segment_offsets = torch.tensor([0, out_features], dtype=torch.int32)
    packed_weight, scale_values, layout_header, segment_offsets = pack_bitnet_weight(
        weight.to(dtype=dtype),
        scale_values=scale,
        layout_header=layout_header,
        segment_offsets=segment_offsets,
    )
    bitnet = QuantizedLinearBitNet(in_features, out_features, bias=False)
    bitnet._assign_packed_state(packed_weight, scale_values, layout_header, segment_offsets)
    return bitnet


def _effective_runtime_row_weight(pg_module, entry: dict, *, dtype: torch.dtype) -> torch.Tensor:
    if entry["type"] == "ternary":
        q = pg_module.unpack_ternary(entry["packed"], entry["n_trits"])
    elif entry["type"] == "ternary_bitmask":
        q = pg_module.unpack_ternary_bitmask(entry["packed"], entry["n_trits"])
    else:
        raise ValueError(f"exact runtime-row export only supports ternary entries, got {entry['type']}")
    shape = entry["shape"]
    q = q.reshape(shape).to(dtype=torch.float32)
    scale = entry["scale"].float().reshape(-1, 1)
    return (q * scale).to(dtype=dtype).contiguous()


def _runtime_row_q_and_scale(pg_module, entry: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if entry.get("scale_layout") not in {"runtime_row", "row", "per_row"}:
        raise ValueError("runtime-row export requires scale_layout=runtime_row")
    if entry["type"] == "ternary":
        q = pg_module.unpack_ternary(entry["packed"], entry["n_trits"])
    elif entry["type"] == "ternary_bitmask":
        q = pg_module.unpack_ternary_bitmask(entry["packed"], entry["n_trits"])
    else:
        raise ValueError(f"runtime-row export only supports ternary entries, got {entry['type']}")
    shape = tuple(int(v) for v in entry["shape"])
    if len(shape) != 2:
        raise ValueError(f"runtime-row export expects rank-2 weights, got {shape}")
    return q.reshape(shape).to(dtype=torch.float32), entry["scale"].float().reshape(-1)


def _pack_runtime_row_entry(pg_module, name: str, entry: dict, *, dtype: torch.dtype) -> dict[str, object]:
    q, scale = _runtime_row_q_and_scale(pg_module, entry)
    weight = q * scale.unsqueeze(-1)
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    layout_header = torch.tensor(
        [
            1,
            16,
            32,
            out_features,
            in_features,
            ((out_features + 15) // 16) * 16,
            ((in_features + 31) // 32) * 32,
            2,
            1,
            1,
            80,
            1,
            0,
        ],
        dtype=torch.int32,
    )
    segment_offsets = torch.tensor([0, out_features], dtype=torch.int32)
    packed_weight, scale_values, layout_header, segment_offsets = pack_bitnet_weight(
        weight.to(dtype=dtype),
        scale_values=scale,
        layout_header=layout_header,
        segment_offsets=segment_offsets,
    )
    return {
        "name": name,
        "kind": "bitnet_runtime_row",
        "shape": (out_features, in_features),
        "source_type": entry.get("type"),
        "packed_weight": packed_weight.cpu(),
        "scale_values": scale_values.cpu(),
        "layout_header": layout_header.cpu(),
        "segment_offsets": segment_offsets.cpu(),
    }


def _export_packed_bundle(pg_module, quantized: dict, output: Path, *, dtype: torch.dtype) -> dict[str, object]:
    packed: dict[str, object] = {}
    floating: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    for name, entry in quantized.items():
        if not isinstance(entry, dict) or "type" not in entry:
            skipped[str(name)] = "non_tensor_metadata"
            continue
        entry_type = str(entry.get("type"))
        if entry_type in {"ternary", "ternary_bitmask"}:
            if entry.get("scale_layout") in {"runtime_row", "row", "per_row"} and len(entry.get("shape", ())) == 2:
                packed[name] = _pack_runtime_row_entry(pg_module, name, entry, dtype=dtype)
            else:
                skipped[name] = f"unsupported_ternary_layout:{entry.get('scale_layout')}"
        else:
            floating[name] = pg_module.deq_sd({name: entry}, target_dtype=dtype)[name].cpu()
    bundle = {
        "format": "parameter_golf_runtime_row_bitnet_bundle_v1",
        "dtype": str(dtype),
        "packed": packed,
        "floating": floating,
        "skipped": skipped,
        "summary": {
            "packed_tensors": len(packed),
            "floating_tensors": len(floating),
            "skipped_tensors": len(skipped),
            "packed_params": int(sum(int(v["shape"][0]) * int(v["shape"][1]) for v in packed.values())),
            "floating_params": int(sum(v.numel() for v in floating.values())),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    return bundle["summary"]


def _load_packed_bundle(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _bitnet_from_packed_bundle_entry(entry: dict[str, object]) -> QuantizedLinearBitNet:
    shape = tuple(int(v) for v in entry["shape"])
    if len(shape) != 2:
        raise ValueError(f"packed bundle entry expects rank-2 shape, got {shape}")
    out_features, in_features = shape
    bitnet = QuantizedLinearBitNet(in_features, out_features, bias=False)
    bitnet._assign_packed_state(
        entry["packed_weight"],
        entry["scale_values"],
        entry["layout_header"],
        entry["segment_offsets"],
    )
    return bitnet


def _verify_packed_bundle(
    pg_module,
    quantized: dict,
    bundle_path: Path,
    *,
    dtype: torch.dtype,
    atol: float,
) -> dict[str, object]:
    bundle = _load_packed_bundle(bundle_path)
    if bundle.get("format") != "parameter_golf_runtime_row_bitnet_bundle_v1":
        raise ValueError(f"unsupported packed bundle format: {bundle.get('format')}")

    packed: dict[str, dict[str, object]] = bundle.get("packed", {})
    floating: dict[str, torch.Tensor] = bundle.get("floating", {})
    skipped: dict[str, str] = bundle.get("skipped", {})

    expected_packed: set[str] = set()
    expected_floating: set[str] = set()
    expected_skipped: set[str] = set()
    for name, entry in quantized.items():
        if not isinstance(entry, dict) or "type" not in entry:
            expected_skipped.add(str(name))
            continue
        entry_type = str(entry.get("type"))
        if entry_type in {"ternary", "ternary_bitmask"}:
            if entry.get("scale_layout") in {"runtime_row", "row", "per_row"} and len(entry.get("shape", ())) == 2:
                expected_packed.add(name)
            else:
                expected_skipped.add(name)
        else:
            expected_floating.add(name)

    failures: list[str] = []
    packed_names = set(packed)
    floating_names = set(floating)
    skipped_names = set(skipped)
    for name in sorted(expected_packed - packed_names):
        failures.append(f"missing packed tensor: {name}")
    for name in sorted(packed_names - expected_packed):
        failures.append(f"unexpected packed tensor: {name}")
    for name in sorted(expected_floating - floating_names):
        failures.append(f"missing floating tensor: {name}")
    for name in sorted(floating_names - expected_floating):
        failures.append(f"unexpected floating tensor: {name}")
    for name in sorted(expected_skipped - skipped_names):
        failures.append(f"missing skipped tensor: {name}")
    for name in sorted(skipped_names - expected_skipped):
        failures.append(f"unexpected skipped tensor: {name}")

    max_packed_abs_err = 0.0
    max_floating_abs_err = 0.0
    worst_packed = ""
    worst_floating = ""
    packed_params = 0
    floating_params = 0

    for name in sorted(expected_packed & packed_names):
        source_entry = quantized[name]
        bundle_entry = packed[name]
        expected = _effective_runtime_row_weight(pg_module, source_entry, dtype=torch.float32)
        bitnet = _bitnet_from_packed_bundle_entry(bundle_entry)
        actual = bitnet.runtime_weight(dtype=torch.float32, device="cpu")
        if tuple(actual.shape) != tuple(expected.shape):
            failures.append(f"packed shape mismatch for {name}: got {tuple(actual.shape)} expected {tuple(expected.shape)}")
            continue
        diff = (actual - expected).abs()
        err = float(diff.max().item()) if diff.numel() else 0.0
        packed_params += int(expected.numel())
        if err > max_packed_abs_err:
            max_packed_abs_err = err
            worst_packed = name
        if err > float(atol):
            failures.append(f"packed value drift for {name}: max_abs_err={err:.9g} atol={float(atol):.9g}")

    for name in sorted(expected_floating & floating_names):
        expected = pg_module.deq_sd({name: quantized[name]}, target_dtype=dtype)[name].cpu()
        actual = floating[name].cpu()
        if tuple(actual.shape) != tuple(expected.shape):
            failures.append(f"floating shape mismatch for {name}: got {tuple(actual.shape)} expected {tuple(expected.shape)}")
            continue
        diff = (actual.float() - expected.float()).abs()
        err = float(diff.max().item()) if diff.numel() else 0.0
        floating_params += int(expected.numel())
        if err > max_floating_abs_err:
            max_floating_abs_err = err
            worst_floating = name
        if err > float(atol):
            failures.append(f"floating value drift for {name}: max_abs_err={err:.9g} atol={float(atol):.9g}")

    summary = {
        "bundle": str(bundle_path),
        "verified": not failures,
        "failures": failures,
        "packed_tensors": len(packed),
        "floating_tensors": len(floating),
        "skipped_tensors": len(skipped),
        "packed_params": int(packed_params),
        "floating_params": int(floating_params),
        "max_packed_abs_err": float(max_packed_abs_err),
        "max_floating_abs_err": float(max_floating_abs_err),
        "worst_packed": worst_packed,
        "worst_floating": worst_floating,
    }
    if failures:
        raise RuntimeError(json.dumps(summary, sort_keys=True))
    return summary


def _bench_one(
    name: str,
    weight_cpu: torch.Tensor,
    *,
    rows: int,
    dtype: torch.dtype,
    device: torch.device,
    activation_quant: str,
    activation_quant_bits: int,
    activation_quant_method: str,
    activation_quant_percentile: float,
    backend: str | None,
    warmup: int,
    iters: int,
    pg_module=None,
    artifact_entry: dict | None = None,
) -> dict[str, object]:
    exact_runtime_row = artifact_entry is not None and artifact_entry.get("scale_layout") in {"runtime_row", "row", "per_row"}
    if exact_runtime_row:
        weight_cpu = _effective_runtime_row_weight(pg_module, artifact_entry, dtype=dtype)
    weight = weight_cpu.to(device=device, dtype=dtype).contiguous()
    x = torch.randn(int(rows), int(weight.shape[1]), device=device, dtype=dtype)
    export_mode = "requantized"
    if exact_runtime_row:
        bitnet = _to_exact_runtime_row_bitnet_layer(pg_module, artifact_entry, dtype=dtype)
        bitnet.act_quant_mode = str(activation_quant)
        bitnet.act_quant_bits = int(activation_quant_bits)
        bitnet.act_quant_method = str(activation_quant_method)
        bitnet.act_quant_percentile = float(activation_quant_percentile)
        export_mode = "exact_runtime_row"
    else:
        bitnet = _to_bitnet_layer(
            weight_cpu,
            activation_quant=activation_quant,
            activation_quant_bits=activation_quant_bits,
            activation_quant_method=activation_quant_method,
            activation_quant_percentile=activation_quant_percentile,
        )
    bitnet = bitnet.to(device=device)

    with torch.inference_mode():
        dense_out = F.linear(x, weight)
        bitnet_out = bitnet.runtime_linear(x, backend=backend)
        diff = (bitnet_out.float() - dense_out.float()).abs()

    use_cuda_events = device.type == "cuda"
    dense_ms = _timeit(lambda: F.linear(x, weight), warmup=warmup, iters=iters, use_cuda_events=use_cuda_events)
    bitnet_ms = _timeit(
        lambda: bitnet.runtime_linear(x, backend=backend),
        warmup=warmup,
        iters=iters,
        use_cuda_events=use_cuda_events,
    )
    return {
        "name": name,
        "shape": tuple(int(v) for v in weight.shape),
        "rows": int(rows),
        "dense_ms": float(dense_ms),
        "bitnet_ms": float(bitnet_ms),
        "speedup": float(dense_ms / bitnet_ms) if bitnet_ms > 0 else float("inf"),
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
        "dense_out_abs_mean": float(dense_out.float().abs().mean().item()),
        "export_mode": export_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark naive export of trained Parameter Golf ternary weights through the local BitNet runtime."
    )
    parser.add_argument("--pg-script", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--modules", type=str, default=DEFAULT_MODULES)
    parser.add_argument("--rows", type=str, default="1,16,4096,65536")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--activation-quant", type=str, default="dynamic_int8")
    parser.add_argument("--activation-quant-bits", type=int, default=8)
    parser.add_argument("--activation-quant-method", type=str, default="absmax")
    parser.add_argument("--activation-quant-percentile", type=float, default=0.999)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--export-packed", type=Path, default=None)
    parser.add_argument("--verify-packed", type=Path, default=None)
    parser.add_argument("--verify-export", action="store_true")
    parser.add_argument("--verify-atol", type=float, default=1e-6)
    parser.add_argument("--summary-json", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = _dtype_from_name(args.dtype)
    pg_module = _load_pg_module(args.pg_script)
    quantized = _load_quantized_artifact(args.artifact)
    if args.export_packed is not None:
        summary = _export_packed_bundle(pg_module, quantized, args.export_packed, dtype=dtype)
        verification = None
        if args.verify_export:
            verification = _verify_packed_bundle(
                pg_module,
                quantized,
                args.export_packed,
                dtype=dtype,
                atol=float(args.verify_atol),
            )
        if args.summary_json:
            payload = {"artifact": str(args.artifact), "export": str(args.export_packed), **summary}
            if verification is not None:
                payload["verification"] = verification
            print(json.dumps(payload, sort_keys=True))
        else:
            print(
                f"export={args.export_packed} packed_tensors={summary['packed_tensors']} "
                f"floating_tensors={summary['floating_tensors']} skipped_tensors={summary['skipped_tensors']} "
                f"packed_params={summary['packed_params']} floating_params={summary['floating_params']}"
            )
            if verification is not None:
                print(
                    f"verified={verification['verified']} max_packed_abs_err={verification['max_packed_abs_err']:.9g} "
                    f"max_floating_abs_err={verification['max_floating_abs_err']:.9g}"
                )
        return
    if args.verify_packed is not None:
        verification = _verify_packed_bundle(
            pg_module,
            quantized,
            args.verify_packed,
            dtype=dtype,
            atol=float(args.verify_atol),
        )
        if args.summary_json:
            print(json.dumps({"artifact": str(args.artifact), **verification}, sort_keys=True))
        else:
            print(
                f"verified={verification['verified']} bundle={args.verify_packed} "
                f"packed_tensors={verification['packed_tensors']} floating_tensors={verification['floating_tensors']} "
                f"skipped_tensors={verification['skipped_tensors']} packed_params={verification['packed_params']} "
                f"floating_params={verification['floating_params']} "
                f"max_packed_abs_err={verification['max_packed_abs_err']:.9g} "
                f"max_floating_abs_err={verification['max_floating_abs_err']:.9g}"
            )
        return
    state = _load_dequantized_state(pg_module, quantized, dtype=dtype)
    module_names = [item.strip() for item in str(args.modules).split(",") if item.strip()]
    rows_values = [int(item.strip()) for item in str(args.rows).split(",") if item.strip()]

    results: list[dict[str, object]] = []
    for module_name in module_names:
        if module_name not in state:
            raise KeyError(f"state tensor not found: {module_name}")
        weight = state[module_name]
        if weight.ndim != 2:
            raise ValueError(f"expected 2D weight for {module_name}, got shape {tuple(weight.shape)}")
        for rows in rows_values:
            results.append(
                _bench_one(
                    module_name,
                    weight,
                    rows=rows,
                    dtype=dtype,
                    device=device,
                    activation_quant=str(args.activation_quant),
                    activation_quant_bits=int(args.activation_quant_bits),
                    activation_quant_method=str(args.activation_quant_method),
                    activation_quant_percentile=float(args.activation_quant_percentile),
                    backend=args.backend,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    pg_module=pg_module,
                    artifact_entry=quantized.get(module_name),
                )
            )

    print(f"artifact={args.artifact}")
    print(f"dtype={dtype} device={device} activation_quant={args.activation_quant}")
    for result in results:
        print(
            "module={name} rows={rows} shape={shape} dense_ms={dense_ms:.4f} "
            "bitnet_ms={bitnet_ms:.4f} speedup={speedup:.3f}x export={export_mode} "
            "max_err={max_abs_err:.6f} mean_err={mean_abs_err:.6f} out_abs_mean={dense_out_abs_mean:.6f}".format(
                **result
            )
        )
    by_rows: dict[int, list[float]] = {}
    for result in results:
        by_rows.setdefault(int(result["rows"]), []).append(float(result["speedup"]))
    for rows, speedups in sorted(by_rows.items()):
        print(f"rows={rows} median_speedup={statistics.median(speedups):.3f}x")


if __name__ == "__main__":
    main()
