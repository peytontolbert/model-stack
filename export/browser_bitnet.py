from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from compress.quantization import QuantizedLinearBitNet


ROOT = Path(__file__).resolve().parents[1]
BROWSER_BITNET_DIR = ROOT / "browser" / "bitnet"


def _safe_layer_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_tensor(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.uint8:
        data = cpu.numpy()
        dtype = "uint8"
    elif cpu.dtype in {torch.int32, torch.int64}:
        data = cpu.to(dtype=torch.int32).numpy()
        dtype = "int32"
    elif cpu.dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        data = cpu.to(dtype=torch.float32).numpy()
        dtype = "float32"
    else:
        raise TypeError(f"unsupported browser BitNet tensor dtype: {cpu.dtype}")
    path.write_bytes(data.tobytes(order="C"))
    return {
        "path": path.name,
        "dtype": dtype,
        "shape": list(cpu.shape),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _model_config_dict(model_cfg: object | None) -> dict[str, Any] | None:
    if model_cfg is None:
        return None
    if is_dataclass(model_cfg):
        return asdict(model_cfg)
    if hasattr(model_cfg, "__dict__"):
        return dict(vars(model_cfg))
    return None


def _layer_manifest(name: str, layer: QuantizedLinearBitNet, layer_dir: Path) -> dict[str, Any]:
    spec = layer.runtime_packed_linear_spec(backend="bitnet", dtype=torch.float32, device="cpu")
    if spec is None or spec.get("format") != "bitnet_w2a8":
        raise ValueError(f"layer {name} does not expose Model Stack BitNet packed state")
    if spec.get("spin_enabled"):
        raise ValueError(
            f"layer {name} uses spin transforms; browser BitNet export requires spin=False for v1"
        )
    if spec.get("pre_scale") is not None:
        raise ValueError(
            f"layer {name} uses AWQ/pre_scale; browser BitNet export requires quant_weight_opt='none' for v1"
        )
    act_quant_mode = str(spec.get("act_quant_mode", "none"))
    if act_quant_mode not in {"none", "static_int8"}:
        raise ValueError(
            f"layer {name} uses act_quant_mode={act_quant_mode!r}; "
            "browser BitNet v1 supports only 'none' and 'static_int8'"
        )

    layer_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_layer_name(name)
    tensors = {
        "packed_weight": _write_tensor(layer_dir / f"{safe_name}.packed_weight.u8.bin", spec["packed_weight"]),
        "scale_values": _write_tensor(layer_dir / f"{safe_name}.scale_values.f32.bin", spec["scale_values"]),
        "layout_header": _write_tensor(layer_dir / f"{safe_name}.layout_header.i32.bin", spec["layout_header"]),
        "segment_offsets": _write_tensor(layer_dir / f"{safe_name}.segment_offsets.i32.bin", spec["segment_offsets"]),
        "act_scale": _write_tensor(layer_dir / f"{safe_name}.act_scale.f32.bin", spec["act_scale"]),
    }
    if spec.get("bias") is not None:
        tensors["bias"] = _write_tensor(layer_dir / f"{safe_name}.bias.f32.bin", spec["bias"])

    layout_header = spec["layout_header"].detach().cpu().to(dtype=torch.int32).tolist()
    return {
        "name": name,
        "format": "bitnet_w2a8",
        "in_features": int(layer.in_features),
        "out_features": int(layer.out_features),
        "layout_header": [int(v) for v in layout_header],
        "act_quant_mode": act_quant_mode,
        "act_quant_method": str(spec.get("act_quant_method", "absmax")),
        "act_quant_bits": int(spec.get("act_quant_bits", 8)),
        "act_quant_percentile": float(spec.get("act_quant_percentile", 0.999)),
        "tensors": tensors,
    }


def _copy_runtime_files(outdir: Path) -> dict[str, str]:
    runtime_dir = outdir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "webgpu_js": "bitnet_webgpu.js",
        "wgsl": "bitnet_linear.wgsl",
    }
    copied: dict[str, str] = {}
    for key, filename in files.items():
        src = BROWSER_BITNET_DIR / filename
        dst = runtime_dir / filename
        shutil.copyfile(src, dst)
        copied[key] = str(dst.relative_to(outdir))
    return copied


def export_browser_bitnet_bundle(
    model: torch.nn.Module,
    outdir: str | Path,
    *,
    model_cfg: object | None = None,
    max_seq_len: int | None = None,
) -> Path:
    """Export packed BitNet linear tensors and browser runtime metadata.

    This is not an ONNX export. It preserves Model Stack's packed ternary state
    so Safari can run the custom WGSL/WASM runtime instead of relying on a
    nonexistent browser-native BitNet ONNX op.
    """

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    layers = [
        _layer_manifest(name, module, layers_dir)
        for name, module in model.named_modules()
        if isinstance(module, QuantizedLinearBitNet)
    ]
    if not layers:
        raise ValueError("browser-bitnet export found no QuantizedLinearBitNet modules")

    runtime_files = _copy_runtime_files(output_dir)
    manifest: dict[str, Any] = {
        "format": "model-stack-browser-bitnet",
        "format_version": 1,
        "runtime": {
            "primary": "webgpu",
            "fallback": "wasm-simd-required",
            "safari": {
                "https_required": True,
                "webgpu_feature_detection": "navigator.gpu",
                "threads_require_cross_origin_isolation": True,
            },
            "files": runtime_files,
        },
        "model": _model_config_dict(model_cfg),
        "max_seq_len": None if max_seq_len is None else int(max_seq_len),
        "layers": layers,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output_dir / "modelcard.json").write_text(
        json.dumps(
            {
                "format": "browser-bitnet",
                "manifest": manifest_path.name,
                "layer_count": len(layers),
                "sha256": _sha256(manifest_path),
                "runtime_files": runtime_files,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest_path
