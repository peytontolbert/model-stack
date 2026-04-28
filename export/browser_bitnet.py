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


def _write_dense_tensor(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    entry = _write_tensor(path, tensor)
    entry["path"] = f"dense/{entry['path']}"
    return entry


def _model_config_dict(model_cfg: object | None) -> dict[str, Any] | None:
    if model_cfg is None:
        return None
    if is_dataclass(model_cfg):
        return asdict(model_cfg)
    if hasattr(model_cfg, "__dict__"):
        return dict(vars(model_cfg))
    return None


def _layer_manifest(name: str, layer: QuantizedLinearBitNet, layer_dir: Path) -> dict[str, Any]:
    act_quant_mode = str(getattr(layer, "act_quant_mode", "none"))
    if act_quant_mode not in {"none", "static_int8"}:
        raise ValueError(
            f"layer {name} uses act_quant_mode={act_quant_mode!r}; "
            "browser BitNet v1 supports only 'none' and 'static_int8'"
        )
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
        "encdec_js": "encdec_runtime.js",
    }
    copied: dict[str, str] = {}
    for key, filename in files.items():
        src = BROWSER_BITNET_DIR / filename
        dst = runtime_dir / filename
        shutil.copyfile(src, dst)
        copied[key] = str(dst.relative_to(outdir))
    return copied


def _is_inside_bitnet_parameter(name: str, bitnet_module_names: set[str]) -> bool:
    return any(name == module_name or name.startswith(module_name + ".") for module_name in bitnet_module_names)


def _dense_tensor_manifest(model: torch.nn.Module, outdir: Path, bitnet_module_names: set[str]) -> dict[str, Any]:
    dense_dir = outdir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if _is_inside_bitnet_parameter(name, bitnet_module_names):
            continue
        tensors[name] = _write_dense_tensor(dense_dir / f"{_safe_layer_name(name)}.f32.bin", param)
    for name, buffer in model.named_buffers():
        if _is_inside_bitnet_parameter(name, bitnet_module_names):
            continue
        if buffer.numel() == 0:
            continue
        tensors[name] = _write_dense_tensor(dense_dir / f"{_safe_layer_name(name)}.f32.bin", buffer)
    return tensors


def _infer_architecture(model: torch.nn.Module) -> str:
    if all(hasattr(model, attr) for attr in ("enc_embed", "dec_embed", "encoder", "decoder", "enc_norm", "dec_norm")):
        return "encoder_decoder"
    if hasattr(model, "encoder"):
        return "encoder"
    return "unknown"


def _linear_role(name: str) -> dict[str, Any]:
    parts = name.split(".")
    role: dict[str, Any] = {"name": name, "stack": "unknown", "kind": "linear"}
    if parts[0] == "encoder" and len(parts) >= 2 and parts[1].isdigit():
        role.update({"stack": "encoder", "layer": int(parts[1])})
        rest = ".".join(parts[2:])
    elif parts[0] == "decoder" and len(parts) >= 2 and parts[1].isdigit():
        role.update({"stack": "decoder", "layer": int(parts[1])})
        rest = ".".join(parts[2:])
    else:
        rest = name

    if ".attn." in f".{rest}." or rest.startswith("attn."):
        role["block"] = "self_attention"
    if rest.startswith("self_attn_block.attn."):
        role["block"] = "self_attention"
    elif rest.startswith("cross_block.cross."):
        role["block"] = "cross_attention"
    elif ".mlp." in f".{rest}." or rest.startswith("mlp."):
        role["block"] = "mlp"
    elif name == "lm_head":
        role.update({"stack": "output", "block": "lm_head"})

    projection = parts[-1]
    projection_map = {
        "w_q": "q",
        "w_k": "k",
        "w_v": "v",
        "w_o": "o",
        "w_in": "mlp_in",
        "w_out": "mlp_out",
        "lm_head": "lm_head",
    }
    role["projection"] = projection_map.get(projection, projection)
    return role


def _graph_manifest(model: torch.nn.Module, layers: list[dict[str, Any]]) -> dict[str, Any]:
    architecture = _infer_architecture(model)
    graph: dict[str, Any] = {
        "architecture": architecture,
        "linear_roles": [_linear_role(layer["name"]) for layer in layers],
    }
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        graph["d_model"] = int(getattr(cfg, "d_model", 0) or 0)
        graph["n_heads"] = int(getattr(cfg, "n_heads", 0) or 0)
        graph["n_layers"] = int(getattr(cfg, "n_layers", 0) or 0)
        graph["d_ff"] = int(getattr(cfg, "d_ff", 0) or 0)
        head_dim = getattr(cfg, "head_dim", None)
        graph["head_dim"] = int(head_dim) if head_dim is not None else (
            graph["d_model"] // graph["n_heads"] if graph["n_heads"] else 0
        )
        graph["activation"] = str(getattr(cfg, "activation", "silu"))
        graph["norm"] = str(getattr(cfg, "norm", "layernorm"))
        graph["rms_norm_eps"] = float(getattr(cfg, "rms_norm_eps", 1e-6))
        graph["vocab_size"] = int(getattr(cfg, "vocab_size", 0) or 0)
    if architecture == "encoder_decoder":
        graph["embeddings"] = {
            "encoder": "enc_embed.weight",
            "decoder": "dec_embed.weight",
        }
        graph["final_norms"] = {
            "encoder": {"weight": "enc_norm.weight", "bias": "enc_norm.bias"},
            "decoder": {"weight": "dec_norm.weight", "bias": "dec_norm.bias"},
        }
        graph["lm_head"] = "lm_head"
        graph["supports"] = {
            "encode": True,
            "decode": True,
            "cross_attention": True,
            "kv_cache": False,
            "batch_size": 1,
        }
    return graph


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

    bitnet_names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, QuantizedLinearBitNet)
    }
    layers = [
        _layer_manifest(name, module, layers_dir)
        for name, module in model.named_modules()
        if isinstance(module, QuantizedLinearBitNet)
    ]
    if not layers:
        raise ValueError("browser-bitnet export found no QuantizedLinearBitNet modules")

    runtime_files = _copy_runtime_files(output_dir)
    dense_tensors = _dense_tensor_manifest(model, output_dir, bitnet_names)
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
        "graph": _graph_manifest(model, layers),
        "dense_tensors": dense_tensors,
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
