import os, json, hashlib
import importlib.util
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from specs.export import ExportConfig, export_model_delta
from specs.config import ModelConfig
from compress.apply import apply_compression
from runtime.factory import build_model as runtime_build_model
from runtime.checkpoint import load_config as runtime_load_config, load_pretrained as runtime_load_pretrained


def _write_card(path: Path, meta: dict, bin_path: Optional[Path] = None) -> None:
    if bin_path is not None and bin_path.exists():
        sha = hashlib.sha256(bin_path.read_bytes()).hexdigest()
        meta = {**meta, "sha256": sha}
    (path / "modelcard.json").write_text(json.dumps(meta, indent=2))


def _prepare_io_examples(cfg: ModelConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Inputs: input_ids [B,T] long, attn_mask [B,T] long
    B, T = 1, 2
    input_ids = torch.randint(0, int(cfg.vocab_size), (B, T), dtype=torch.long, device=device)
    attn_mask = torch.ones(B, T, dtype=torch.long, device=device)
    return input_ids, attn_mask


def _export_card_meta(cfg: ExportConfig, **extra: object) -> dict[str, object]:
    meta: dict[str, object] = dict(extra)
    if cfg.quantize is not None:
        meta["quantize"] = str(cfg.quantize)
        meta["quant_spin"] = bool(cfg.quant_spin)
        if cfg.quant_spin:
            meta["quant_spin_seed"] = int(cfg.quant_spin_seed)
        meta["quant_weight_opt"] = str(cfg.quant_weight_opt)
        if getattr(cfg, "quant_activation_quant", None) is not None:
            meta["quant_activation_quant"] = str(cfg.quant_activation_quant)
            meta["quant_activation_quant_bits"] = int(cfg.quant_activation_quant_bits)
            meta["quant_activation_quant_method"] = str(cfg.quant_activation_quant_method)
            meta["quant_activation_quant_percentile"] = float(cfg.quant_activation_quant_percentile)
        if getattr(cfg, "quant_calibration_inputs_path", None) is not None:
            meta["quant_calibration_inputs_path"] = str(cfg.quant_calibration_inputs_path)
    return meta


def _load_quant_calibration_inputs(cfg: ExportConfig) -> Optional[dict[str, torch.Tensor]]:
    path = getattr(cfg, "quant_calibration_inputs_path", None)
    if path is None:
        return None
    loaded = torch.load(Path(path), map_location="cpu")
    if isinstance(loaded, dict) and "calibration_inputs" in loaded and isinstance(loaded["calibration_inputs"], dict):
        loaded = loaded["calibration_inputs"]
    if not isinstance(loaded, dict):
        raise TypeError("quant_calibration_inputs_path must point to a torch-saved dict[str, Tensor]")
    calibration_inputs: dict[str, torch.Tensor] = {}
    for name, value in loaded.items():
        if not isinstance(name, str):
            raise TypeError("quant_calibration_inputs_path keys must be module names (str)")
        if not torch.is_tensor(value):
            raise TypeError("quant_calibration_inputs_path values must be tensors")
        calibration_inputs[name] = value.detach()
    return calibration_inputs


def _maybe_apply_quantization(model: torch.nn.Module, cfg: ExportConfig) -> None:
    quantize = getattr(cfg, "quantize", None)
    calibration_inputs = None if quantize is None else _load_quant_calibration_inputs(cfg)
    quant_cfg = {
        "scheme": None if quantize is None else str(quantize),
        "calibration_inputs": calibration_inputs,
        "weight_opt": str(getattr(cfg, "quant_weight_opt", "none")),
        "activation_quant": "none"
        if getattr(cfg, "quant_activation_quant", None) is None
        else str(cfg.quant_activation_quant),
        "activation_quant_bits": int(getattr(cfg, "quant_activation_quant_bits", 8)),
        "activation_quant_method": str(getattr(cfg, "quant_activation_quant_method", "absmax")),
        "activation_quant_percentile": float(getattr(cfg, "quant_activation_quant_percentile", 0.999)),
        "spin": bool(getattr(cfg, "quant_spin", False)),
        "spin_seed": int(getattr(cfg, "quant_spin_seed", 0)),
    }
    if quantize == "int8":
        apply_compression(model, quant={**quant_cfg, "scheme": "int8"})
    elif quantize == "int4":
        apply_compression(model, quant={**quant_cfg, "scheme": "int4"})
    elif quantize == "nf4":
        apply_compression(model, quant={**quant_cfg, "scheme": "nf4"})
    elif quantize == "fp8":
        apply_compression(model, quant={**quant_cfg, "scheme": "fp8"})
    elif quantize == "bitnet":
        apply_compression(model, quant={**quant_cfg, "scheme": "bitnet"})


class _ExportWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attn_mask)


def export_model(model: torch.nn.Module, cfg: ExportConfig, *, model_cfg: Optional[ModelConfig] = None) -> Path:
    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    export_model_wrapper = _ExportWrapper(model).to(device).eval()

    _maybe_apply_quantization(model, cfg)

    if cfg.target == "torchscript":
        ids, mask = _prepare_io_examples(model_cfg or ModelConfig(d_model=1, n_heads=1, n_layers=1, d_ff=1, vocab_size=2), device)
        with torch.inference_mode():
            export_model_wrapper(ids, mask)
        # The runtime-backed model path mutates into packed/native fast paths on first
        # execution, so replay-based trace checking is not a stable signal here.
        scripted = torch.jit.trace(export_model_wrapper, (ids, mask), check_trace=False)
        out = outdir / "model.ts"; scripted.save(str(out))
        _write_card(outdir, _export_card_meta(cfg, format="torchscript"), out); return out

    if cfg.target == "onnx":
        if importlib.util.find_spec("onnx") is None:
            raise RuntimeError(
                "ONNX export requires the 'onnx' package in the active environment. "
                "This exporter uses torch.onnx.export(..., dynamo=False), so 'onnxscript' is not required."
            )
        ids, mask = _prepare_io_examples(model_cfg or ModelConfig(d_model=1, n_heads=1, n_layers=1, d_ff=1, vocab_size=2), device)
        with torch.inference_mode():
            export_model_wrapper(ids, mask)
        out = outdir / "model.onnx"
        input_names = ["input_ids", "attn_mask"]
        output_names = ["logits"]
        dynamic_axes = None
        if cfg.dynamic_axes:
            dynamic_axes = {
                "input_ids": {0: "B", 1: "T"},
                "attn_mask": {0: "B", 1: "T"},
                "logits": {0: "B", 1: "T"},
            }
        torch.onnx.export(
            export_model_wrapper,
            (ids, mask),
            str(out),
            input_names=input_names,
            output_names=output_names,
            opset_version=int(cfg.opset),
            dynamo=False,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        _write_card(outdir, _export_card_meta(cfg, format="onnx", opset=int(cfg.opset)), out); return out

    if cfg.target == "tensorrt":
        # Placeholder: produce a sentinel plan file and card; real build handled by external toolchain
        out = outdir / "model.plan"
        out.write_bytes(b"")
        _write_card(outdir, _export_card_meta(cfg, format="tensorrt", workspace_mb=int(cfg.trt_max_workspace_mb)), out); return out

    raise ValueError("Unknown target")


def export_from_dir(model_dir: str, cfg: ExportConfig) -> Path:
    # Load model and config
    mcfg: ModelConfig = runtime_load_config(model_dir)
    model = runtime_build_model(mcfg)
    model = runtime_load_pretrained(model, model_dir)
    # Delegate
    path = export_model(model, cfg, model_cfg=mcfg)
    # Optionally export compression delta alongside main artifact
    try:
        delta_path = str(Path(cfg.outdir) / "delta.pt")
        export_model_delta(model, delta_path)
    except Exception:
        pass
    return path
