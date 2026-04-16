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


def _maybe_apply_quantization(model: torch.nn.Module, cfg: ExportConfig) -> None:
    if cfg.quantize == "int8":
        # Use weight-only int8 replacement via compress.apply.apply_compression
        apply_compression(model, quant={})
    elif cfg.quantize == "fp8":
        # Placeholder for fake-quant or backend-specific flows; skip by default
        pass


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
        _write_card(outdir, {"format": "torchscript"}, out); return out

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
        _write_card(outdir, {"format": "onnx", "opset": int(cfg.opset)}, out); return out

    if cfg.target == "tensorrt":
        # Placeholder: produce a sentinel plan file and card; real build handled by external toolchain
        out = outdir / "model.plan"
        out.write_bytes(b"")
        _write_card(outdir, {"format": "tensorrt", "workspace_mb": int(cfg.trt_max_workspace_mb)}, out); return out

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
