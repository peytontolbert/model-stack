import os, json, hashlib
from pathlib import Path
from typing import Optional, Tuple

import torch

from specs.export import ExportConfig, export_model_delta
from specs.config import ModelConfig
from model.factory import build_model
from model.checkpoint import load_config, load_pretrained
from compress.apply import apply_compression


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


def export_model(model: torch.nn.Module, cfg: ExportConfig, *, model_cfg: Optional[ModelConfig] = None) -> Path:
    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    _maybe_apply_quantization(model, cfg)

    if cfg.target == "torchscript":
        ids, mask = _prepare_io_examples(model_cfg or ModelConfig(d_model=1, n_heads=1, n_layers=1, d_ff=1, vocab_size=2), device)
        scripted = torch.jit.trace(model, (ids, mask, None))
        out = outdir / "model.ts"; scripted.save(str(out))
        _write_card(outdir, {"format": "torchscript"}, out); return out

    if cfg.target == "onnx":
        ids, mask = _prepare_io_examples(model_cfg or ModelConfig(d_model=1, n_heads=1, n_layers=1, d_ff=1, vocab_size=2), device)
        out = outdir / "model.onnx"
        input_names = ["input_ids", "attn_mask", "cache"]
        output_names = ["logits"]
        dynamic_axes = None
        if cfg.dynamic_axes:
            dynamic_axes = {
                "input_ids": {0: "B", 1: "T"},
                "attn_mask": {0: "B", 1: "T"},
                "logits": {0: "B", 1: "T"},
            }
        torch.onnx.export(
            model,
            (ids, mask, None),
            str(out),
            input_names=input_names,
            output_names=output_names,
            opset_version=int(cfg.opset),
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
    mcfg: ModelConfig = load_config(model_dir)
    model = build_model(mcfg)
    model = load_pretrained(model, model_dir)
    # Delegate
    path = export_model(model, cfg, model_cfg=mcfg)
    # Optionally export compression delta alongside main artifact
    try:
        delta_path = str(Path(cfg.outdir) / "delta.pt")
        export_model_delta(model, delta_path)
    except Exception:
        pass
    return path

