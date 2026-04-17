from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from export.exporter import export_model as runtime_export_model
from specs.config import ModelConfig
from specs.export import ExportConfig


def _legacy_export_config(
    out_path: str,
    *,
    target: str,
    export_cfg: ExportConfig | None = None,
) -> ExportConfig:
    params = asdict(export_cfg) if export_cfg is not None else asdict(ExportConfig())
    params["target"] = str(target)
    params["outdir"] = str(Path(out_path).parent or ".")
    return ExportConfig(**params)


def _move_artifact_to_requested_path(artifact_path: str | Path, out_path: str) -> str:
    artifact = Path(artifact_path)
    requested = Path(out_path)
    requested.parent.mkdir(parents=True, exist_ok=True)
    if artifact.resolve() != requested.resolve():
        artifact.replace(requested)
    return str(requested)


def export_onnx(
    model: torch.nn.Module,
    cfg: ModelConfig,
    out_path: str,
    *,
    export_cfg: ExportConfig | None = None,
) -> str:
    artifact = runtime_export_model(
        model,
        _legacy_export_config(out_path, target="onnx", export_cfg=export_cfg),
        model_cfg=cfg,
    )
    return _move_artifact_to_requested_path(artifact, out_path)


def export_torchscript(model: torch.nn.Module, cfg: ModelConfig, out_path: str) -> str:
    artifact = runtime_export_model(
        model,
        _legacy_export_config(out_path, target="torchscript"),
        model_cfg=cfg,
    )
    return _move_artifact_to_requested_path(artifact, out_path)


__all__ = [
    "export_onnx",
    "export_torchscript",
]
