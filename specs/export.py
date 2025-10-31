# specs/export.py
from dataclasses import dataclass
from typing import Literal, Optional

from compress.export import build_delta, export_delta

@dataclass
class ExportConfig:
    target: Literal["torchscript","onnx","tensorrt"] = "onnx"
    opset: int = 19
    quantize: Optional[Literal["int8","fp8"]] = None
    dynamic_axes: bool = True
    max_seq_len: int = 4096
    trt_max_workspace_mb: int = 4096
    outdir: str = "artifacts/"


def export_model_delta(model, path: str) -> None:
    """Export a lightweight delta artifact capturing compression state."""
    delta = build_delta(model=model)
    export_delta(path, delta)
