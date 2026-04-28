# specs/export.py
from dataclasses import dataclass
from typing import Literal, Optional

from compress.export import build_delta, export_delta

@dataclass
class ExportConfig:
    target: Literal["torchscript","onnx","tensorrt","browser-bitnet"] = "onnx"
    opset: int = 19
    quantize: Optional[Literal["int8","int4","nf4","fp8","bitnet"]] = None
    quant_spin: bool = False
    quant_spin_seed: int = 0
    quant_weight_opt: Literal["none", "awq", "gptq"] = "none"
    quant_activation_quant: Optional[Literal["static_int8", "dynamic_int8"]] = None
    quant_activation_quant_bits: int = 8
    quant_activation_quant_method: Literal["absmax", "percentile", "mse"] = "absmax"
    quant_activation_quant_percentile: float = 0.999
    quant_calibration_inputs_path: Optional[str] = None
    dynamic_axes: bool = True
    max_seq_len: Optional[int] = None
    trt_max_workspace_mb: int = 4096
    outdir: str = "artifacts/"


def export_model_delta(model, path: str) -> None:
    """Export a lightweight delta artifact capturing compression state."""
    delta = build_delta(model=model)
    export_delta(path, delta)
