from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not hasattr(torch, "nn"):
    pytest.skip("PyTorch with torch.nn is required for browser BitNet export tests", allow_module_level=True)

from compress.quantization import QuantizedLinearBitNet
from export.exporter import export_model
from specs.config import ModelConfig
from specs.export import ExportConfig


class TinyBitNetModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dense = torch.nn.Linear(4, 3, bias=True)
        with torch.no_grad():
            dense.weight.copy_(
                torch.tensor(
                    [
                        [-1.0, 0.0, 1.0, 0.5],
                        [0.25, -0.5, 0.0, 1.0],
                        [1.0, 1.0, -1.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )
            dense.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32))
        self.proj = QuantizedLinearBitNet(4, 3, bias=True).from_float(dense, spin=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_browser_bitnet_export_writes_manifest_runtime_and_layer_bins(tmp_path: Path) -> None:
    out = export_model(
        TinyBitNetModel(),
        ExportConfig(target="browser-bitnet", outdir=str(tmp_path)),
        model_cfg=ModelConfig(d_model=4, n_heads=1, n_layers=1, d_ff=8, vocab_size=16),
    )

    assert out == tmp_path / "manifest.json"
    manifest = json.loads(out.read_text())

    assert manifest["format"] == "model-stack-browser-bitnet"
    assert manifest["runtime"]["primary"] == "webgpu"
    assert manifest["runtime"]["files"]["webgpu_js"] == "runtime/bitnet_webgpu.js"
    assert manifest["runtime"]["files"]["wgsl"] == "runtime/bitnet_linear.wgsl"
    assert manifest["runtime"]["files"]["encdec_js"] == "runtime/encdec_runtime.js"
    assert (tmp_path / "runtime" / "bitnet_webgpu.js").exists()
    assert (tmp_path / "runtime" / "bitnet_linear.wgsl").exists()
    assert (tmp_path / "runtime" / "encdec_runtime.js").exists()

    assert len(manifest["layers"]) == 1
    layer = manifest["layers"][0]
    assert layer["name"] == "proj"
    assert layer["format"] == "bitnet_w2a8"
    assert layer["layout_header"][:3] == [1, 16, 32]
    assert layer["layout_header"][9] == 1

    for tensor in layer["tensors"].values():
        path = tmp_path / "layers" / tensor["path"]
        assert path.exists()
        assert path.stat().st_size == tensor["bytes"]


def test_browser_bitnet_export_rejects_dynamic_activation_quant(tmp_path: Path) -> None:
    model = TinyBitNetModel()
    model.proj.act_quant_mode = "dynamic_int8"

    try:
        export_model(model, ExportConfig(target="browser-bitnet", outdir=str(tmp_path)))
    except ValueError as exc:
        assert "dynamic_int8" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dynamic_int8 browser export to fail")
