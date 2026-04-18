from __future__ import annotations

from pathlib import Path

import torch

import export.exporter as runtime_exporter_mod
import model.export as model_export_mod


def _cfg():
    from specs.config import ModelConfig

    return ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        vocab_size=64,
        dtype="float32",
    )


def test_model_export_onnx_delegates_to_runtime_exporter(monkeypatch, tmp_path):
    model = torch.nn.Linear(4, 4, bias=False)
    seen = {}

    def fake_export_model(model_in, export_cfg, *, model_cfg=None):
        seen["model"] = model_in
        seen["export_cfg"] = export_cfg
        seen["model_cfg"] = model_cfg
        artifact = Path(export_cfg.outdir) / "model.onnx"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"onnx")
        return artifact

    monkeypatch.setattr(model_export_mod, "runtime_export_model", fake_export_model)

    out_path = tmp_path / "custom" / "artifact.onnx"
    out = model_export_mod.export_onnx(model, _cfg(), str(out_path))

    assert out == str(out_path)
    assert out_path.read_bytes() == b"onnx"
    assert seen["model"] is model
    assert seen["model_cfg"] == _cfg()
    assert seen["export_cfg"].target == "onnx"
    assert seen["export_cfg"].outdir == str(out_path.parent)


def test_model_export_torchscript_delegates_to_runtime_exporter(monkeypatch, tmp_path):
    model = torch.nn.Linear(4, 4, bias=False)
    seen = {}

    def fake_export_model(model_in, export_cfg, *, model_cfg=None):
        seen["model"] = model_in
        seen["export_cfg"] = export_cfg
        seen["model_cfg"] = model_cfg
        artifact = Path(export_cfg.outdir) / "model.ts"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"ts")
        return artifact

    monkeypatch.setattr(model_export_mod, "runtime_export_model", fake_export_model)

    out_path = tmp_path / "custom" / "artifact.ts"
    out = model_export_mod.export_torchscript(model, _cfg(), str(out_path))

    assert out == str(out_path)
    assert out_path.read_bytes() == b"ts"
    assert seen["model"] is model
    assert seen["model_cfg"] == _cfg()
    assert seen["export_cfg"].target == "torchscript"
    assert seen["export_cfg"].outdir == str(out_path.parent)


def test_pack_cli_uses_current_export_model_signature() -> None:
    source = (Path(__file__).resolve().parents[2] / "pack/cli.py").read_text(encoding="utf-8")
    assert "from export.exporter import export_model" in source
    assert "export_model(model, exp_cfg, model_cfg=cfg)" in source


def test_runtime_exporter_loads_calibration_input_map(tmp_path):
    calibration_path = tmp_path / "calibration.pt"
    expected = {"layer.w_q": torch.randn(4, 16)}
    torch.save({"calibration_inputs": expected}, calibration_path)

    loaded = runtime_exporter_mod._load_quant_calibration_inputs(
        type("ExportCfg", (), {"quant_calibration_inputs_path": str(calibration_path)})()
    )

    assert loaded is not None
    assert set(loaded) == {"layer.w_q"}
    assert torch.equal(loaded["layer.w_q"], expected["layer.w_q"])


def test_runtime_exporter_threads_awq_and_activation_quantization(monkeypatch, tmp_path):
    calibration_path = tmp_path / "calibration.pt"
    calibration_inputs = {"layer.w_q": torch.randn(8, 16)}
    torch.save(calibration_inputs, calibration_path)
    seen = {}

    def fake_apply_compression(model, *, quant=None, lora=None):
        del model, lora
        seen["quant"] = quant
        return {}

    monkeypatch.setattr(runtime_exporter_mod, "apply_compression", fake_apply_compression)

    runtime_exporter_mod._maybe_apply_quantization(
        torch.nn.Linear(16, 8, bias=False),
        type(
            "ExportCfg",
            (),
            {
                "quantize": "int4",
                "quant_spin": True,
                "quant_spin_seed": 7,
                "quant_weight_opt": "awq",
                "quant_activation_quant": "static_int8",
                "quant_activation_quant_bits": 8,
                "quant_activation_quant_method": "percentile",
                "quant_activation_quant_percentile": 0.995,
                "quant_calibration_inputs_path": str(calibration_path),
            },
        )(),
    )

    assert seen["quant"] is not None
    assert seen["quant"]["scheme"] == "int4"
    assert seen["quant"]["spin"] is True
    assert seen["quant"]["spin_seed"] == 7
    assert seen["quant"]["weight_opt"] == "awq"
    assert seen["quant"]["activation_quant"] == "static_int8"
    assert seen["quant"]["activation_quant_method"] == "percentile"
    assert seen["quant"]["activation_quant_percentile"] == 0.995
    assert set(seen["quant"]["calibration_inputs"]) == {"layer.w_q"}
