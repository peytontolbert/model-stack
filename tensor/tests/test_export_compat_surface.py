from __future__ import annotations

from pathlib import Path

import torch

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
