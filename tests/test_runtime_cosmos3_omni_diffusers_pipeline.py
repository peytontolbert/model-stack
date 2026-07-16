from __future__ import annotations

import json
from pathlib import Path

from runtime.cosmos3_omni_diffusers_pipeline import (
    LEGACY_CLASS_NAME,
    UPSTREAM_CLASS_NAME,
    load_cosmos3_upstream_diffusers_pipeline,
    patch_diffusers_cosmos3,
    prepare_cosmos3_upstream_diffusers_snapshot,
)


def _make_cosmos3_snapshot(tmp_path: Path) -> Path:
    model = tmp_path / "Cosmos3-Nano"
    model.mkdir()
    (model / "transformer").mkdir()
    (model / "vae").mkdir()
    (model / "vision_encoder").mkdir()
    (model / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": LEGACY_CLASS_NAME,
                "_diffusers_version": "0.37.1",
                "scheduler": ["diffusers", "UniPCMultistepScheduler"],
                "transformer": ["diffusers", "Cosmos3OmniTransformer"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "vision_encoder": ["transformers", "Qwen3VLVisionModel"],
            }
        ),
        encoding="utf-8",
    )
    return model


def test_prepare_cosmos3_upstream_snapshot_rewrites_only_metadata(tmp_path):
    model = _make_cosmos3_snapshot(tmp_path)
    output = tmp_path / "adapter"

    prepared = prepare_cosmos3_upstream_diffusers_snapshot(model, output)

    assert prepared == output
    patched = json.loads((prepared / "model_index.json").read_text(encoding="utf-8"))
    assert patched["_class_name"] == UPSTREAM_CLASS_NAME
    assert patched["_model_stack_original_class_name"] == LEGACY_CLASS_NAME
    assert patched["_model_stack_removed_components"] == ["vision_encoder"]
    assert "vision_encoder" not in patched
    assert (prepared / "transformer").is_symlink()
    assert (prepared / "transformer").resolve() == model / "transformer"
    assert (prepared / "vae").resolve() == model / "vae"


def test_patch_diffusers_cosmos3_registers_upstream_alias():
    cls = patch_diffusers_cosmos3()

    import diffusers
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline

    assert cls is Cosmos3OmniPipeline
    assert diffusers.Cosmos3OmniDiffusersPipeline is Cosmos3OmniPipeline


def test_load_cosmos3_upstream_diffusers_pipeline_delegates_to_diffusers(monkeypatch, tmp_path):
    model = _make_cosmos3_snapshot(tmp_path)
    calls = {}

    class FakeDiffusionPipeline:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["path"] = Path(path)
            calls["kwargs"] = kwargs
            return "loaded"

    import diffusers

    monkeypatch.setattr(diffusers, "DiffusionPipeline", FakeDiffusionPipeline)

    loaded = load_cosmos3_upstream_diffusers_pipeline(model, adapter_dir=tmp_path / "adapter", torch_dtype="bf16")

    assert loaded == "loaded"
    assert calls["path"] == tmp_path / "adapter"
    assert calls["kwargs"]["local_files_only"] is True
    assert calls["kwargs"]["torch_dtype"] == "bf16"
