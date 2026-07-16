from __future__ import annotations

import json
from pathlib import Path

from runtime.cosmos3_lightx2v_bridge import (
    Cosmos3LightX2VPaths,
    build_cosmos3_lightx2v_launch_plan,
    cosmos3_lightx2v_status,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _make_layout(tmp_path: Path, *, sound: bool = True) -> Cosmos3LightX2VPaths:
    model = tmp_path / "Cosmos3-Nano"
    root = tmp_path / "LightX2V"
    for rel in (
        "lightx2v/__init__.py",
        "lightx2v/models/runners/cosmos3/cosmos3_runner.py",
        "lightx2v/models/networks/cosmos3/model.py",
        "lightx2v/models/schedulers/cosmos3/scheduler.py",
        "configs/cosmos3/cosmos3_super_omni_t2v.json",
    ):
        _touch(root / rel)
    (model / "model_index.json").parent.mkdir(parents=True, exist_ok=True)
    (model / "model_index.json").write_text(json.dumps({"_class_name": "Cosmos3OmniDiffusersPipeline"}), encoding="utf-8")
    _touch(model / "config.json")
    _touch(model / "transformer/config.json")
    (model / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors", "b": "diffusion_pytorch_model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    _touch(model / "transformer/diffusion_pytorch_model-00001-of-00002.safetensors")
    _touch(model / "transformer/diffusion_pytorch_model-00002-of-00002.safetensors")
    _touch(model / "vae/config.json")
    _touch(model / "vae/diffusion_pytorch_model.safetensors")
    _touch(model / "vision_encoder/config.json")
    _touch(model / "vision_encoder/model.safetensors")
    if sound:
        _touch(model / "sound_tokenizer/config.json")
        _touch(model / "sound_tokenizer/diffusion_pytorch_model.safetensors")
    return Cosmos3LightX2VPaths(model_path=model, lightx2v_root=root)


def test_cosmos3_lightx2v_status_complete_layout(tmp_path):
    paths = _make_layout(tmp_path)

    status = cosmos3_lightx2v_status(paths)

    assert status.status == "candidate_cosmos3_lightx2v_bridge"
    assert status.runnable is True
    assert status.preferred_env == "ai"
    assert status.supports_audio is True
    assert len(status.transformer_shards) == 2


def test_cosmos3_lightx2v_missing_shard_is_explicit(tmp_path):
    paths = _make_layout(tmp_path)
    (paths.transformer_dir / "diffusion_pytorch_model-00002-of-00002.safetensors").unlink()

    status = cosmos3_lightx2v_status(paths)

    assert status.status == "incomplete_cosmos3_lightx2v_assets"
    assert status.runnable is False
    assert any("missing transformer shard" in blocker for blocker in status.blockers)


def test_cosmos3_lightx2v_launch_plan_uses_ai_source_root_and_cache_env(tmp_path):
    paths = _make_layout(tmp_path, sound=False)
    config_json = tmp_path / "tiny_config.json"

    plan = build_cosmos3_lightx2v_launch_plan(paths, seed=7, config_json=str(config_json))

    assert plan.env["PYTHONPATH"] == str(paths.lightx2v_root)
    assert plan.env["HF_HOME"] == "/data/huggingface"
    assert plan.env["HUGGINGFACE_HUB_CACHE"] == "/data/huggingface/hub"
    assert "ai" in plan.command
    assert "lightx2v.infer" in plan.command
    assert str(paths.model_path) in plan.command
    assert str(config_json) in plan.command
    assert "7" in plan.command



def test_cosmos3_lightx2v_launch_plan_can_use_lazy_wrapper(tmp_path):
    paths = _make_layout(tmp_path, sound=False)
    config_json = tmp_path / "tiny_config.json"

    plan = build_cosmos3_lightx2v_launch_plan(paths, config_json=str(config_json), use_lazy_wrapper=True)

    assert "scripts/lightx2v_cosmos3_lazy_infer.py" in plan.command
    assert "lightx2v.infer" not in plan.command
    assert str(config_json) in plan.command
