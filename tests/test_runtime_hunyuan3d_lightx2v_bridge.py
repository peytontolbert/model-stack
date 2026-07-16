from __future__ import annotations

from runtime.hunyuan3d_lightx2v_bridge import Hunyuan3DLightX2VPaths, hunyuan3d_lightx2v_status


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_hunyuan3d_21_shape_assets_route_to_lightx2v(tmp_path):
    model_dir = tmp_path / "Hunyuan3D-2.1"
    lightx2v = tmp_path / "LightX2V"
    for rel in (
        "lightx2v/__init__.py",
        "lightx2v/models/runners/hunyuan3d/hunyuan3d_shape_runner.py",
        "lightx2v/models/networks/hunyuan3d/model.py",
        "lightx2v/models/schedulers/hunyuan3d/scheduler.py",
        "lightx2v/models/input_encoders/hf/hunyuan3d/encoder.py",
        "lightx2v/models/video_encoders/hf/hunyuan3d/decoder.py",
        "configs/hunyuan3d/hunyuan3d_shape.json",
    ):
        _touch(lightx2v / rel)
    _touch(model_dir / "hunyuan3d-dit-v2-1" / "config.yaml")
    _touch(model_dir / "hunyuan3d-dit-v2-1" / "model.fp16.ckpt")

    status = hunyuan3d_lightx2v_status(Hunyuan3DLightX2VPaths(model_path=model_dir, lightx2v_root=lightx2v))

    assert status.status == "candidate_hunyuan3d_lightx2v_bridge"
    assert status.runnable is True
    assert status.preferred_env == "ai"
    assert status.shape_variants == ("hunyuan3d-dit-v2-1",)


def test_hunyuan3d_2mini_tracks_multiple_shape_variants(tmp_path):
    model_dir = tmp_path / "Hunyuan3D-2mini"
    lightx2v = tmp_path / "LightX2V"
    for rel in (
        "lightx2v/__init__.py",
        "lightx2v/models/runners/hunyuan3d/hunyuan3d_shape_runner.py",
        "lightx2v/models/networks/hunyuan3d/model.py",
        "lightx2v/models/schedulers/hunyuan3d/scheduler.py",
        "lightx2v/models/input_encoders/hf/hunyuan3d/encoder.py",
        "lightx2v/models/video_encoders/hf/hunyuan3d/decoder.py",
        "configs/hunyuan3d/hunyuan3d_shape.json",
    ):
        _touch(lightx2v / rel)
    for variant in ("hunyuan3d-dit-v2-mini", "hunyuan3d-dit-v2-mini-fast", "hunyuan3d-dit-v2-mini-turbo"):
        _touch(model_dir / variant / "config.yaml")
        _touch(model_dir / variant / "model.fp16.safetensors")

    status = hunyuan3d_lightx2v_status(Hunyuan3DLightX2VPaths(model_path=model_dir, lightx2v_root=lightx2v))

    assert status.runnable is True
    assert status.shape_variants == (
        "hunyuan3d-dit-v2-mini",
        "hunyuan3d-dit-v2-mini-fast",
        "hunyuan3d-dit-v2-mini-turbo",
    )


def test_hunyuan3d_omni_needs_specific_bridge(tmp_path):
    model_dir = tmp_path / "Hunyuan3D-Omni"
    for rel in (
        "config.json",
        "model/config.json",
        "model/pytorch_model.bin",
        "cond_encoder/pytorch_model.bin",
        "vae/pytorch_model.bin",
    ):
        _touch(model_dir / rel)

    status = hunyuan3d_lightx2v_status(Hunyuan3DLightX2VPaths(model_path=model_dir), model_id="Hunyuan3D-Omni")

    assert status.status == "needs_hunyuan3d_omni_bridge"
    assert status.runnable is False
    assert status.preferred_env == "hunyuan3d_omni_or_custom_bridge"
