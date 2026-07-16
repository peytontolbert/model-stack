from __future__ import annotations

import json
from pathlib import Path

from runtime.hunyuan_avatar_bridge import (
    HunyuanAvatarPaths,
    build_hunyuan_avatar_launch_plan,
    hunyuan_avatar_status,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_manifest(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for rank in range(2):
        _touch(path / f"avatar_transformer.rank{rank:02d}.pt")
    (path / "avatar_transformer.manifest.json").write_text(
        json.dumps(
            {
                "format": "hunyuan-avatar-fsdp-sharded-v1",
                "world_size": 2,
                "shards": ["avatar_transformer.rank00.pt", "avatar_transformer.rank01.pt"],
            }
        ),
        encoding="utf-8",
    )


def _make_avatar_layout(tmp_path: Path) -> HunyuanAvatarPaths:
    model = tmp_path / "HunyuanVideo-Avatar"
    root = tmp_path / "hunyuanvideo-avatar"
    for rel in (
        "hymm_sp/__init__.py",
        "hymm_sp/config.py",
        "hymm_sp/sample_inference_audio.py",
    ):
        _touch(root / rel)
    for rel in (
        "ckpts/config.json",
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8_map.pt",
        "ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
        "ckpts/hunyuan-video-t2v-720p/vae/config.json",
        "ckpts/llava_llama_image/config.json",
        "ckpts/llava_llama_image/model.safetensors.index.json",
        "ckpts/text_encoder_2/model.safetensors",
        "ckpts/whisper-tiny/model.safetensors",
        "ckpts/det_align/detface.pt",
    ):
        _touch(model / rel)
    bf16 = tmp_path / "bf16_shards"
    fp8 = tmp_path / "fp8_shards"
    _write_manifest(bf16)
    _write_manifest(fp8)
    return HunyuanAvatarPaths(model_path=model, avatar_root=root, bf16_shard_dir=bf16, fp8_shard_dir=fp8)


def test_hunyuan_avatar_complete_layout_is_runnable_with_shards(tmp_path):
    paths = _make_avatar_layout(tmp_path)

    status = hunyuan_avatar_status(paths)

    assert status.status == "verified_hunyuan_avatar_custom_bridge_assets"
    assert status.runnable is True
    assert status.preferred_env == "py311build"
    assert status.supports_audio is True
    assert status.supports_image is True
    assert status.supports_video is True
    assert len(status.available_shard_dirs) == 2


def test_hunyuan_avatar_missing_runtime_is_explicit(tmp_path):
    paths = _make_avatar_layout(tmp_path)
    (paths.avatar_root / "hymm_sp" / "sample_inference_audio.py").unlink()

    status = hunyuan_avatar_status(paths)

    assert status.status == "incomplete_hunyuan_avatar_bridge_assets"
    assert status.runnable is False
    assert any("upstream sampler" in blocker for blocker in status.blockers)


def test_hunyuan_avatar_launch_plan_uses_local_model_base(tmp_path):
    paths = _make_avatar_layout(tmp_path)

    plan = build_hunyuan_avatar_launch_plan(paths=paths, mode="fp8_fsdp2", infer_steps=1, sample_frames=5)

    assert plan.env["MODEL_BASE"] == str(paths.model_path)
    assert plan.env["HUNYUAN_AVATAR_ROOT"] == str(paths.avatar_root)
    assert plan.shard_dir == str(paths.fp8_shard_dir)
    assert "scripts/sample_hunyuan_avatar_fp8_fsdp2.py" in plan.command
    assert "--use-fp8" in plan.command
    assert "--infer-steps" in plan.command
    assert "1" in plan.command
