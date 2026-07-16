from __future__ import annotations

from runtime.cosmos25_bridge import (
    build_cosmos25_predict_launch_plan,
    build_cosmos25_transfer_launch_plan,
    cosmos25_status,
)


def test_cosmos25_predict_reports_repo_checkpoint_and_runtime_gap(tmp_path):
    model_dir = tmp_path / "Cosmos-Predict2.5-14B"
    (model_dir / "base" / "pre-trained").mkdir(parents=True)
    (model_dir / "base" / "post-trained").mkdir(parents=True)
    (model_dir / "README.md").write_text("", encoding="utf-8")
    (model_dir / "base" / "pre-trained" / "54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt").write_text("", encoding="utf-8")
    (model_dir / "base" / "post-trained" / "e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt").write_text("", encoding="utf-8")

    status = cosmos25_status(model_dir, model_id="nvidia/Cosmos-Predict2.5-14B")

    assert status.family == "cosmos25_predict"
    assert status.status == "candidate_cosmos25_repo_checkpoint"
    assert status.runnable is False
    assert status.recommended_dtype == "bfloat16"
    assert status.supports_text is True
    assert status.supports_image is True
    assert status.supports_video is True
    assert "Cosmos 2.5 runtime bridge not wired into load_world_model yet." in status.blockers
    assert "base/pre-trained ema_bf16 checkpoint" in status.present_artifacts


def test_cosmos25_transfer_tracks_all_local_variants(tmp_path):
    model_dir = tmp_path / "Cosmos-Transfer2.5-2B"
    (model_dir / "auto" / "multiview").mkdir(parents=True)
    (model_dir / "distilled" / "general" / "edge").mkdir(parents=True)
    (model_dir / "README.md").write_text("", encoding="utf-8")
    (model_dir / "auto" / "multiview" / "4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt").write_text("", encoding="utf-8")
    (model_dir / "auto" / "multiview" / "b5ab002d-a120-4fbf-a7f9-04af8615710b_ema_bf16.pt").write_text("", encoding="utf-8")
    (model_dir / "distilled" / "general" / "edge" / "41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt").write_text("", encoding="utf-8")

    status = cosmos25_status(model_dir, model_id="nvidia/Cosmos-Transfer2.5-2B")

    assert status.family == "cosmos25_transfer"
    assert status.status == "candidate_cosmos25_repo_checkpoint"
    assert status.runnable is False
    assert not status.missing_artifacts
    assert "auto/multiview checkpoint A" in status.present_artifacts
    assert "distilled/general/edge checkpoint" in status.present_artifacts



def test_cosmos25_predict_launch_plan_uses_local_14b_checkpoint(tmp_path):
    model_dir = tmp_path / "Cosmos-Predict2.5-14B"
    runtime_root = tmp_path / "cosmos-predict2.5"
    checkpoint = model_dir / "base" / "post-trained" / "e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("", encoding="utf-8")

    plan = build_cosmos25_predict_launch_plan(model_dir, runtime_root=runtime_root)

    assert plan.cwd == str(runtime_root)
    assert plan.env["HF_HOME"] == "/data/huggingface"
    assert "cosmos25_py310" in plan.command
    assert f"--checkpoint-path={checkpoint}" in plan.command
    assert "--model=14B/post-trained" in plan.command
    assert "--offload-diffusion-model" in plan.command
    assert "--offload-tokenizer" in plan.command
    assert "--offload-text-encoder" in plan.command


def test_cosmos25_transfer_launch_plan_uses_distilled_edge_checkpoint_and_experimental_flag(tmp_path):
    model_dir = tmp_path / "Cosmos-Transfer2.5-2B"
    runtime_root = tmp_path / "cosmos-transfer2.5"
    checkpoint = model_dir / "distilled" / "general" / "edge" / "41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("", encoding="utf-8")

    plan = build_cosmos25_transfer_launch_plan(model_dir, runtime_root=runtime_root)

    assert plan.cwd == str(runtime_root)
    assert plan.env["COSMOS_EXPERIMENTAL_CHECKPOINTS"] == "1"
    assert "cosmos25_py310" in plan.command
    assert f"--checkpoint-path={checkpoint}" in plan.command
    assert "--model=edge/distilled" in plan.command
