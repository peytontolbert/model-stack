from __future__ import annotations

from pathlib import Path

from train import agent_kernel_lite


ROOT = Path(__file__).resolve().parents[1]


def test_agent_kernel_lite_training_lanes_resolve_to_imported_scripts() -> None:
    lanes = agent_kernel_lite.lane_map()

    for expected in {
        "diffusion-flow",
        "diffusion-rollout",
        "diffusion-live-teacher",
        "sana-latent",
        "f5tts-distill",
        "f5tts-streaming",
        "seq2seq",
    }:
        assert expected in lanes
        script = agent_kernel_lite.resolve_script(expected)
        assert script.is_file()
        assert script.is_relative_to(ROOT / "scripts" / "agent_kernel_lite")


def test_agent_kernel_lite_training_list_documents_model_stack_launcher() -> None:
    text = agent_kernel_lite.list_lanes()

    assert "python -m train.agent_kernel_lite run seq2seq" in text
    assert "train_agentkernel_lite_encdec.py" in text
    assert "distill_f5tts_q4_teacher.py" in text
    assert "train_agentkernel_lite_image_flux_flow_distill.py" in text
