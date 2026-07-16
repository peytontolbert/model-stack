from __future__ import annotations

import json

import pytest
import torch

if not hasattr(torch, "nn"):
    pytest.skip("requires a usable PyTorch install", allow_module_level=True)

from runtime.instruction_data_guard_bridge import InstructionDataGuardNet, load_instruction_data_guard


def _save_safetensors_or_skip(state_dict, path):
    try:
        from safetensors.torch import save_file
    except Exception as exc:
        pytest.skip(f"safetensors.torch unavailable in this env: {type(exc).__name__}:{exc}")
    save_file(state_dict, path)


def test_load_instruction_data_guard_scores_embeddings(tmp_path):
    model_dir = tmp_path / "instruction-data-guard"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"input_dim": 8, "dropout": 0.1}), encoding="utf-8")
    model = InstructionDataGuardNet(input_dim=8, dropout=0.1)
    _save_safetensors_or_skip(model.state_dict(), model_dir / "model.safetensors")

    artifacts = load_instruction_data_guard(model_dir, device="cpu", dtype="float32")
    scores = artifacts.score_embeddings(torch.ones(2, 8))

    assert artifacts.input_dim == 8
    assert artifacts.parameter_count == sum(param.numel() for param in model.parameters())
    assert tuple(scores.shape) == (2, 1)
    assert torch.all(scores >= 0)
    assert torch.all(scores <= 1)


def test_instruction_data_guard_rejects_wrong_embedding_dim(tmp_path):
    model_dir = tmp_path / "instruction-data-guard"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"input_dim": 8}), encoding="utf-8")
    _save_safetensors_or_skip(InstructionDataGuardNet(input_dim=8).state_dict(), model_dir / "model.safetensors")

    artifacts = load_instruction_data_guard(model_dir, device="cpu")

    try:
        artifacts.score_embeddings(torch.ones(2, 7))
    except ValueError as exc:
        assert "expected embeddings" in str(exc)
    else:
        raise AssertionError("expected dimension mismatch to fail")
