from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch.nn")

from interpret import (
    activation_gradient_summary,
    capture_activation_gradients,
    diffusion_noise_prediction_metrics,
    diffusion_velocity_target,
    gradient_norm_summary,
    parameter_drift_summary,
    sequence_loss_attribution,
    snapshot_parameters,
    timestep_loss_buckets,
    token_cross_entropy_map,
    token_loss_summary,
    training_step_diagnostics,
)
from runtime.causal import CausalLM
from specs.config import ModelConfig


def _causal_model() -> CausalLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = CausalLM(cfg)
    model.train()
    return model


def test_training_gradient_and_parameter_diagnostics() -> None:
    torch.manual_seed(0)
    model = _causal_model()
    before = snapshot_parameters(model)
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 4))
    target = torch.randint(0, model.cfg.vocab_size, (1, 4))
    with capture_activation_gradients(model, ["blocks.0"]) as records:
        logits = model(input_ids)
        loss = token_cross_entropy_map(logits, target).mean()
        loss.backward()
    assert "blocks.0" in activation_gradient_summary(records)
    assert gradient_norm_summary(model)
    diag = training_step_diagnostics(model, before=before)
    assert "gradients" in diag
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= 0.01 * param.grad
    drift = parameter_drift_summary(before, snapshot_parameters(model))
    assert any(row["delta_l2"] > 0 for row in drift.values())


def test_sequence_and_diffusion_training_metrics() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[0, 1, 2], [2, 3, 4]])
    loss_map = token_cross_entropy_map(logits, targets)
    assert loss_map.shape == (2, 3)
    assert token_loss_summary(loss_map)["max"] >= token_loss_summary(loss_map)["min"]
    assert len(sequence_loss_attribution(logits, targets, topk=2)) == 2

    pred = torch.randn(4, 2, 2)
    target = torch.randn(4, 2, 2)
    timesteps = torch.tensor([0, 1, 5, 9])
    metrics = diffusion_noise_prediction_metrics(pred, target, timesteps=timesteps)
    assert "timestep_mse" in metrics
    assert timestep_loss_buckets(torch.ones(4), timesteps, bins=2)
    velocity = diffusion_velocity_target(target, pred, torch.ones(4), torch.zeros(4))
    assert velocity.shape == pred.shape
