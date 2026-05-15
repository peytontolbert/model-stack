from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch.nn")

from interpret import (
    CostToGoCritic,
    DistributionCritic,
    DynamicLossWeightNet,
    PSDMetricLossCritic,
    cost_to_go_targets,
    critic_gradient_penalty,
    distribution_critic_scores,
    high_cost_metric_direction,
    inspect_psd_metric,
    learned_renormalization_loss,
    metric_condition_summary,
    outer_validation_objective,
    psd_metric_matrix,
    psd_metric_quadratic,
    psd_metric_spectrum,
    validation_loss_correlation,
    weighted_loss_operator,
    wgan_critic_loss,
    wgan_generator_loss,
)


def test_psd_metric_loss_and_inspection() -> None:
    torch.manual_seed(0)
    state = torch.randn(4, 5)
    error = torch.randn(4, 6)
    critic = PSDMetricLossCritic(5, 6, rank=2, hidden_dim=16)
    loss = critic(state, error)
    assert loss.shape == (4,)
    assert torch.all(loss >= 0)
    diag, factors = critic.metric_params(state)
    direct = psd_metric_quadratic(error, diag, factors)
    assert torch.allclose(loss, direct)
    inspection = inspect_psd_metric(critic, state)
    assert inspection.channel_importance.shape == (6,)
    assert psd_metric_matrix(critic, state).shape == (6, 6)
    assert psd_metric_spectrum(critic, state).shape == (6,)
    assert metric_condition_summary(critic, state)["condition_mean"] >= 1.0
    direction, value = high_cost_metric_direction(critic, state)
    assert direction.shape == (6,)
    assert value.item() >= 0.0
    assert validation_loss_correlation(torch.arange(4.0), torch.arange(4.0)).item() > 0.99


def test_dynamic_loss_weights_and_cost_to_go() -> None:
    torch.manual_seed(1)
    state = torch.randn(3, 4)
    error = torch.randn(3, 5)
    metric = PSDMetricLossCritic(4, 5, rank=1, hidden_dim=8)
    value = CostToGoCritic(4, hidden_dim=8)
    loss = learned_renormalization_loss(state=state, error=error, metric_critic=metric, value_critic=value, next_state=state)
    assert loss.shape == (3,)

    weight_net = DynamicLossWeightNet(4, 2, hidden_dim=8)
    weights = weight_net(state)
    weighted = weighted_loss_operator({"a": torch.ones(3), "b": torch.full((3,), 2.0)}, weights, ["a", "b"])
    assert weighted.shape == (3,)
    targets = cost_to_go_targets(torch.ones(3), torch.arange(3), gamma=0.5)
    assert targets.shape == (3,)
    outer = outer_validation_objective({"fresh": torch.ones(3), "cached": torch.ones(3) * 2}, {"fresh": 1.0, "cached": 0.5})
    assert outer.item() == 2.0


def test_distribution_critic_losses() -> None:
    torch.manual_seed(2)
    real = torch.randn(4, 3)
    fake = torch.randn(4, 3)
    critic = DistributionCritic(3, hidden_dim=8)
    real_scores = critic(real)
    fake_scores = critic(fake)
    assert wgan_critic_loss(real_scores, fake_scores).ndim == 0
    assert wgan_generator_loss(fake_scores).ndim == 0
    penalty = critic_gradient_penalty(critic, real, fake)
    assert penalty.item() >= 0.0
    scores = distribution_critic_scores(critic, real, fake)
    assert "gap" in scores
    critic_gradient_penalty,
    distribution_critic_scores,
    high_cost_metric_direction,
