from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


def psd_metric_quadratic(error: torch.Tensor, diag: torch.Tensor, factors: torch.Tensor | None = None) -> torch.Tensor:
    e = error.float().reshape(error.shape[0], -1)
    d = torch.nn.functional.softplus(diag.float()).reshape(e.shape[0], -1)
    if d.shape[-1] != e.shape[-1]:
        raise ValueError(f"diag width must match error width, got {d.shape[-1]} and {e.shape[-1]}")
    loss = (d * e.pow(2)).sum(dim=-1)
    if factors is not None:
        u = factors.float().reshape(e.shape[0], e.shape[-1], -1)
        projected = torch.einsum("bd,bdr->br", e, u)
        loss = loss + projected.pow(2).sum(dim=-1)
    return loss


class PSDMetricLossCritic(nn.Module):
    """Constrained learned metric: M(s) = diag+(s) + U(s)U(s)^T."""

    def __init__(self, state_dim: int, error_dim: int, *, rank: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.error_dim = int(error_dim)
        self.rank = int(rank)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.error_dim + self.error_dim * self.rank),
        )

    def metric_params(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(state.float())
        diag = out[..., : self.error_dim]
        factors = out[..., self.error_dim :].reshape(state.shape[0], self.error_dim, self.rank)
        return diag, factors

    def forward(self, state: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        diag, factors = self.metric_params(state)
        return psd_metric_quadratic(error, diag, factors)


class DynamicLossWeightNet(nn.Module):
    def __init__(self, state_dim: int, n_losses: int, *, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(state_dim), int(hidden_dim)), nn.SiLU(), nn.Linear(int(hidden_dim), int(n_losses)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(state.float()))


class CostToGoCritic(nn.Module):
    def __init__(self, state_dim: int, *, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(state_dim), int(hidden_dim)), nn.SiLU(), nn.Linear(int(hidden_dim), 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state.float()).squeeze(-1)


def weighted_loss_operator(loss_terms: Mapping[str, torch.Tensor], weights: torch.Tensor, names: list[str]) -> torch.Tensor:
    if weights.shape[-1] != len(names):
        raise ValueError("weights width must match names")
    stacked = torch.stack([loss_terms[name].float().reshape(weights.shape[0]) for name in names], dim=-1)
    return (weights.float() * stacked).sum(dim=-1)


def learned_renormalization_loss(
    *,
    state: torch.Tensor,
    error: torch.Tensor,
    metric_critic: PSDMetricLossCritic,
    value_critic: CostToGoCritic | None = None,
    next_state: torch.Tensor | None = None,
    analytic_anchor: torch.Tensor | None = None,
    anchor_weight: float = 1.0,
) -> torch.Tensor:
    loss = metric_critic(state, error)
    if value_critic is not None and next_state is not None:
        loss = loss + value_critic(next_state)
    if analytic_anchor is not None:
        loss = loss + float(anchor_weight) * analytic_anchor.float().reshape(loss.shape)
    return loss


@dataclass(frozen=True)
class MetricInspection:
    diag_mean: torch.Tensor
    factor_norm: torch.Tensor
    channel_importance: torch.Tensor


def inspect_psd_metric(critic: PSDMetricLossCritic, state: torch.Tensor) -> MetricInspection:
    diag, factors = critic.metric_params(state)
    diag_pos = torch.nn.functional.softplus(diag.detach().float())
    factor_norm = factors.detach().float().norm(dim=-1)
    return MetricInspection(
        diag_mean=diag_pos.mean(dim=0).cpu(),
        factor_norm=factor_norm.mean(dim=0).cpu(),
        channel_importance=(diag_pos + factor_norm).mean(dim=0).cpu(),
    )


def cost_to_go_targets(immediate_error: torch.Tensor, future_error: torch.Tensor, *, gamma: float = 1.0) -> torch.Tensor:
    return immediate_error.float().reshape(-1) + float(gamma) * future_error.float().reshape(-1)


def outer_validation_objective(terms: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
    total = None
    for name, weight in weights.items():
        if name not in terms:
            continue
        value = float(weight) * terms[name].float().mean()
        total = value if total is None else total + value
    return total if total is not None else torch.tensor(0.0)
