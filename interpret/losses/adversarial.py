from __future__ import annotations

import torch
import torch.nn as nn


class DistributionCritic(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float().reshape(x.shape[0], -1)).squeeze(-1)


def wgan_critic_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    return fake_scores.float().mean() - real_scores.float().mean()


def wgan_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.float().mean()


def critic_gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    weight: float = 10.0,
) -> torch.Tensor:
    batch = int(real.shape[0])
    eps_shape = [batch] + [1] * (real.ndim - 1)
    eps = torch.rand(eps_shape, device=real.device, dtype=real.dtype)
    mixed = (eps * real + (1.0 - eps) * fake).detach().requires_grad_(True)
    scores = critic(mixed)
    grad = torch.autograd.grad(scores.sum(), mixed, create_graph=True, retain_graph=True)[0]
    grad_norm = grad.flatten(1).norm(dim=1)
    return float(weight) * (grad_norm - 1.0).pow(2).mean()


def distribution_critic_scores(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        real_scores = critic(real)
        fake_scores = critic(fake)
    return {
        "real_mean": float(real_scores.float().mean().item()),
        "fake_mean": float(fake_scores.float().mean().item()),
        "gap": float((real_scores.float().mean() - fake_scores.float().mean()).item()),
    }
