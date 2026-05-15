from __future__ import annotations

import torch

from .renormalization import PSDMetricLossCritic


def psd_metric_matrix(critic: PSDMetricLossCritic, state: torch.Tensor, *, index: int = 0) -> torch.Tensor:
    diag, factors = critic.metric_params(state)
    d = torch.nn.functional.softplus(diag[int(index)].float())
    u = factors[int(index)].float()
    return torch.diag(d) + u @ u.T


def psd_metric_spectrum(critic: PSDMetricLossCritic, state: torch.Tensor, *, index: int = 0) -> torch.Tensor:
    matrix = psd_metric_matrix(critic, state, index=index)
    return torch.linalg.eigvalsh(0.5 * (matrix + matrix.T)).detach().cpu()


def metric_condition_summary(critic: PSDMetricLossCritic, state: torch.Tensor) -> dict[str, float]:
    conds = []
    max_eigs = []
    min_eigs = []
    for idx in range(int(state.shape[0])):
        eig = psd_metric_spectrum(critic, state, index=idx).float().clamp_min(1e-12)
        conds.append(eig.max() / eig.min())
        max_eigs.append(eig.max())
        min_eigs.append(eig.min())
    cond = torch.stack(conds)
    max_e = torch.stack(max_eigs)
    min_e = torch.stack(min_eigs)
    return {
        "condition_mean": float(cond.mean().item()),
        "condition_max": float(cond.max().item()),
        "max_eigen_mean": float(max_e.mean().item()),
        "min_eigen_mean": float(min_e.mean().item()),
    }


def validation_loss_correlation(learned_loss: torch.Tensor, validation_error: torch.Tensor) -> torch.Tensor:
    a = learned_loss.float().flatten()
    b = validation_error.float().flatten()
    if a.numel() != b.numel():
        raise ValueError("learned_loss and validation_error must have the same number of elements")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom.item()) == 0.0:
        return torch.tensor(0.0)
    return torch.dot(a, b) / denom


def high_cost_metric_direction(critic: PSDMetricLossCritic, state: torch.Tensor, *, index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = psd_metric_matrix(critic, state, index=index)
    eig, vec = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    top = torch.argmax(eig)
    return vec[:, top].detach().cpu(), eig[top].detach().cpu()
