from __future__ import annotations

import torch


def diffusion_noise_prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    timesteps: torch.Tensor | None = None,
) -> dict[str, object]:
    err = (prediction.float() - target.float()).pow(2)
    per_example = err.flatten(1).mean(dim=1)
    out: dict[str, object] = {
        "mse": float(err.mean().item()),
        "per_example_mse": per_example.detach().cpu(),
    }
    if timesteps is not None:
        out["timestep_mse"] = timestep_loss_buckets(per_example, timesteps)
    return out


def timestep_loss_buckets(losses: torch.Tensor, timesteps: torch.Tensor, *, bins: int = 10) -> list[dict[str, float | int]]:
    losses_f = losses.detach().float().flatten()
    steps = timesteps.detach().float().flatten()
    if losses_f.numel() != steps.numel():
        raise ValueError("losses and timesteps must have the same number of elements")
    if losses_f.numel() == 0:
        return []
    lo, hi = float(steps.min().item()), float(steps.max().item())
    if lo == hi:
        return [{"start": int(lo), "end": int(hi), "count": int(losses_f.numel()), "mean_loss": float(losses_f.mean().item())}]
    edges = torch.linspace(lo, hi, steps=int(bins) + 1, device=steps.device)
    rows: list[dict[str, float | int]] = []
    for idx in range(int(bins)):
        start, end = edges[idx], edges[idx + 1]
        mask = (steps >= start) & ((steps <= end) if idx == int(bins) - 1 else (steps < end))
        if not mask.any():
            continue
        vals = losses_f[mask]
        rows.append({"start": int(start.item()), "end": int(end.item()), "count": int(mask.sum().item()), "mean_loss": float(vals.mean().item())})
    return rows


def diffusion_velocity_target(noise: torch.Tensor, sample: torch.Tensor, alpha_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
    while alpha_t.ndim < sample.ndim:
        alpha_t = alpha_t.unsqueeze(-1)
    while sigma_t.ndim < sample.ndim:
        sigma_t = sigma_t.unsqueeze(-1)
    return alpha_t.to(sample) * noise - sigma_t.to(sample) * sample
