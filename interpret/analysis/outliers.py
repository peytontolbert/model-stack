from __future__ import annotations

import torch


def channel_outlier_scores(x: torch.Tensor) -> torch.Tensor:
    flat = x.float().reshape(-1, x.shape[-1])
    med = flat.median(dim=0).values
    mad = (flat - med).abs().median(dim=0).values.clamp_min(1e-6)
    return (flat - med).abs().max(dim=0).values / mad


def activation_kurtosis(x: torch.Tensor) -> torch.Tensor:
    flat = x.float().reshape(-1, x.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    var = centered.pow(2).mean(dim=0).clamp_min(1e-12)
    return centered.pow(4).mean(dim=0) / var.pow(2)


def summarize_activation_outliers(x: torch.Tensor, *, topk: int = 10) -> list[dict[str, float | int]]:
    scores = channel_outlier_scores(x)
    kurt = activation_kurtosis(x)
    values, indices = torch.topk(scores, k=min(int(topk), scores.numel()))
    return [
        {
            "channel": int(idx.item()),
            "outlier_score": float(value.item()),
            "kurtosis": float(kurt[int(idx)].item()),
        }
        for value, idx in zip(values, indices)
    ]
