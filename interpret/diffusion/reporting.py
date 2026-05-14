from __future__ import annotations

from typing import Iterable

import torch

from .attribution import PromptTokenAttribution
from .tracing import DiffusionStepRecord


def summarize_diffusion_steps(records: Iterable[DiffusionStepRecord]) -> dict[str, object]:
    rows = list(records)
    timesteps = [row.timestep for row in rows]
    latent_shapes = [row.latent_shape for row in rows]
    return {
        "steps": len(rows),
        "timesteps": timesteps,
        "latent_shapes": latent_shapes,
    }


def summarize_prompt_token_attribution(rows: Iterable[PromptTokenAttribution], *, topk: int = 10) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: abs(row.delta), reverse=True)[: int(topk)]
    return [
        {
            "index": row.index,
            "token": row.token,
            "score": row.score,
            "delta": row.delta,
            "prompt": row.prompt,
        }
        for row in ordered
    ]


def summarize_token_heatmaps(heatmaps: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    return [
        {
            "token": token,
            "shape": tuple(heatmap.shape),
            "max": float(heatmap.max().item()),
            "mean": float(heatmap.float().mean().item()),
        }
        for token, heatmap in heatmaps.items()
    ]
