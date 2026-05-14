from __future__ import annotations

from typing import Iterable

import torch

from .faithfulness import spearman_rank_correlation


def explanation_stability(reference_scores: torch.Tensor, variant_scores: Iterable[torch.Tensor]) -> dict[str, object]:
    correlations = torch.tensor(
        [float(spearman_rank_correlation(reference_scores, scores).item()) for scores in variant_scores],
        dtype=torch.float32,
    )
    out: dict[str, object] = {"correlations": correlations}
    if correlations.numel() > 0:
        out.update(
            {
                "mean": float(correlations.mean().item()),
                "std": float(correlations.std(unbiased=False).item()),
                "min": float(correlations.min().item()),
                "max": float(correlations.max().item()),
            }
        )
    else:
        out.update({"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0})
    return out


def randomization_rank_baseline(
    scores: torch.Tensor,
    *,
    trials: int = 100,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    flat = scores.detach().flatten()
    values = []
    for _ in range(int(trials)):
        perm = torch.randperm(flat.numel(), generator=generator, device=flat.device)
        values.append(spearman_rank_correlation(flat, flat[perm]).detach().cpu())
    return torch.stack(values) if values else torch.empty(0)


def stability_summary(reference_scores: torch.Tensor, variant_scores: Iterable[torch.Tensor], *, random_trials: int = 100) -> dict[str, object]:
    stability = explanation_stability(reference_scores, variant_scores)
    random_corr = randomization_rank_baseline(reference_scores, trials=random_trials)
    stability["random_mean"] = float(random_corr.mean().item()) if random_corr.numel() else 0.0
    stability["random_std"] = float(random_corr.std(unbiased=False).item()) if random_corr.numel() else 0.0
    return stability
