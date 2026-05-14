from __future__ import annotations

from typing import Any, Callable

import torch


def extract_tensor_output(output: Any) -> torch.Tensor:
    """Best-effort extraction of an image or latent tensor from pipeline output."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("images", "latents", "sample", "prev_sample"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                return torch.stack(list(value))
    for key in ("images", "latents", "sample", "prev_sample"):
        value = getattr(output, key, None)
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
            return torch.stack(list(value))
    raise TypeError(f"Could not extract tensor output from {type(output).__name__}")


def mse_distance(a: Any, b: Any) -> torch.Tensor:
    x = extract_tensor_output(a).float()
    y = extract_tensor_output(b).float()
    return torch.mean((x - y) ** 2)


def cosine_distance(a: Any, b: Any) -> torch.Tensor:
    x = extract_tensor_output(a).float().flatten()
    y = extract_tensor_output(b).float().flatten()
    return 1.0 - torch.nn.functional.cosine_similarity(x, y, dim=0)


def resolve_diffusion_score(
    output: Any,
    *,
    reference: Any | None = None,
    score_fn: Callable[[Any], torch.Tensor | float] | None = None,
    distance_fn: Callable[[Any, Any], torch.Tensor] = mse_distance,
) -> torch.Tensor:
    if score_fn is not None:
        score = score_fn(output)
        if not isinstance(score, torch.Tensor):
            score = torch.as_tensor(score)
        if score.ndim != 0:
            raise ValueError("score_fn must return a scalar")
        return score
    if reference is not None:
        score = distance_fn(output, reference)
        if score.ndim != 0:
            raise ValueError("distance_fn must return a scalar")
        return score
    tensor = extract_tensor_output(output).float()
    return tensor.square().mean()
