from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

from .adapter import DiffusionInputs, get_diffusion_adapter, iter_prompt_token_replacements
from .metrics import mse_distance, resolve_diffusion_score


@dataclass(frozen=True)
class PromptTokenAttribution:
    index: int
    token: str
    prompt: str
    score: float
    delta: float


def prompt_token_occlusion_importance(
    pipeline: Any,
    prompt: str,
    *,
    replacement: str = "",
    score_fn: Callable[[Any], torch.Tensor | float] | None = None,
    distance_fn: Callable[[Any, Any], torch.Tensor] = mse_distance,
    generation_kwargs: dict[str, Any] | None = None,
) -> list[PromptTokenAttribution]:
    """Rank prompt tokens by generation score change under token removal/replacement."""

    adapter = get_diffusion_adapter(pipeline)
    kwargs = generation_kwargs or {}
    base_output = adapter.generate(DiffusionInputs(prompt=prompt, extra_kwargs=kwargs))
    base_score = resolve_diffusion_score(base_output, score_fn=score_fn)
    rows: list[PromptTokenAttribution] = []
    for idx, token, ablated_prompt in iter_prompt_token_replacements(prompt, replacement=replacement):
        output = adapter.generate(DiffusionInputs(prompt=ablated_prompt, extra_kwargs=kwargs))
        if score_fn is None:
            score = resolve_diffusion_score(output, reference=base_output, distance_fn=distance_fn)
            delta = score
        else:
            score = resolve_diffusion_score(output, score_fn=score_fn)
            delta = score - base_score
        rows.append(
            PromptTokenAttribution(
                index=int(idx),
                token=token,
                prompt=ablated_prompt,
                score=float(score.detach().cpu().item()),
                delta=float(delta.detach().cpu().item()),
            )
        )
    return rows


def prompt_counterfactual_delta(
    pipeline: Any,
    prompt: str,
    counterfactual_prompt: str,
    *,
    distance_fn: Callable[[Any, Any], torch.Tensor] = mse_distance,
    generation_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    adapter = get_diffusion_adapter(pipeline)
    kwargs = generation_kwargs or {}
    base = adapter.generate(DiffusionInputs(prompt=prompt, extra_kwargs=kwargs))
    changed = adapter.generate(DiffusionInputs(prompt=counterfactual_prompt, extra_kwargs=kwargs))
    return distance_fn(base, changed)


def token_region_attribution(
    attention_maps: torch.Tensor | Iterable[torch.Tensor],
    tokens: list[str],
    *,
    spatial_shape: tuple[int, int] | None = None,
    token_offset: int = 0,
    reduce_heads: str = "mean",
) -> dict[str, torch.Tensor]:
    """Aggregate cross-attention maps into per-token spatial heatmaps.

    Expected attention shape is ``[B, H, Q, K]`` where ``Q`` is latent spatial
    positions and ``K`` is prompt-token positions. Multiple maps are averaged.
    """

    if isinstance(attention_maps, torch.Tensor):
        maps = [attention_maps]
    else:
        maps = list(attention_maps)
    if not maps:
        return {}
    processed: list[torch.Tensor] = []
    for attn in maps:
        if attn.ndim == 3:
            attn = attn.unsqueeze(1)
        if attn.ndim != 4:
            raise ValueError(f"Expected attention map with rank 3 or 4, got shape {tuple(attn.shape)}")
        if reduce_heads == "mean":
            reduced = attn.float().mean(dim=1)
        elif reduce_heads == "max":
            reduced = attn.float().amax(dim=1)
        else:
            raise ValueError("reduce_heads must be 'mean' or 'max'")
        processed.append(reduced.mean(dim=0))
    merged = torch.stack(processed, dim=0).mean(dim=0)  # [Q, K]
    q_len = int(merged.shape[0])
    if spatial_shape is None:
        side = int(q_len ** 0.5)
        if side * side != q_len:
            spatial_shape = (q_len, 1)
        else:
            spatial_shape = (side, side)
    out: dict[str, torch.Tensor] = {}
    for idx, token in enumerate(tokens):
        token_index = idx + int(token_offset)
        if token_index >= merged.shape[-1]:
            break
        heatmap = merged[:, token_index].reshape(spatial_shape)
        denom = heatmap.max().clamp_min(1e-12)
        out[token] = heatmap / denom
    return out
