from __future__ import annotations

import re
from typing import Iterable

import torch

from interpret.activation_cache import ActivationCache
from .objectives import attention_entropy


_STEP_RE = re.compile(r"\.step_(\d+)\.")


def _step_from_key(key: str) -> int | None:
    match = _STEP_RE.search(key)
    return int(match.group(1)) if match else None


def diffusion_attention_phase_summary(cache: ActivationCache, *, attention_suffix: str = ".attn_probs") -> dict[str, object]:
    """Summarize cross-attention entropy and change over denoising steps."""

    by_step: dict[int, list[torch.Tensor]] = {}
    for key, value in cache.items():
        if not key.endswith(attention_suffix):
            continue
        step = _step_from_key(key)
        if step is None:
            continue
        by_step.setdefault(step, []).append(value.float())
    if not by_step:
        return {"steps": [], "entropy": [], "delta": [], "semantic_planning_end": None}

    steps = sorted(by_step)
    reduced: list[torch.Tensor] = []
    entropies: list[float] = []
    for step in steps:
        maps = [m.mean(dim=1) if m.ndim == 4 else m for m in by_step[step]]
        merged = torch.stack([m.reshape(-1, m.shape[-1]).mean(dim=0) for m in maps], dim=0).mean(dim=0)
        reduced.append(merged)
        entropies.append(float(attention_entropy(merged, dim=-1).mean().item()))

    deltas = [0.0]
    for prev, cur in zip(reduced, reduced[1:]):
        if prev.shape == cur.shape:
            deltas.append(float((cur - prev).abs().mean().item()))
        else:
            deltas.append(0.0)
    threshold = 0.1 * max(deltas) if deltas else 0.0
    semantic_end = None
    for step, delta in zip(steps[1:], deltas[1:]):
        if delta <= threshold:
            semantic_end = step
            break
    return {"steps": steps, "entropy": entropies, "delta": deltas, "semantic_planning_end": semantic_end}


def collect_diffusion_attention_maps(cache: ActivationCache, *, keys: Iterable[str] | None = None) -> list[torch.Tensor]:
    selected = list(keys) if keys is not None else [key for key in cache.keys() if key.endswith(".attn_probs")]
    return [value for key in selected if isinstance((value := cache.get(key)), torch.Tensor)]
