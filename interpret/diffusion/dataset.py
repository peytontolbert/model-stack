from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch

from .metrics import resolve_diffusion_score
from .tracing import DiffusionTracer


@dataclass(frozen=True)
class DiffusionTraceExample:
    prompt: str
    score: float | None
    cache_keys: tuple[str, ...]
    timesteps: tuple[int | float | None, ...]


@dataclass(frozen=True)
class DiffusionTraceDataset:
    examples: tuple[DiffusionTraceExample, ...]

    def scores(self) -> torch.Tensor:
        values = [example.score for example in self.examples if example.score is not None]
        return torch.tensor(values, dtype=torch.float32)


def trace_prompt_dataset(
    pipeline: Any,
    prompts: Iterable[str],
    *,
    num_inference_steps: int = 20,
    score_fn: Callable[[Any], torch.Tensor | float] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
) -> DiffusionTraceDataset:
    rows: list[DiffusionTraceExample] = []
    kwargs = dict(generation_kwargs or {})
    kwargs["num_inference_steps"] = int(num_inference_steps)
    for prompt in prompts:
        tracer = DiffusionTracer(pipeline)
        output, cache, records = tracer.trace_generation(prompt, **kwargs)
        score: float | None = None
        if score_fn is not None:
            score = float(resolve_diffusion_score(output, score_fn=score_fn).detach().cpu().item())
        rows.append(
            DiffusionTraceExample(
                prompt=prompt,
                score=score,
                cache_keys=tuple(cache.keys()),
                timesteps=tuple(record.timestep for record in records),
            )
        )
    return DiffusionTraceDataset(tuple(rows))


def summarize_diffusion_trace_dataset(dataset: DiffusionTraceDataset) -> dict[str, object]:
    scores = dataset.scores()
    out: dict[str, object] = {
        "examples": len(dataset.examples),
        "scored_examples": int(scores.numel()),
        "cache_key_counts": [len(example.cache_keys) for example in dataset.examples],
    }
    if scores.numel() > 0:
        out.update(
            {
                "score_mean": float(scores.mean().item()),
                "score_std": float(scores.std(unbiased=False).item()),
                "score_min": float(scores.min().item()),
                "score_max": float(scores.max().item()),
            }
        )
    return out
