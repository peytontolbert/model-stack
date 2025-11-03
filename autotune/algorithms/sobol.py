from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch

from ..spaces import Choice, IntRange, LogUniform, SearchSpace, Uniform


def _decode_from_unit(space: SearchSpace, unit_vector: List[float]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    keys = list(space.parameters.keys())
    for i, name in enumerate(keys):
        u = float(unit_vector[i])
        spec = space.parameters[name]
        if isinstance(spec, Choice):
            n = max(1, len(spec.options))
            idx = min(int(u * n), n - 1)
            params[name] = list(spec.options)[idx]
        elif isinstance(spec, IntRange):
            n = max(1, (spec.stop - spec.start + (spec.step - 1)) // spec.step)
            idx = min(int(u * n), n - 1)
            params[name] = spec.start + idx * spec.step
        elif isinstance(spec, Uniform):
            params[name] = spec.low + u * (spec.high - spec.low)
        elif isinstance(spec, LogUniform):
            import math
            a = math.log(spec.low, spec.base)
            b = math.log(spec.high, spec.base)
            params[name] = spec.base ** (a + u * (b - a))
        else:  # fallback to first grid point if unknown type
            try:
                params[name] = next(iter(spec.grid()))  # type: ignore[attr-defined]
            except Exception:
                params[name] = None
    return params


class SobolSearch:
    def __init__(self, scramble: bool = True, seed: int | None = None) -> None:
        self.scramble = scramble
        self.seed = seed
        self._engine: torch.quasirandom.SobolEngine | None = None

    def reset(self, space: SearchSpace, seed: int) -> None:  # noqa: ARG002
        dim = max(1, len(space.parameters))
        self._engine = torch.quasirandom.SobolEngine(dim=dim, scramble=bool(self.scramble), seed=self.seed if self.seed is not None else seed)

    def suggestions(self, space: SearchSpace, max_trials: int) -> Iterable[Dict[str, Any]]:
        if self._engine is None:
            self.reset(space, seed=0)
        assert self._engine is not None
        dim = max(1, len(space.parameters))
        # Draw in small batches to avoid large tensors for huge max_trials
        remaining = int(max_trials)
        while remaining > 0:
            batch = min(remaining, 1024)
            xs = self._engine.draw(batch)  # shape [batch, dim], values in [0,1)
            for i in range(batch):
                vec = xs[i].tolist()
                if len(vec) < dim:  # safety
                    vec = (vec + [0.0] * dim)[:dim]
                yield _decode_from_unit(space, vec)
            remaining -= batch


