from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List

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
        else:
            try:
                params[name] = next(iter(spec.grid()))  # type: ignore[attr-defined]
            except Exception:
                params[name] = None
    return params


class LatinHypercubeSearch:
    def __init__(self, seed: int | None = None, jitter: bool = True) -> None:
        self.seed = seed
        self.jitter = jitter
        self._rng = random.Random(seed)

    def reset(self, space: SearchSpace, seed: int) -> None:  # noqa: ARG002
        self._rng.seed(self.seed if self.seed is not None else seed)

    def suggestions(self, space: SearchSpace, max_trials: int) -> Iterable[Dict[str, Any]]:
        dim = max(1, len(space.parameters))
        n = int(max_trials)
        # Create per-dimension permutations of bins 0..n-1
        perms: List[List[int]] = []
        for _ in range(dim):
            arr = list(range(n))
            self._rng.shuffle(arr)
            perms.append(arr)

        for i in range(n):
            vec: List[float] = []
            for d in range(dim):
                base = perms[d][i] / n
                if self.jitter:
                    base += self._rng.random() / n
                vec.append(min(base, 1.0 - 1e-12))
            yield _decode_from_unit(space, vec)


