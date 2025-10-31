from __future__ import annotations

import random
from typing import Any, Dict, Iterable

from ..spaces import SearchSpace


class RandomSearch:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, space: SearchSpace, seed: int) -> None:  # noqa: ARG002
        self._rng.seed(seed)

    def suggestions(self, space: SearchSpace, max_trials: int) -> Iterable[Dict[str, Any]]:
        for _ in range(max_trials):
            yield space.sample(self._rng)


