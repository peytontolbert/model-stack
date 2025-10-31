from __future__ import annotations

from typing import Any, Dict, Iterable

from ..spaces import SearchSpace


class GridSearch:
    def __init__(self, max_points: int = 10_000) -> None:
        self.max_points = max_points

    def reset(self, space: SearchSpace, seed: int) -> None:  # noqa: ARG002
        pass

    def suggestions(self, space: SearchSpace, max_trials: int) -> Iterable[Dict[str, Any]]:
        count = 0
        if not space.is_finite():
            # fallback: use degenerate small grid
            for params in space.grid():
                yield params
                count += 1
                if count >= min(self.max_points, max_trials):
                    break
            return

        for params in space.grid():
            yield params
            count += 1
            if count >= min(self.max_points, max_trials):
                break


