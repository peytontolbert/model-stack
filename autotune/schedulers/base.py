from __future__ import annotations

from typing import Iterable, Protocol


class Scheduler(Protocol):
    def reset(self, seed: int) -> None: ...  # noqa: E701
    def budgets(self, max_trials: int) -> Iterable[int]: ...  # noqa: E701


