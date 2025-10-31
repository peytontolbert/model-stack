from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List

from .base import Scheduler


@dataclass
class ASHAScheduler(Scheduler):
    budgets: List[int]  # e.g., [1, 3, 9] epochs/steps
    eta: int = 3        # reduction factor (controls number of configs promoted)
    seed: int | None = None

    def reset(self, seed: int) -> None:
        self.seed = seed

    def budgets(self, max_trials: int) -> Iterable[int]:
        # Multi-fidelity assignment: allocate many low-budget trials and fewer high-budget ones.
        rng = random.Random(self.seed)
        # Approximate per-rung counts like ASHA (no mid-trial prunes here):
        per_rung = []
        total = 0
        for i, _b in enumerate(self.budgets):
            # Highest rung has fewest trials
            n = max(1, int(max_trials / (self.eta ** i)))
            per_rung.append(n)
            total += n
        # Normalize to exactly max_trials
        if total > max_trials:
            # trim from lowest rungs first
            i = 0
            while total > max_trials and i < len(per_rung):
                if per_rung[i] > 0:
                    per_rung[i] -= 1
                    total -= 1
                if per_rung[i] == 0:
                    i += 1
        elif total < max_trials:
            per_rung[0] += (max_trials - total)

        # Emit budgets with slight shuffle per rung to mix search params
        out: List[int] = []
        for rung, count in enumerate(per_rung):
            out.extend([self.budgets[rung]] * max(0, count))
        rng.shuffle(out)
        return out


