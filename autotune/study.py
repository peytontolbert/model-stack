from __future__ import annotations

import time
import random as _random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .spaces import SearchSpace
from .trial import Trial, TrialStatus
from .storage.local import LocalStorage


class Searcher(Protocol):
    def reset(self, space: SearchSpace, seed: int) -> None: ...  # noqa: E701
    def suggestions(self, space: SearchSpace, max_trials: int) -> Iterable[Dict[str, Any]]: ...  # noqa: E701


@dataclass
class StudyConfig:
    metric: str
    mode: str = "min"  # or "max"
    max_trials: int = 20
    seed: int = 1337
    log_dir: str = ".autotune"
    timeout_s: Optional[float] = None
    write_best_to: Optional[str] = None
    concurrency: int = 1
    budget_param: str = "budget"


class Study:
    def __init__(
        self,
        cfg: StudyConfig,
        space: SearchSpace,
        searcher: Searcher,
        storage: Optional[LocalStorage] = None,
        callbacks: Optional[List[Callable[[Trial], None]]] = None,
        scheduler: Optional["Scheduler"] = None,
    ) -> None:
        self.cfg = cfg
        self.space = space
        self.searcher = searcher
        self.storage = storage or LocalStorage(cfg.log_dir)
        self.callbacks = callbacks or []
        self.scheduler = scheduler

        self._best: Optional[Tuple[float, Dict[str, Any], int]] = None  # score, params, trial_id
        self._is_better = (lambda a, b: a < b) if cfg.mode == "min" else (lambda a, b: a > b)
        self._rng = _random.Random(cfg.seed)

    def _update_best(self, score: float, params: Dict[str, Any], trial_id: int) -> None:
        if self._best is None or self._is_better(score, self._best[0]):
            self._best = (score, dict(params), trial_id)
            self.storage.save_best(params, score)

    def optimize(self, objective: Callable[[Dict[str, Any], Trial], float]) -> Dict[str, Any]:
        self.searcher.reset(self.space, self.cfg.seed)
        if self.scheduler is not None:
            try:
                from .schedulers.base import Scheduler as _Scheduler  # type: ignore
                _ = isinstance(self.scheduler, _Scheduler)  # noqa: F841
            except Exception:
                pass
            self.scheduler.reset(self.cfg.seed)

        t_end = time.time() + self.cfg.timeout_s if self.cfg.timeout_s else None
        # Pre-generate suggestions to support concurrency
        suggestions = list(self.searcher.suggestions(self.space, self.cfg.max_trials))
        if self.scheduler is not None:
            budgets = list(self.scheduler.budgets(self.cfg.max_trials))
        else:
            budgets = [None] * len(suggestions)  # type: ignore

        def _run_one(idx_params_budget: Tuple[int, Dict[str, Any], Optional[int]]):
            idx, params, budget = idx_params_budget
            if budget is not None:
                params = {**params, self.cfg.budget_param: int(budget)}
            trial = Trial(id=idx, params=params)
            trial.set_running()
            try:
                score = float(objective(params, trial))
                trial.set_result(score)
            except Exception as e:  # pragma: no cover
                trial.set_failed(str(e))
            self.storage.save_trial(trial)
            for cb in self.callbacks:
                try:
                    cb(trial)
                except Exception:
                    pass
            return trial

        tasks: List[Tuple[int, Dict[str, Any], Optional[int]]] = []
        for i, params in enumerate(suggestions[: self.cfg.max_trials]):
            if t_end is not None and time.time() >= t_end:
                break
            bud = budgets[i] if i < len(budgets) else None
            tasks.append((i, params, bud))

        if self.cfg.concurrency <= 1:
            for idx, params, bud in tasks:
                trial = _run_one((idx, params, bud))
                if trial.result is not None:
                    self._update_best(trial.result, trial.params, trial.id)
        else:
            with ThreadPoolExecutor(max_workers=int(self.cfg.concurrency)) as ex:
                futures = [ex.submit(_run_one, t) for t in tasks]
                for fut in as_completed(futures):
                    trial = fut.result()
                    if trial.result is not None:
                        self._update_best(trial.result, trial.params, trial.id)

        if self._best is None:
            raise RuntimeError("No successful trials completed")

        best_score, best_params, _ = self._best
        if self.cfg.write_best_to:
            from pathlib import Path
            import json
            out = Path(self.cfg.write_best_to)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"metric": self.cfg.metric, "mode": self.cfg.mode, "score": best_score, "params": best_params}, indent=2))
        return best_params


