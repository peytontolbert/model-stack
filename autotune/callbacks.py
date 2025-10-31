from __future__ import annotations

from pathlib import Path
from typing import Optional

from .trial import Trial


class VizLogger:
    def __init__(self, log_dir: str, metric_key: Optional[str] = None) -> None:
        self.log_dir = log_dir
        self.metric_key = metric_key or "autotune.objective"

        # Lazy import to avoid hard dependency
        try:
            from viz.session import VizSession  # type: ignore
        except Exception:  # pragma: no cover
            VizSession = None  # type: ignore
        self._viz = None if VizSession is None else VizSession(type("Cfg", (), {"log_dir": log_dir}))

        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def __call__(self, trial: Trial) -> None:
        if trial.result is None:
            return
        if self._viz is not None:
            self._viz.log_scalar(trial.id, self.metric_key, float(trial.result))
        else:  # minimal CSV logger
            import csv
            with open(Path(self.log_dir) / "autotune.csv", "a", newline="") as f:
                csv.writer(f).writerow([trial.id, self.metric_key, float(trial.result)])


class EarlyStopOnNoImprovement:
    def __init__(self, patience: int, mode: str = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.stale = 0

    def __call__(self, trial: Trial) -> None:
        if trial.result is None:
            return
        val = float(trial.result)
        better = (lambda a, b: a < b) if self.mode == "min" else (lambda a, b: a > b)
        if self.best is None or better(val, self.best):
            self.best = val
            self.stale = 0
        else:
            self.stale += 1
        if self.stale >= self.patience:
            trial.set_pruned()


