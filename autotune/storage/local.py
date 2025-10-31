from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..trial import Trial


class LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "trials").mkdir(parents=True, exist_ok=True)

    def save_trial(self, trial: Trial) -> Path:
        path = self.root / "trials" / f"trial_{trial.id:05d}.json"
        with open(path, "w") as f:
            json.dump(asdict(trial), f, indent=2, default=str)
        return path

    def save_best(self, best_params: Dict[str, Any], best_score: float) -> Path:
        path = self.root / "best.json"
        with open(path, "w") as f:
            json.dump({"score": float(best_score), "params": best_params}, f, indent=2)
        return path

    def iter_trials(self) -> Iterable[Dict[str, Any]]:
        for p in sorted((self.root / "trials").glob("trial_*.json")):
            try:
                yield json.loads(p.read_text())
            except Exception:
                continue


