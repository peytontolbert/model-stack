from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional


def write_json(outdir: str | Path, name: str, payload: Dict) -> Path:
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    f = p / f"{name}.json"
    f.write_text(json.dumps(payload, indent=2))
    return f


def append_csv(outdir: str | Path, name: str, row: Dict[str, object]) -> Path:
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    f = p / f"{name}.csv"
    write_header = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    return f


def viz_log_scalar(log_dir: Optional[str], step: int, key: str, value: float) -> None:
    if not log_dir:
        return
    try:
        from viz.session import VizSession  # type: ignore
        viz = VizSession(type("Cfg", (), {"log_dir": log_dir}))
        viz.log_scalar(step, key, float(value))
    except Exception:
        pass


