from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
import json
import hashlib
import os
import time


@dataclass
class ProgramState:
    root: str
    candidates_modules: List[str]
    candidates_files: List[str]
    citations: List[Tuple[str, int, int]]
    vec: Optional[List[float]] = None
    vec_weight: float = 1.0
    H: float = 0.0
    behavior_log: List[Dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.behavior_log is None:
            self.behavior_log = []

    def checksum(self) -> str:
        try:
            payload = {
                "root": self.root,
                "mods": sorted(list(self.candidates_modules)),
                "files": sorted(list(self.candidates_files)),
                "cites": sorted([(p, int(a), int(b)) for (p, a, b) in self.citations]),
                "H": float(self.H),
                "vec_w": float(self.vec_weight),
                "vec_n": int(len(self.vec) if isinstance(self.vec, list) else 0),
            }
            raw = json.dumps(payload, sort_keys=True).encode("utf-8", errors="ignore")
            return hashlib.sha256(raw).hexdigest()
        except Exception:
            return ""


def _ensure_dir(p: str) -> None:
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.loads(fh.read())
    except Exception:
        return {}


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(path)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, indent=2))
    except Exception:
        pass


def _from_dict(d: Dict[str, Any]) -> ProgramState:
    return ProgramState(
        root=str(d.get("root") or d.get("program_root") or ""),
        candidates_modules=list(d.get("candidates_modules") or []),
        candidates_files=list(d.get("candidates_files") or []),
        citations=[(str(p), int(a), int(b)) for (p, a, b) in (d.get("citations") or [])],
        vec=(list(d.get("vec")) if isinstance(d.get("vec"), list) else None),
        vec_weight=float(d.get("vec_weight") or 1.0),
        H=float(d.get("H") or 0.0),
        behavior_log=list(d.get("behavior_log") or []),
    )


def _to_dict(s: ProgramState) -> Dict[str, Any]:
    obj = asdict(s)
    obj["schema_version"] = 1
    obj["updated_at"] = time.time()
    return obj


def load_program_state(root: str, *, path: str | None = None) -> ProgramState:
    if not path or not os.path.exists(path):
        return ProgramState(root=str(root), candidates_modules=[], candidates_files=[], citations=[], vec=None, vec_weight=1.0, H=0.0, behavior_log=[])
    return _from_dict(_load_json(path))


def save_program_state(state: ProgramState, *, path: str | None = None) -> None:
    if not path:
        return
    _save_json(path, _to_dict(state))


def join_program_states(a: ProgramState, b: ProgramState) -> ProgramState:
    mods = sorted(list(set(a.candidates_modules) | set(b.candidates_modules)))
    files = sorted(list(set(a.candidates_files) | set(b.candidates_files)))
    cites = sorted(list(set(a.citations) | set(b.citations)))
    # Prefer newer vec if provided; blend weight heuristically
    vec = b.vec if (b.vec and len(b.vec) > 0) else a.vec
    vw = float(max(0.0, (a.vec_weight if a.vec else 0.0) + (b.vec_weight if b.vec else 0.0)))
    H = float(max(0.0, a.H + b.H))
    beh = list(a.behavior_log or []) + list(b.behavior_log or [])
    return ProgramState(
        root=(b.root or a.root),
        candidates_modules=mods,
        candidates_files=files,
        citations=cites,
        vec=vec,
        vec_weight=(vw if vw > 0 else 1.0),
        H=H,
        behavior_log=beh,
    )


def new_state_from_run(
    root: str,
    *,
    modules: List[str],
    files: List[str],
    citations: List[Tuple[str, int, int]],
    z_vec: Any = None,
    beh_event: Dict[str, Any] | None = None,
    H_increment: float = 0.0,
) -> ProgramState:
    return ProgramState(
        root=str(root),
        candidates_modules=list(modules or []),
        candidates_files=list(files or []),
        citations=list(citations or []),
        vec=(list(z_vec) if isinstance(z_vec, list) else (z_vec.tolist() if hasattr(z_vec, "tolist") else None)),
        vec_weight=1.0,
        H=float(max(0.0, H_increment or 0.0)),
        behavior_log=[(beh_event or {})],
    )


def changed_bits(a: ProgramState, b: ProgramState) -> bool:
    try:
        return a.checksum() != b.checksum()
    except Exception:
        return True


