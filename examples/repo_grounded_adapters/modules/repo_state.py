from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set, Any
import os
import json
import time
import math
import numpy as np


def _now_ts() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0


def _unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return (x / n) if n > 0 else x


@dataclass
class RepoState:
    version: int = 1
    repo_root: str = ""
    # Monotone/idempotent components
    candidates_modules: Set[str] = field(default_factory=set)
    candidates_files: Set[str] = field(default_factory=set)
    facts: Set[Tuple[str, int, int]] = field(default_factory=set)  # (rel_path, a, b)
    beh_events: List[Dict[str, Any]] = field(default_factory=list)  # append-only
    # Vector view (anytime): keep a unit vector and a running weight
    vec: Optional[List[float]] = None
    vec_weight: float = 0.0
    # Capacity budget H
    H: float = 0.0
    # Last updated
    updated_at: float = 0.0

    def checksum(self) -> str:
        try:
            base = {
                "m": sorted(list(self.candidates_modules)),
                "f": sorted(list(self.candidates_files)),
                "facts": sorted([(p, int(a), int(b)) for (p, a, b) in self.facts]),
                "vw": float(self.vec_weight),
                "vl": len(self.vec or []),
                "H": float(self.H),
            }
            raw = json.dumps(base, sort_keys=True)
            import hashlib as _hh  # local
            return _hh.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return ""


def _default_state_path(repo_root: str) -> str:
    try:
        return os.path.join(os.path.abspath(repo_root), ".repo_state.json")
    except Exception:
        return ".repo_state.json"


def load_repo_state(repo_root: str, path: Optional[str] = None) -> RepoState:
    p = path or _default_state_path(repo_root)
    try:
        if os.path.isfile(p):
            obj = json.loads(open(p, "r", encoding="utf-8").read())
            st = RepoState(
                version=int(obj.get("version", 1)),
                repo_root=str(obj.get("repo_root") or repo_root),
                candidates_modules=set(obj.get("candidates_modules", []) or []),
                candidates_files=set(obj.get("candidates_files", []) or []),
                facts=set((t[0], int(t[1]), int(t[2])) for t in (obj.get("facts", []) or [])),
                beh_events=list(obj.get("beh_events", []) or []),
                vec=(obj.get("vec") if isinstance(obj.get("vec"), list) else None),
                vec_weight=float(obj.get("vec_weight", 0.0) or 0.0),
                H=float(obj.get("H", 0.0) or 0.0),
                updated_at=float(obj.get("updated_at", 0.0) or 0.0),
            )
            return st
    except Exception:
        pass
    return RepoState(version=1, repo_root=os.path.abspath(repo_root))


def save_repo_state(state: RepoState, path: Optional[str] = None) -> None:
    p = path or _default_state_path(state.repo_root)
    try:
        state.updated_at = _now_ts()
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass
    try:
        obj = {
            "version": state.version,
            "repo_root": state.repo_root,
            "candidates_modules": sorted(list(state.candidates_modules)),
            "candidates_files": sorted(list(state.candidates_files)),
            "facts": sorted([(p, int(a), int(b)) for (p, a, b) in state.facts]),
            "beh_events": state.beh_events,
            "vec": state.vec,
            "vec_weight": float(state.vec_weight),
            "H": float(state.H),
            "updated_at": float(state.updated_at),
        }
        open(p, "w", encoding="utf-8").write(json.dumps(obj, indent=2))
    except Exception:
        pass


def join_repo_states(old: RepoState, new: RepoState) -> RepoState:
    st = RepoState(version=max(int(old.version), int(new.version)))
    st.repo_root = old.repo_root or new.repo_root
    st.candidates_modules = set(old.candidates_modules) | set(new.candidates_modules)
    st.candidates_files = set(old.candidates_files) | set(new.candidates_files)
    st.facts = set(old.facts) | set(new.facts)
    st.beh_events = list(old.beh_events) + list(new.beh_events)
    # Vector join: unit-sum with weights, monotone in weight
    zv_old = np.array(old.vec, dtype=np.float32) if old.vec is not None else None
    zv_new = np.array(new.vec, dtype=np.float32) if new.vec is not None else None
    w_old = float(max(0.0, old.vec_weight))
    w_new = float(max(0.0, new.vec_weight))
    if zv_old is None and zv_new is None:
        st.vec, st.vec_weight = None, float(w_old + w_new)
    elif zv_old is None:
        st.vec, st.vec_weight = list(_unit(zv_new).tolist()), float(w_new + w_old)
    elif zv_new is None:
        st.vec, st.vec_weight = list(_unit(zv_old).tolist()), float(w_old + w_new)
    else:
        try:
            z = (w_old * _unit(zv_old)) + (w_new * _unit(zv_new))
            st.vec = list(_unit(z).tolist())
        except Exception:
            st.vec = list(_unit(zv_old).tolist())
        st.vec_weight = float(w_old + w_new)
    st.H = max(float(old.H), float(new.H))
    st.updated_at = max(float(old.updated_at), float(new.updated_at), _now_ts())
    return st


def changed_bits(prev: RepoState, cur: RepoState) -> bool:
    try:
        if len(cur.candidates_modules) > len(prev.candidates_modules):
            return True
        if len(cur.candidates_files) > len(prev.candidates_files):
            return True
        if len(cur.facts) > len(prev.facts):
            return True
        if float(cur.vec_weight) > float(prev.vec_weight):
            return True
    except Exception:
        return True
    return False


def new_state_from_run(
    repo_root: str,
    *,
    modules: List[str],
    files: List[str],
    citations: List[Tuple[str, int, int]],
    z_vec: Optional[np.ndarray],
    beh_event: Optional[Dict[str, Any]] = None,
    H_increment: float = 0.0,
) -> RepoState:
    st = RepoState(version=1, repo_root=os.path.abspath(repo_root))
    st.candidates_modules = set([m for m in (modules or []) if m])
    st.candidates_files = set([f for f in (files or []) if f])
    st.facts = set([(p, int(a), int(b)) for (p, a, b) in (citations or [])])
    st.beh_events = ([beh_event] if beh_event else [])
    if z_vec is not None:
        try:
            st.vec = list(_unit(z_vec.astype(np.float32)).tolist())
            st.vec_weight = 1.0
        except Exception:
            st.vec, st.vec_weight = None, 0.0
    st.H = float(max(0.0, H_increment))
    st.updated_at = _now_ts()
    return st


