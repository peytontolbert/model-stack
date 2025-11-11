import os
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore


def _default_embed(text: str, dim: int = 256) -> List[float]:
    # Lightweight hashed bag-of-words embedding; deterministic and dependency-light
    vec = [0.0] * dim
    try:
        for tok in str(text or "").lower().split():
            h = (hash(tok) % dim + dim) % dim
            vec[h] += 1.0
        # L2 normalize to unit length (if non-zero)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
    except Exception:
        return vec


def _cosine(a: List[float], b: List[float]) -> float:
    # Cosine similarity for Python lists
    try:
        if _np is not None:
            va = _np.asarray(a, dtype=float)
            vb = _np.asarray(b, dtype=float)
            denom = (float(_np.linalg.norm(va)) * float(_np.linalg.norm(vb))) or 1.0
            return float(_np.dot(va, vb)) / denom
        # Pure-Python fallback
        denom = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
        return float(sum(x * y for x, y in zip(a, b))) / denom
    except Exception:
        return 0.0


class SymbolIndex:
    def __init__(self, items: List[Dict[str, Any]], embed: Optional[Callable[[str], List[float]]] = None) -> None:
        self._items = items or []
        self._embed = embed or _default_embed
        # Precompute embeddings for items without vec
        for it in self._items:
            if it.get("vec") is None:
                it["vec"] = self._embed(str(it.get("text") or it.get("name") or ""))

    def query(self, text: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self._items:
            return []
        qv = self._embed(text or "")
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for it in self._items:
            sc = _cosine(qv, it.get("vec") or [])
            scored.append((sc, it))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [it for _, it in scored[: max(1, int(k))]]


def _abs_paths_like(g, files: List[str]) -> List[str]:
    out: List[str] = []
    for rel in files or []:
        if not rel:
            continue
        if os.path.isabs(rel):
            out.append(os.path.abspath(rel))
        else:
            try:
                out.append(os.path.abspath(os.path.join(g.root, rel)))
            except Exception:
                out.append(os.path.abspath(rel))
    return out


def build_symbol_index(g, files: List[str], embed: Optional[Callable[[str], List[float]]] = None) -> SymbolIndex:
    """Build a small, in-memory index of symbols scoped to the selected files."""
    want = set(_abs_paths_like(g, files))
    items: List[Dict[str, Any]] = []
    try:
        # CodeGraph exposes symbols via symbols_by_fqn; iterate over values
        symbols_iter = []
        try:
            symbols_iter = list(getattr(g, "symbols_by_fqn", {}).values())
        except Exception:
            symbols_iter = []
        for s in symbols_iter or []:
            if s.kind not in ("function", "class", "module"):
                continue
            # Normalize path to absolute for matching
            sp = os.path.abspath(s.file)
            if want and sp not in want:
                continue
            # Prefer doc/signature text to aid matching
            txt = " ".join([t for t in [s.name, s.doc or "", s.signature or ""] if t])
            items.append({
                "name": s.name,
                "kind": s.kind,
                "file": sp,
                "rel": _to_repo_relative(getattr(g, "root", ""), sp),
                "line": int(getattr(s, "line", 1)),
                "end_line": int(getattr(s, "end_line", getattr(s, "line", 1))),
                "text": txt,
                "vec": None,  # deferred
            })
    except Exception:
        pass
    return SymbolIndex(items, embed=embed)


def choose_path_style(paths: List[str]) -> str:
    """Return 'basename' unless basenames collide, then 'repo_relative'."""
    try:
        bases = [os.path.basename(p) for p in paths or []]
        uniq = set(bases)
        return "repo_relative" if len(uniq) < len(bases) else "basename"
    except Exception:
        return "basename"


def _to_repo_relative(root: str, abs_path: str) -> str:
    try:
        root_abs = os.path.abspath(root or "")
        p_abs = os.path.abspath(abs_path or "")
        if p_abs.startswith(root_abs):
            rel = p_abs[len(root_abs):].lstrip(os.sep)
            return rel.replace("\\", "/")
        return abs_path.replace("\\", "/")
    except Exception:
        return abs_path.replace("\\", "/")


