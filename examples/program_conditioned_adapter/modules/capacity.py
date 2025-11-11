from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _parse_weights(spec: str) -> Dict[str, float]:
    # accept both "program" and "repo" as primary component key
    out: Dict[str, float] = {"program": 0.4, "subgraph": 0.4, "question": 0.2}
    try:
        parts = [p.strip() for p in str(spec).split(",") if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            out[str(k).strip()] = float(v)
    except Exception:
        pass
    s = float(sum(max(0.0, v) for v in out.values()))
    if s > 0:
        out = {k: max(0.0, v) / s for k, v in out.items()}
    return out


def _safe_div(a: float, b: float) -> float:
    return (a / b) if (b > 0) else 0.0


def entropy_score(
    g: object,
    mods: List[str],
    files_rel: List[str],
    *,
    weights: str = "program=0.4,subgraph=0.4,question=0.2",
) -> Tuple[float, Dict[str, float]]:
    """Compute a 0..1 capacity score from program graph + selection.

    Components:
      - repo_component: module count (log-normalized) and import density
      - subgraph_component: selected modules' degree and breadth
      - question_component: number of files hit by the query
    """
    try:
        n_mod = float(len(getattr(g, "modules", {}) or {}))
    except Exception:
        n_mod = 0.0
    try:
        import_edges = float(sum(len(v or []) for v in getattr(g, "module_imports", {}).values()))
    except Exception:
        import_edges = 0.0

    # Program-level
    repo_mod = min(1.0, _safe_div(np.log1p(n_mod), np.log1p(1000.0)))
    repo_deg = min(1.0, _safe_div(import_edges, max(1.0, n_mod * 8.0)))
    repo_comp = float(0.5 * repo_mod + 0.5 * repo_deg)

    # Subgraph-level
    mods_list = list(mods or [])
    sub_m = float(len(mods_list))
    sub_deg_sum = 0.0
    try:
        imports = getattr(g, "module_imports", {}) or {}
    except Exception:
        imports = {}
    for m in mods_list:
        try:
            indeg = sum(1 for _x, deps in imports.items() if m in (deps or []))
            outdeg = float(len(imports.get(m, []) or []))
            sub_deg_sum += float(indeg + outdeg)
        except Exception:
            continue
    sub_deg_norm = _safe_div(sub_deg_sum, max(1.0, sub_m * 8.0))
    sub_breadth = min(1.0, _safe_div(sub_m, max(1.0, n_mod)))
    sub_comp = float(0.5 * sub_deg_norm + 0.5 * sub_breadth)

    # Question-level
    q_files = float(len(files_rel or []))
    q_comp = min(1.0, _safe_div(q_files, 24.0))

    w = _parse_weights(weights)
    prim = w.get("program", w.get("repo", 0.4))
    score = float(prim * repo_comp + w.get("subgraph", 0.4) * sub_comp + w.get("question", 0.2) * q_comp)
    diag = {
        "program_modules": n_mod,
        "program_import_edges": import_edges,
        "program_component": repo_comp,
        "subgraph_modules": sub_m,
        "subgraph_deg_norm": sub_deg_norm,
        "subgraph_component": sub_comp,
        "question_files": q_files,
        "question_component": q_comp,
        "score": score,
    }
    return max(0.0, min(1.0, score)), diag


def scale_capacity(
    es: float, *, rank_min: int, rank_max: int, gsub_min: float, gsub_max: float
) -> Tuple[int, float]:
    r = int(round(rank_min + es * max(0, (rank_max - rank_min))))
    g = float(gsub_min + es * max(0.0, (gsub_max - gsub_min)))
    r = max(rank_min, min(rank_max, r))
    g = max(min(gsub_max, max(gsub_min, g)), 0.0)
    return r, g

