from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import importlib
import numpy as np


def _load_symbol(path: str):
    mod, _, attr = path.partition(":")
    m = importlib.import_module(mod)
    return getattr(m, attr)


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True)
    ap.add_argument("--pg-backend", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-modules", type=int, default=200)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pg_ctor = _load_symbol(args.pg_backend)
    pg = pg_ctor(args.program, ignore=None)

    # Collect entities and a simple owner map
    entities = list(pg.entities())
    modules = [e for e in entities if e.kind == "module"]
    functions = [e for e in entities if e.kind == "function"]
    classes = [e for e in entities if e.kind == "class"]

    # 1) planning_components.jsonl (owners/modules with primary artifact spans)
    comps: List[Dict] = []
    for m in modules[: max(1, int(args.max_modules))]:
        try:
            ra = pg.resolve(m.uri)
            comps.append({
                "owner": (m.name or m.id),
                "kind": "module",
                "artifact_uri": ra.artifact_uri,
                "span": {"start": int(ra.span.start_line), "end": int(ra.span.end_line)},
            })
        except Exception:
            continue
    _write_jsonl(out_dir / "planning_components.jsonl", comps)

    # 2) planning_entrypoints.jsonl (very light heuristic: functions named main or having owner 'cli'/'__main__')
    entries: List[Dict] = []
    for fn in functions:
        name = (fn.name or "").lower()
        own = (fn.owner or "").lower()
        if name == "main" or "__main__" in own or "cli" in own:
            try:
                ra = pg.resolve(fn.uri)
                entries.append({
                    "type": "cli",
                    "name": fn.name,
                    "handler": f"{own}:{fn.name}" if own and fn.name else fn.id,
                    "decl_uri": ra.artifact_uri,
                    "span": {"start": int(ra.span.start_line), "end": int(ra.span.end_line)},
                })
            except Exception:
                continue
    _write_jsonl(out_dir / "planning_entrypoints.jsonl", entries)

    # 3) planning_mutations.jsonl (modules as candidate edit sites with affordances)
    muts: List[Dict] = []
    for m in modules[: max(1, int(args.max_modules))]:
        try:
            ra = pg.resolve(m.uri)
            muts.append({
                "target": (m.name or m.id),
                "kind": "module",
                "artifact_uri": ra.artifact_uri,
                "affordances": ["add_function", "edit_imports", "edit_exports"],
            })
        except Exception:
            continue
    _write_jsonl(out_dir / "planning_mutations.jsonl", muts)

    # 4) planning_tests_map.jsonl (map owners to test files by simple heuristic)
    tests_map: List[Dict] = []
    # harvest artifacts(kind="source") and flag ones with "test" in path
    test_arts: Set[str] = set()
    for art in pg.artifacts("source"):
        p = art.uri.lower()
        if ("test" in p) or ("/tests/" in p):
            test_arts.add(art.uri)
    for m in modules:
        # naive mapping: module owner name in test path
        owner = (m.name or m.id)
        for ta in test_arts:
            if owner.split(".")[-1] in ta:
                tests_map.append({
                    "owner": owner,
                    "test_file": ta,
                    "span": {"start": 1, "end": 1},
                })
                break
    _write_jsonl(out_dir / "planning_tests_map.jsonl", tests_map)

    # 5) planning_dependencies.jsonl (import/call edges at entity granularity)
    deps: List[Dict] = []
    try:
        for e in pg.edges():
            deps.append({"src": e.src, "dst": e.dst, "edge": e.type})
    except Exception:
        deps = []
    _write_jsonl(out_dir / "planning_dependencies.jsonl", deps)

    # 6) planning_rerank_features.npz (very light owner graph centrality and placeholders)
    owners = sorted(list({(m.owner or m.name or m.id) for m in modules}))
    owner_index: Dict[str, int] = {o: i for i, o in enumerate(owners)}
    centrality = np.zeros((len(owners),), dtype=np.float32)
    try:
        owner_edges: Dict[str, Set[str]] = {}
        for e in pg.edges():
            # map entity ids to owners if available
            srco = None
            dsto = None
            # build entity -> owner map lazily
        ent_owner: Dict[str, Optional[str]] = {}
        for ent in entities:
            ent_owner[ent.id] = ent.owner
        for e in pg.edges():
            so = ent_owner.get(e.src)
            do = ent_owner.get(e.dst)
            if so and do:
                owner_edges.setdefault(so, set()).add(do)
                owner_edges.setdefault(do, set()).add(so)
        for o, nbs in owner_edges.items():
            idx = owner_index.get(o)
            if idx is not None:
                centrality[idx] = float(len(nbs))
        if centrality.max() > 0:
            centrality = centrality / centrality.max()
    except Exception:
        centrality = np.zeros((len(owners),), dtype=np.float32)
    registry_score = np.zeros_like(centrality, dtype=np.float32)
    test_coverage = np.zeros_like(centrality, dtype=np.float32)
    np.savez_compressed(
        out_dir / "planning_rerank_features.npz",
        owners=np.array(owners, dtype=object),
        centrality=centrality,
        registry_score=registry_score,
        test_coverage=test_coverage,
    )

    if args.verbose:
        print(json.dumps({
            "components": len(comps),
            "entrypoints": len(entries),
            "mutations": len(muts),
            "tests_map": len(tests_map),
            "dependencies": len(deps),
            "owners": len(owners),
        }, indent=2))


if __name__ == "__main__":
    main()


