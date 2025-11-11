from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any


def _resolve_pg_ctor(pg_backend: str):
    mod, _, attr = pg_backend.partition(":")
    if not mod or not attr:
        raise ValueError(f"Invalid ProgramGraph backend '{pg_backend}', expected 'module:ClassName'")
    m = __import__(mod, fromlist=[attr])
    return getattr(m, attr)


def emit_repository_knowledge(repo_root: str, out_path: str, pg_backend: str) -> str:
    """Emit a consolidated repository_knowledge.json with entities, edges, and artifact spans."""
    repo_root_abs = os.path.abspath(repo_root)
    ctor = _resolve_pg_ctor(pg_backend)
    pg = ctor(repo_root_abs, ignore=None)
    ents = []
    ents_by_id: Dict[str, Any] = {}
    # Entities
    for e in pg.entities():
        rec = {
            "id": e.id,
            "name": e.name,
            "kind": e.kind,
            "owner": e.owner,
            "uri": e.uri,
        }
        ents.append(rec)
        ents_by_id[e.id] = rec
    # Edges
    eds = []
    try:
        for ed in pg.edges():
            eds.append({"src": ed.src, "dst": ed.dst, "type": ed.type})
    except Exception:
        eds = []
    # Artifact anchors per entity (best-effort)
    anchors = {}
    for e in pg.entities():
        try:
            ra = pg.resolve(e.uri)
        except Exception:
            continue
        anchors[e.id] = {
            "artifact_uri": ra.artifact_uri,
            "path": ra.artifact_uri.split("/artifact/", 1)[-1],
            "span": {"start": int(ra.span.start_line), "end": int(ra.span.end_line)},
            "hash": ra.hash,
        }
    # Artifacts list (lightweight)
    arts = []
    try:
        for a in pg.artifacts("source"):
            arts.append({
                "uri": a.uri,
                "type": a.type,
                "hash": getattr(a, "hash", ""),
                "path": a.uri.split("/artifact/", 1)[-1],
            })
    except Exception:
        arts = []
    obj = {
        "schema_version": 1,
        "program_id": getattr(pg, "program_id", Path(repo_root_abs).name),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": repo_root_abs,
        "entities": ents,
        "edges": eds,
        "anchors": anchors,
        "artifacts": arts,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=2))
    return out_path


def main() -> None:
    # CLI: python emit_repository_knowledge.py <repo_root> <out_path> <pg_backend_module:Class>
    if len(sys.argv) < 4:
        print(
            "usage: python emit_repository_knowledge.py <repo_root> <out_path> <pg_backend module:Class>",
            file=sys.stderr,
        )
        sys.exit(2)
    repo_root = sys.argv[1]
    out_path = sys.argv[2]
    pg_backend = sys.argv[3]
    out = emit_repository_knowledge(repo_root, out_path, pg_backend)
    print(out)


if __name__ == "__main__":
    main()


