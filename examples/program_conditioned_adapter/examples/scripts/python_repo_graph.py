from __future__ import annotations

import os
from typing import Iterable, List, Tuple, Dict, Optional, Set

from examples.program_conditioned_adapter.modules.program_graph import (
    Entity,
    Edge,
    Artifact,
    Span,
    ResolvedAnchor,
    EntityId,
)
from .codegraph_core import CodeGraph, CGEntity, CGEdge
from .repo_graph import RepoGraph, artifact_uri, program_id_for_repo, parse_program_uri


def _entity_uri(program_id: str, cg: CGEntity) -> str:
    if cg.kind == "module" or cg.kind == "test_module":
        resource = cg.name
        kind = "module"
    else:
        resource = cg.id.split("py:", 1)[-1]
        kind = cg.kind
    return f"program://{program_id}/{kind}/{resource}#L{cg.start_line}-L{cg.end_line}"


class PythonRepoGraph(RepoGraph):
    def __init__(self, repo_root: str, ignore: Optional[List[str]] = None):
        super().__init__(repo_root, ignore=ignore)
        self._cg = CodeGraph(self.repo_root, ignore=ignore).build()
        self._ent_cache: Optional[List[Entity]] = None
        self._edge_cache: Optional[List[Edge]] = None
        self._ent_by_id: Dict[str, CGEntity] = {e.id: e for e in self._cg.entities_by_id.values()}
        self._ids_by_name: Dict[str, List[str]] = {}
        for e in self._cg.entities():
            self._ids_by_name.setdefault(e.name.lower(), []).append(e.id)

    def entities(self) -> Iterable[Entity]:
        if self._ent_cache is None:
            out: List[Entity] = []
            for cg in self._cg.entities():
                uri = _entity_uri(self.program_id, cg)
                out.append(Entity(uri=uri, id=cg.id, kind=cg.kind if cg.kind != "test_module" else "module", name=cg.name, owner=cg.owner or None, labels=None))
            self._ent_cache = out
        return list(self._ent_cache)

    def edges(self) -> Iterable[Edge]:
        if self._edge_cache is None:
            out: List[Edge] = []
            seen: Set[Tuple[str, str, str]] = set()
            for ce in self._cg.edges():
                key = (ce.src, ce.dst, ce.type)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Edge(src=ce.src, dst=ce.dst, type=ce.type, meta=None))
            self._edge_cache = out
        return list(self._edge_cache)

    def search_refs(self, token: str) -> Iterable[Tuple[EntityId, Span]]:
        t = (token or "").strip()
        if not t:
            return []
        ids = self._cg.find_identifier_ids(t) or []
        out: List[Tuple[EntityId, Span]] = []
        for eid in ids:
            cg = self._ent_by_id.get(eid)
            if not cg:
                continue
            out.append((eid, Span(start_line=int(cg.start_line), end_line=int(cg.end_line))))
        return out

    def artifacts(self, kind: str) -> Iterable[Artifact]:
        if kind not in ("source", "artifact"):
            return []
        out: List[Artifact] = []
        seen: Set[str] = set()
        for e in self._cg.entities():
            fp = e.file
            if fp in seen:
                continue
            seen.add(fp)
            rel = os.path.relpath(fp, self.repo_root).replace("\\", "/")
            art_uri = artifact_uri(self.program_id, rel)
            out.append(Artifact(uri=art_uri, type="source", hash=self._cg.file_hash(fp), span=None))
        return out

    def _resolve_entity_uri(self, kind: str, resource: str, span: Optional[Tuple[int, int]]) -> ResolvedAnchor:
        if kind in ("module", "function", "class"):
            ent_id = f"py:{resource}"
        else:
            ent_id = f"py:{resource}"
        base = self._ent_by_id.get(ent_id)
        if not base:
            raise KeyError(f"entity not found for uri resource: {resource}")
        abs_fp = base.file
        a = int(span[0]) if span else int(base.start_line)
        b = int(span[1]) if span else int(base.end_line)
        rel = os.path.relpath(abs_fp, self.repo_root).replace("\\", "/")
        art_uri = artifact_uri(self.program_id, rel)
        return ResolvedAnchor(artifact_uri=art_uri, span=Span(start_line=a, end_line=b), hash=self._cg.file_hash(abs_fp))

    def subgraph(self, seeds: List[EntityId], radius: int) -> "PythonRepoGraph":
        if not seeds or radius <= 0:
            return self
        # Build adjacency over current edge view
        adj: Dict[str, List[str]] = {}
        for e in self.edges():
            adj.setdefault(e.src, []).append(e.dst)
            adj.setdefault(e.dst, []).append(e.src)
        cur = set(seeds)
        seen = set(cur)
        for _ in range(max(1, radius)):
            nxt: Set[str] = set()
            for s in list(cur):
                for nb in adj.get(s, []):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.add(nb)
            cur = nxt
        return _FilteredPythonRepoGraph(self, include_ids=seen)


class _FilteredPythonRepoGraph(PythonRepoGraph):
    def __init__(self, base: PythonRepoGraph, include_ids: Set[str]):
        # Shallow copy of references; restrict to include_ids
        self.repo_root = base.repo_root
        self.program_id = base.program_id
        self._cg = base._cg
        self._ent_by_id = base._ent_by_id
        self._ids_by_name = base._ids_by_name
        self._include_ids = set(include_ids)
        self._ent_cache = None
        self._edge_cache = None

    def entities(self) -> Iterable[Entity]:
        out: List[Entity] = []
        for cg in self._cg.entities():
            if cg.id not in self._include_ids:
                continue
            uri = _entity_uri(self.program_id, cg)
            out.append(Entity(uri=uri, id=cg.id, kind=cg.kind if cg.kind != "test_module" else "module", name=cg.name, owner=cg.owner or None, labels=None))
        return out

    def edges(self) -> Iterable[Edge]:
        out: List[Edge] = []
        inc = self._include_ids
        for ce in self._cg.edges():
            if (ce.src in inc) and (ce.dst in inc):
                out.append(Edge(src=ce.src, dst=ce.dst, type=ce.type, meta=None))
        return out



