from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import time
import re

from .program_graph import ProgramGraph, Span, ResolvedAnchor


@dataclass
class CitationPolicy:
    enforce: bool = True
    per_paragraph: bool = False
    repair: bool = True


def _extract_identifier_tokens(text: str) -> List[str]:
    toks = re.findall(r"`([A-Za-z_][A-Za-z0-9_\.]+)`", text or "")
    toks2 = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text or "")
    out: List[str] = []
    for t in toks + toks2:
        tt = t.split(".")[-1]
        if tt and tt not in out:
            out.append(tt)
    return out[:24]


class CitationManager:
    def __init__(self, policy: CitationPolicy, pg: ProgramGraph, manifest: Dict[str, Any]):
        self.policy = policy
        self.pg = pg
        self.manifest = manifest or {}

    def collect(self, region_entity_ids: List[str], contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Best-effort: attach evidence from region seeds (first-class Program URIs would be passed by caller)
        evidence: List[Dict[str, Any]] = []
        for e in self.pg.entities():
            if e.id in region_entity_ids:
                try:
                    # Resolve existing entity URI to get artifact span/hash
                    ra = self.pg.resolve(e.uri)
                    evidence.append({
                        "uri": e.uri,
                        "artifact_hash": ra.hash,
                        "span": {"start": {"line": ra.span.start_line}, "end": {"line": ra.span.end_line}},
                        "kind": e.kind,
                        "confidence": 0.5,
                        "retrieval": {"score": 0.0, "features": {}},
                    })
                except Exception:
                    continue
        return evidence

    def enforce(self, draft_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.policy.enforce:
            return draft_units
        out = []
        for u in draft_units:
            ev = list(u.get("evidence") or [])
            if self.policy.per_paragraph:
                paras = [p for p in (u.get("text") or "").split("\n\n") if p.strip()]
                if not paras:
                    out.append(u)
                    continue
                # If any paragraph lacks evidence, attempt repair (or drop)
                repaired = self.repair(u) if self.policy.repair else u
                out.append(repaired)
            else:
                if not ev:
                    repaired = self.repair(u) if self.policy.repair else u
                    out.append(repaired)
                else:
                    out.append(u)
        return out

    def repair(self, unit: Dict[str, Any]) -> Dict[str, Any]:
        # Try anchoring based on identifier tokens
        text = str(unit.get("text") or "")
        tokens = _extract_identifier_tokens(text)
        anchors: List[Dict[str, Any]] = []
        for t in tokens:
            try:
                refs = list(self.pg.search_refs(t))
            except Exception:
                refs = []
            for (eid, sp) in refs[:2]:
                # find entity to get its canonical URI (if present)
                euri = None
                for e in self.pg.entities():
                    if e.id == eid:
                        euri = e.uri
                        break
                try:
                    if euri:
                        ra = self.pg.resolve(euri)
                        anchors.append({
                            "uri": euri,
                            "artifact_hash": ra.hash,
                            "span": {"start": {"line": ra.span.start_line}, "end": {"line": ra.span.end_line}},
                            "kind": (e.kind if euri and e else "entity"),
                            "confidence": 0.4,
                            "retrieval": {"score": 0.0, "features": {"repair": True}},
                        })
                except Exception:
                    continue
            if len(anchors) >= 4:
                break
        if anchors:
            unit = dict(unit)
            unit["evidence"] = anchors
        return unit

    def stamp_provenance(self, unit: Dict[str, Any]) -> Dict[str, Any]:
        prov = {
            "program_id": self.manifest.get("program_id"),
            "manifest_sha": self._manifest_sha(),
            "commit": self.manifest.get("commit"),
            "policy": {
                "enforce": bool(self.policy.enforce),
                "per_paragraph": bool(self.policy.per_paragraph),
                "repair": bool(self.policy.repair),
            },
            "ts": time.time(),
        }
        out = dict(unit)
        out["provenance"] = prov
        return out

    def _manifest_sha(self) -> Optional[str]:
        try:
            blob = str(self.manifest).encode("utf-8", errors="ignore")
            return hashlib.sha256(blob).hexdigest()
        except Exception:
            return None


