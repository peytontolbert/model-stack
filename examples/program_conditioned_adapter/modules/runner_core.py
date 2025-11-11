from __future__ import annotations

from typing import Dict, List, Any, Tuple

from .program_graph import ProgramGraph, Entity
from .retrieval_policy import RetrievalPolicy
from .citations import CitationManager, CitationPolicy


def select_region(query: str, pg: ProgramGraph, policy: RetrievalPolicy, top_k: int = 16) -> List[str]:
    scores = policy.score_entities(query, pg)
    ids = sorted(scores.keys(), key=lambda k: scores.get(k, 0.0), reverse=True)[:max(1, int(top_k))]
    return ids


def prepare_citations(units: List[Dict[str, Any]], region_entity_ids: List[str], pg: ProgramGraph, citations_policy: Dict[str, Any], manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    cm = CitationManager(
        policy=CitationPolicy(
            enforce=bool(citations_policy.get("enforce", True)),
            per_paragraph=bool(citations_policy.get("per_paragraph", False)),
            repair=bool(citations_policy.get("repair", True)),
        ),
        pg=pg,
        manifest=manifest or {},
    )
    # Collect baseline evidence for the region and stamp
    baseline = cm.collect(region_entity_ids, contexts=[])
    out: List[Dict[str, Any]] = []
    for u in units:
        unit = dict(u)
        unit.setdefault("evidence", baseline[:4])
        unit = cm.enforce([unit])[0]
        unit = cm.stamp_provenance(unit)
        out.append(unit)
    return out


