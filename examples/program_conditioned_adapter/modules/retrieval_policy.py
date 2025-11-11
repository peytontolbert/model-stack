from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional, List, Set
import math

from .program_graph import ProgramGraph, Entity, Edge, Span


@dataclass
class RetrievalMix:
    sim: float = 0.6
    struct: float = 0.4
    temp: float = 0.7

    @staticmethod
    def parse(spec: Optional[str], temp: Optional[float]) -> "RetrievalMix":
        if not spec:
            return RetrievalMix(temp=(temp if isinstance(temp, (int, float)) else 0.7))
        w_sim = 0.0
        w_struct = 0.0
        try:
            parts = [p.strip() for p in str(spec).split(",") if p.strip()]
            for p in parts:
                if ":" not in p:
                    continue
                k, v = p.split(":", 1)
                if k.strip() == "sim":
                    w_sim = float(v)
                elif k.strip() == "struct":
                    w_struct = float(v)
        except Exception:
            w_sim, w_struct = 0.6, 0.4
        # normalize weights
        s = max(1e-6, (w_sim + w_struct))
        w_sim /= s
        w_struct /= s
        return RetrievalMix(sim=w_sim, struct=w_struct, temp=(temp if isinstance(temp, (int, float)) else 0.7))


class RetrievalPolicy:
    def __init__(self, mix: RetrievalMix):
        self.mix = mix

    @staticmethod
    def from_spec(policy: Optional[str], temp: Optional[float] = None) -> "RetrievalPolicy":
        return RetrievalPolicy(RetrievalMix.parse(policy, temp))

    def score_entities(self, query: str, pg: ProgramGraph) -> Dict[str, float]:
        # sim score: token overlap against entity name (very light BM25-ish)
        q_tokens = _tokenize(query)
        entities = list(pg.entities())
        sim: Dict[str, float] = {}
        for e in entities:
            name_tokens = _tokenize(e.name)
            overlap = len(q_tokens & name_tokens)
            sim[e.id] = float(overlap)
        # struct score: graph distance from top sim seeds (k=8), shorter distance => higher score
        seeds = sorted(entities, key=lambda x: sim.get(x.id, 0.0), reverse=True)[:8]
        seed_ids = [e.id for e in seeds if sim.get(e.id, 0.0) > 0]
        dist = _bfs_distance(seed_ids, pg)
        struct: Dict[str, float] = {}
        for e in entities:
            d = dist.get(e.id, None)
            if d is None:
                struct[e.id] = 0.0
            else:
                # invert distance with light decay; d=0 => 1.0
                struct[e.id] = 1.0 / float(1 + d)
        # blend and softmax with temperature
        raw: Dict[str, float] = {}
        for e in entities:
            raw[e.id] = self.mix.sim * sim.get(e.id, 0.0) + self.mix.struct * struct.get(e.id, 0.0)
        return _softmax(raw, temperature=max(1e-3, float(self.mix.temp)))


def _tokenize(text: str) -> Set[str]:
    import re
    toks = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "")]
    return set(t for t in toks if len(t) > 1)


def _bfs_distance(seeds: List[str], pg: ProgramGraph, max_depth: int = 3) -> Dict[str, int]:
    # build adjacency from edges()
    adj: Dict[str, List[str]] = {}
    for e in pg.edges():
        adj.setdefault(e.src, []).append(e.dst)
        adj.setdefault(e.dst, []).append(e.src)
    dist: Dict[str, int] = {}
    cur = list(seeds)
    for s in seeds:
        dist[s] = 0
    depth = 0
    while cur and depth < max_depth:
        nxt: List[str] = []
        for u in cur:
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        cur = nxt
        depth += 1
    return dist


def _softmax(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
    # numerically stable softmax over values
    vals = list(scores.values())
    if not vals:
        return {}
    m = max(vals)
    exps = {k: math.exp((v - m) / temperature) for k, v in scores.items()}
    s = sum(exps.values()) or 1.0
    return {k: (v / s) for k, v in exps.items()}


