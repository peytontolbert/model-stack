from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Optional, Tuple, Dict, List


# Core types
EntityId = str


@dataclass(frozen=True)
class Span:
    start_line: int
    end_line: int  # inclusive, 1-based


@dataclass(frozen=True)
class Entity:
    uri: str
    id: EntityId
    kind: str
    name: str
    owner: Optional[str] = None
    labels: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class Edge:
    src: EntityId
    dst: EntityId
    type: str
    meta: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class Artifact:
    uri: str
    type: str  # e.g., "source"
    hash: str
    span: Optional[Span] = None


@dataclass(frozen=True)
class ResolvedAnchor:
    artifact_uri: str
    span: Span
    hash: str


class ProgramGraph(Protocol):
    def entities(self) -> Iterable[Entity]: ...
    def edges(self) -> Iterable[Edge]: ...
    def search_refs(self, token: str) -> Iterable[Tuple[EntityId, Span]]: ...
    def subgraph(self, seeds: List[EntityId], radius: int) -> "ProgramGraph": ...
    def artifacts(self, kind: str) -> Iterable[Artifact]: ...
    def resolve(self, uri: str) -> ResolvedAnchor: ...


