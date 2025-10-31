from dataclasses import dataclass
from typing import Literal


@dataclass
class RAGConfig:
    splitter: Literal["basic"] = "basic"
    embedder: Literal["avg"] = "avg"
    store: Literal["flat"] = "flat"
    retriever: Literal["topk"] = "topk"
    reranker: Literal["none", "dot"] = "none"
    k: int = 4
    dim: int = 768


