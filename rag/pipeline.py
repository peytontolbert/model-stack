from typing import Iterable, List, Tuple

from .config import RAGConfig
from .components import (
    BasicSplitter,
    AvgEmbedder,
    FlatStore,
    TopKRetriever,
    NoOpReranker,
    DotReranker,
)


class RAGPipeline:
    def __init__(self, splitter, embedder, store, retriever, reranker):
        self.splitter = splitter
        self.embedder = embedder
        self.store = store
        self.retriever = retriever
        self.reranker = reranker

    @classmethod
    def from_config(cls, cfg: RAGConfig) -> "RAGPipeline":
        splitter = BasicSplitter()
        embedder = AvgEmbedder(dim=cfg.dim)
        store = FlatStore()
        retriever = TopKRetriever(store=store, embedder=embedder, k=cfg.k)
        reranker = NoOpReranker() if cfg.reranker == "none" else DotReranker(embedder)
        return cls(splitter, embedder, store, retriever, reranker)

    def index_texts(self, texts: Iterable[str], ids: Iterable[str] | None = None) -> None:
        docs = self.splitter.split(texts)
        ids_list: List[str] = list(ids) if ids is not None else [str(i) for i in range(len(docs))]
        embs = self.embedder.encode(docs)
        self.store.add(ids_list, embs, docs)

    def query(self, query: str, k: int | None = None) -> List[Tuple[str, float, str]]:
        pairs = self.retriever.retrieve(query)
        ranked = (
            pairs
            if isinstance(self.reranker, NoOpReranker)
            else self.reranker.rerank(query, pairs, self.store.text)
        )
        top = ranked if k is None else ranked[:k]
        return [(i, s, self.store.text(i)) for i, s in top]


