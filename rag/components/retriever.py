import torch
from typing import List, Tuple


class TopKRetriever:
    def __init__(self, store, embedder, k: int = 4):
        self.store = store
        self.embedder = embedder
        self.k = int(k)

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        q = self.embedder.encode([query])[0]
        return self.store.search(q, k=self.k)


