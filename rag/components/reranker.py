import torch
from typing import List, Tuple


class NoOpReranker:
    def rerank(self, pairs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        return pairs


class DotReranker:
    def __init__(self, embedder):
        self.embedder = embedder

    def rerank(self, query: str, ids_and_scores: List[Tuple[str, float]], id_to_text) -> List[Tuple[str, float]]:
        if not ids_and_scores:
            return ids_and_scores
        q = self.embedder.encode([query])[0]
        # Replace scores with dot product against re-embedded doc
        scored: List[Tuple[str, float]] = []
        for i, _ in ids_and_scores:
            d = self.embedder.encode([id_to_text(i)])[0]
            s = float(torch.dot(q, d))
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


