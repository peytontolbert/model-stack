import torch
from typing import Dict, List, Tuple


class FlatStore:
    def __init__(self):
        self._ids: List[str] = []
        self._vecs: List[torch.Tensor] = []
        self._texts: Dict[str, str] = {}

    def add(self, ids: List[str], embs: torch.Tensor, texts: List[str]) -> None:
        assert embs.ndim == 2 and embs.shape[0] == len(ids) == len(texts)
        for i, emb, txt in zip(ids, embs, texts):
            self._ids.append(i)
            self._vecs.append(emb.detach().clone())
            self._texts[i] = txt

    def search(self, q: torch.Tensor, k: int = 4) -> List[Tuple[str, float]]:
        if not self._vecs:
            return []
        V = torch.stack(self._vecs, dim=0)
        sims = torch.nn.functional.cosine_similarity(q.unsqueeze(0), V, dim=1)
        vals, idx = torch.topk(sims, k=min(k, V.shape[0]))
        out: List[Tuple[str, float]] = []
        for v, j in zip(vals.tolist(), idx.tolist()):
            out.append((self._ids[j], float(v)))
        return out

    def text(self, doc_id: str) -> str:
        return self._texts[doc_id]


