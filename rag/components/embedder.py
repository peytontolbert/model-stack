import torch
from typing import Iterable, List, Tuple


class AvgEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = int(dim)

    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        # Deterministic bag-of-chars embedding for scaffolding
        embs: List[torch.Tensor] = []
        for t in texts:
            if not t:
                embs.append(torch.zeros(self.dim))
                continue
            vals = torch.tensor([ord(c) % 251 for c in t], dtype=torch.float32)
            avg = vals.mean()
            vec = torch.linspace(0, 1, self.dim, dtype=torch.float32) * avg
            embs.append(vec)
        return torch.stack(embs, dim=0)


