from __future__ import annotations

import torch
import torch.nn as nn


def embedding_norms(embedding: nn.Embedding | torch.Tensor) -> torch.Tensor:
    weight = embedding.weight if isinstance(embedding, nn.Embedding) else embedding
    return weight.detach().float().norm(dim=-1)


def embedding_anisotropy(embedding: nn.Embedding | torch.Tensor) -> dict[str, float]:
    weight = embedding.weight if isinstance(embedding, nn.Embedding) else embedding
    centered = weight.detach().float() - weight.detach().float().mean(dim=0, keepdim=True)
    norms = centered.norm(dim=-1)
    mean_vec = weight.detach().float().mean(dim=0)
    return {
        "mean_norm": float(embedding_norms(weight).mean().item()),
        "centered_mean_norm": float(norms.mean().item()),
        "global_mean_vector_norm": float(mean_vec.norm().item()),
    }


def nearest_embedding_neighbors(
    embedding: nn.Embedding | torch.Tensor,
    token_id: int,
    *,
    topk: int = 10,
    exclude_self: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = embedding.weight if isinstance(embedding, nn.Embedding) else embedding
    vectors = torch.nn.functional.normalize(weight.detach().float(), dim=-1)
    query = vectors[int(token_id)]
    scores = vectors @ query
    if exclude_self and 0 <= int(token_id) < scores.numel():
        scores[int(token_id)] = -float("inf")
    return torch.topk(scores, k=min(int(topk), scores.numel()))


def token_embedding_similarity(embedding: nn.Embedding | torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    weight = embedding.weight if isinstance(embedding, nn.Embedding) else embedding
    vecs = torch.nn.functional.normalize(weight[token_ids].float(), dim=-1)
    return vecs @ vecs.transpose(-1, -2)


def embedding_projection_scores(embedding: nn.Embedding | torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    weight = embedding.weight if isinstance(embedding, nn.Embedding) else embedding
    d = direction.to(device=weight.device, dtype=weight.dtype)
    d = d / d.norm().clamp_min(1e-12)
    return (weight.detach() * d).sum(dim=-1)
