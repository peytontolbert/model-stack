from __future__ import annotations

import torch


def feature_correlation_graph(source_features: torch.Tensor, receiver_features: torch.Tensor, *, topk: int = 20) -> dict[str, object]:
    """Build a sparse feature-to-feature graph from activation correlations."""

    src = source_features.float().reshape(-1, source_features.shape[-1])
    dst = receiver_features.float().reshape(-1, receiver_features.shape[-1])
    if src.shape[0] != dst.shape[0]:
        raise ValueError("source and receiver features must have the same row count")
    src = (src - src.mean(dim=0, keepdim=True)) / src.std(dim=0, keepdim=True).clamp_min(1e-6)
    dst = (dst - dst.mean(dim=0, keepdim=True)) / dst.std(dim=0, keepdim=True).clamp_min(1e-6)
    corr = src.T @ dst / max(1, src.shape[0] - 1)
    flat = corr.abs().flatten()
    k = min(int(topk), flat.numel())
    values, indices = torch.topk(flat, k=k)
    edges = []
    width = int(corr.shape[1])
    for value, index in zip(values, indices):
        src_idx = int(index.item() // width)
        dst_idx = int(index.item() % width)
        edges.append(
            {
                "source_feature": src_idx,
                "receiver_feature": dst_idx,
                "correlation": float(corr[src_idx, dst_idx].item()),
                "abs_correlation": float(value.item()),
            }
        )
    return {"matrix": corr.detach().cpu(), "edges": edges}


def feature_activation_jaccard(a: torch.Tensor, b: torch.Tensor, *, threshold: float = 0.0) -> torch.Tensor:
    active_a = a > float(threshold)
    active_b = b > float(threshold)
    intersection = (active_a & active_b).float().sum(dim=0)
    union = (active_a | active_b).float().sum(dim=0).clamp_min(1.0)
    return intersection / union


def summarize_feature_circuit(graph: dict[str, object], *, topk: int = 10) -> list[dict[str, object]]:
    edges = list(graph.get("edges", []))
    return edges[: int(topk)]
