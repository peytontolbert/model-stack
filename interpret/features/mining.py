from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from interpret.activation_cache import ActivationCache
from interpret.features.sae import SparseAutoencoder
from interpret.features.sae_ops import sae_encode


@dataclass(frozen=True)
class ActivationContext:
    batch_index: int
    time_index: int
    score: float
    tokens: List[int]


def _get_feature_tensor(cache: ActivationCache, key: str) -> torch.Tensor:
    x = cache.get(key)
    if x is None:
        raise KeyError(f"ActivationCache has no key '{key}'")
    if x.ndim != 3:
        raise ValueError("Expected cached tensor of shape [B,T,D]")
    return x


def _topk_flat(scores: torch.Tensor, k: int) -> List[Tuple[int, int, float]]:
    batch_size, seq_len = scores.shape
    flat = scores.reshape(-1)
    kk = min(int(k), flat.numel())
    if kk <= 0:
        return []
    topv, topi = torch.topk(flat, k=kk)
    out: List[Tuple[int, int, float]] = []
    for value, index in zip(topv.tolist(), topi.tolist()):
        out.append((int(index // seq_len), int(index % seq_len), float(value)))
    return out


def topk_positions(cache: ActivationCache, key: str, k: int = 20) -> List[Tuple[int, int, float]]:
    """Return top-k (batch_index, time_index, value) for the max channel per position."""
    x = _get_feature_tensor(cache, key)
    return _topk_flat(x.float().amax(dim=-1), k)


def topk_feature_positions(cache: ActivationCache, key: str, feature_index: int, k: int = 20) -> List[Tuple[int, int, float]]:
    """Return top-k positions for one feature channel."""
    x = _get_feature_tensor(cache, key)
    return _topk_flat(x[..., int(feature_index)].float(), k)


def feature_coactivation_matrix(cache: ActivationCache, key: str, *, normalize: bool = True) -> torch.Tensor:
    """Compute a dataset-scale feature co-activation matrix [D,D]."""
    feats = _get_feature_tensor(cache, key)
    x = feats.float().reshape(-1, feats.shape[-1])
    if x.numel() == 0:
        return torch.empty(0, 0)
    if normalize:
        x = x - x.mean(dim=0, keepdim=True)
        x = x / x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x.t() @ x) / max(int(x.shape[0]), 1)


def activation_contexts(
    input_ids: torch.Tensor,
    positions: List[Tuple[int, int, float]],
    *,
    window: int = 8,
) -> List[ActivationContext]:
    """Attach token windows to mined activation positions."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B,T]")
    out: List[ActivationContext] = []
    batch_size, seq_len = input_ids.shape
    for batch_index, time_index, score in positions:
        if not (0 <= batch_index < batch_size and 0 <= time_index < seq_len):
            continue
        lo = max(0, int(time_index) - int(window))
        hi = min(seq_len, int(time_index) + int(window) + 1)
        out.append(
            ActivationContext(
                batch_index=int(batch_index),
                time_index=int(time_index),
                score=float(score),
                tokens=input_ids[int(batch_index), lo:hi].tolist(),
            )
        )
    return out


def search_feature_activations(
    cache: ActivationCache,
    key: str,
    feature_index: int,
    *,
    input_ids: torch.Tensor,
    k: int = 20,
    window: int = 8,
) -> List[ActivationContext]:
    """Return the highest-activating token contexts for one feature."""
    positions = topk_feature_positions(cache, key, feature_index, k=k)
    return activation_contexts(input_ids, positions, window=window)


def sae_feature_dashboard(
    cache: ActivationCache,
    key: str,
    sae: SparseAutoencoder,
    feature_index: int,
    *,
    input_ids: torch.Tensor,
    k: int = 20,
    window: int = 8,
) -> Dict[str, object]:
    """Small dataset-scale dashboard for one SAE feature."""
    x = _get_feature_tensor(cache, key)
    codes = sae_encode(sae, x)
    if codes.ndim != 3:
        raise ValueError("Expected SAE codes with shape [B,T,C]")
    positions = _topk_flat(codes[..., int(feature_index)].float(), k)
    contexts = activation_contexts(input_ids, positions, window=window)
    flat_codes = codes.float().reshape(-1, codes.shape[-1])
    centered = flat_codes - flat_codes.mean(dim=0, keepdim=True)
    normed = centered / centered.std(dim=0, keepdim=True).clamp_min(1e-6)
    coactivation = (normed.t() @ normed) / max(int(normed.shape[0]), 1)
    top_partners = []
    if coactivation.numel() > 0 and 0 <= int(feature_index) < coactivation.shape[0]:
        partner_scores, partner_idx = torch.topk(coactivation[int(feature_index)].float(), k=min(10, coactivation.shape[0]))
        top_partners = [(int(i), float(v)) for v, i in zip(partner_scores.tolist(), partner_idx.tolist()) if int(i) != int(feature_index)]
    return {
        "feature_index": int(feature_index),
        "top_contexts": contexts,
        "top_coactivations": top_partners,
        "mean_activation": float(codes[..., int(feature_index)].float().mean().item()),
        "max_activation": float(codes[..., int(feature_index)].float().amax().item()),
    }


__all__ = [
    "ActivationContext",
    "activation_contexts",
    "feature_coactivation_matrix",
    "sae_feature_dashboard",
    "search_feature_activations",
    "topk_feature_positions",
    "topk_positions",
]
