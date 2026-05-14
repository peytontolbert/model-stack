from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn


def _unembed_weight(lm_head: nn.Module | torch.Tensor) -> torch.Tensor:
    if isinstance(lm_head, torch.Tensor):
        return lm_head
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("lm_head must be a tensor or module with a weight tensor")
    return weight


def unembed_vector_scores(
    vectors: torch.Tensor,
    lm_head: nn.Module | torch.Tensor,
    *,
    target_token_id: Optional[int] = None,
    baseline_token_id: Optional[int] = None,
) -> torch.Tensor:
    weight = _unembed_weight(lm_head).to(device=vectors.device, dtype=vectors.dtype)
    if target_token_id is None:
        logits = vectors @ weight.T
        return logits
    direction = weight[int(target_token_id)]
    if baseline_token_id is not None:
        direction = direction - weight[int(baseline_token_id)]
    return (vectors * direction).sum(dim=-1)


def logit_prism_components(
    components: Mapping[str, torch.Tensor],
    lm_head: nn.Module | torch.Tensor,
    *,
    position: int = -1,
    target_token_id: Optional[int] = None,
    baseline_token_id: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, value in components.items():
        if not isinstance(value, torch.Tensor):
            continue
        vec = value[:, position, :] if value.ndim == 3 else value
        out[name] = unembed_vector_scores(vec, lm_head, target_token_id=target_token_id, baseline_token_id=baseline_token_id).detach().cpu()
    return out


def summarize_logit_prism(prism: dict[str, torch.Tensor], *, topk: int = 10) -> list[dict[str, float | str]]:
    rows = []
    for name, score in prism.items():
        scalar = score.float().mean()
        rows.append({"component": name, "score": float(scalar.item())})
    rows.sort(key=lambda row: abs(float(row["score"])), reverse=True)
    return rows[: int(topk)]
