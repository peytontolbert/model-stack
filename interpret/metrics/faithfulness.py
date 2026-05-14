from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from interpret.model_adapter import coerce_model_inputs, get_model_adapter, resolve_model_score


def _as_row_scores(token_scores: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    scores = token_scores.detach().float()
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if tuple(scores.shape) != tuple(input_ids.shape):
        raise ValueError(f"token_scores must match input_ids shape {tuple(input_ids.shape)}, got {tuple(scores.shape)}")
    return scores


def _score_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    position: int,
    target_token_id: Optional[int],
    target_feature_index: Optional[int],
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> tuple[torch.Tensor, Optional[int], Optional[int]]:
    adapter = get_model_adapter(model)
    inputs = coerce_model_inputs(model, input_ids=input_ids, attention_mask=attention_mask)
    outputs = adapter.forward(inputs)
    return resolve_model_score(
        model,
        outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )


@torch.inference_mode()
def token_deletion_insertion_curves(
    model: nn.Module,
    input_ids: torch.Tensor,
    token_scores: torch.Tensor,
    *,
    baseline_token_id: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    steps: Optional[list[int]] = None,
) -> dict[str, torch.Tensor | int | None]:
    """Compute deletion/insertion curves for token-level explanations.

    Deletion replaces the highest-scored tokens with ``baseline_token_id``.
    Insertion starts from the all-baseline sequence and restores high-scored
    tokens. Curves include step 0 and the full sequence by default.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B,T]")
    scores = _as_row_scores(token_scores, input_ids)
    batch_size, seq_len = input_ids.shape
    if batch_size != 1:
        raise ValueError("token_deletion_insertion_curves currently supports batch size 1")
    if steps is None:
        steps = list(range(seq_len + 1))
    steps = sorted({max(0, min(seq_len, int(step))) for step in steps})

    original_score, target_token_id, target_feature_index = _score_model(
        model,
        input_ids,
        attention_mask,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    order = torch.argsort(scores[0], descending=True)
    baseline = torch.full_like(input_ids, int(baseline_token_id))

    deletion_scores: list[torch.Tensor] = []
    insertion_scores: list[torch.Tensor] = []
    for step in steps:
        selected = order[:step]
        deleted = input_ids.clone()
        deleted[0, selected] = int(baseline_token_id)
        inserted = baseline.clone()
        inserted[0, selected] = input_ids[0, selected]

        deletion_score, _, _ = _score_model(
            model,
            deleted,
            attention_mask,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        insertion_score, _, _ = _score_model(
            model,
            inserted,
            attention_mask,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        deletion_scores.append(deletion_score.detach().cpu())
        insertion_scores.append(insertion_score.detach().cpu())

    return {
        "steps": torch.tensor(steps, dtype=torch.long),
        "order": order.detach().cpu(),
        "original_score": original_score.detach().cpu(),
        "deletion": torch.stack(deletion_scores),
        "insertion": torch.stack(insertion_scores),
        "target_token_id": target_token_id,
        "target_feature_index": target_feature_index,
    }


def area_over_curve(curve: torch.Tensor) -> torch.Tensor:
    if curve.numel() <= 1:
        return curve.new_tensor(0.0)
    return torch.trapz(curve.float(), dx=1.0 / float(curve.numel() - 1))


def faithfulness_summary(curves: dict[str, torch.Tensor | int | None]) -> dict[str, float]:
    deletion = curves["deletion"]
    insertion = curves["insertion"]
    original = curves["original_score"]
    if not isinstance(deletion, torch.Tensor) or not isinstance(insertion, torch.Tensor) or not isinstance(original, torch.Tensor):
        raise TypeError("curves must come from token_deletion_insertion_curves")
    return {
        "deletion_auc": float(area_over_curve(deletion).item()),
        "insertion_auc": float(area_over_curve(insertion).item()),
        "comprehensiveness": float((original - deletion[-1]).item()),
        "sufficiency_gap": float((original - insertion[-1]).item()),
    }


def spearman_rank_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.detach().float().flatten()
    b_flat = b.detach().float().flatten()
    if a_flat.numel() != b_flat.numel():
        raise ValueError("rank correlation inputs must have the same number of elements")
    if a_flat.numel() == 0:
        return torch.tensor(0.0)
    ar = torch.argsort(torch.argsort(a_flat))
    br = torch.argsort(torch.argsort(b_flat))
    ar_f = ar.float() - ar.float().mean()
    br_f = br.float() - br.float().mean()
    denom = ar_f.norm() * br_f.norm()
    if float(denom.item()) == 0.0:
        return torch.tensor(0.0)
    return (ar_f * br_f).sum() / denom
