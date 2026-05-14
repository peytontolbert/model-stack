from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from interpret.model_adapter import coerce_model_inputs, get_model_adapter, resolve_model_score


@torch.inference_mode()
def token_trigger_append_scan(
    model: nn.Module,
    input_ids: torch.Tensor,
    trigger_token_ids: Iterable[int],
    *,
    attention_mask: Optional[torch.Tensor] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> list[dict[str, float | int]]:
    adapter = get_model_adapter(model)
    base_inputs = coerce_model_inputs(model, input_ids=input_ids, attention_mask=attention_mask)
    base_outputs = adapter.forward(base_inputs)
    base_score, target_token_id, target_feature_index = resolve_model_score(
        model,
        base_outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    rows: list[dict[str, float | int]] = []
    for token_id in trigger_token_ids:
        trigger = torch.full((input_ids.shape[0], 1), int(token_id), dtype=input_ids.dtype, device=input_ids.device)
        changed = torch.cat([input_ids, trigger], dim=1)
        changed_mask = None
        if attention_mask is not None:
            changed_mask = torch.cat([attention_mask, torch.ones_like(trigger)], dim=1)
        outputs = adapter.forward(coerce_model_inputs(model, input_ids=changed, attention_mask=changed_mask))
        score, _, _ = resolve_model_score(
            model,
            outputs,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        rows.append(
            {
                "trigger_token_id": int(token_id),
                "score": float(score.detach().cpu().item()),
                "delta": float((score - base_score).detach().cpu().item()),
            }
        )
    rows.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return rows


@torch.inference_mode()
def token_trigger_position_scan(
    model: nn.Module,
    input_ids: torch.Tensor,
    trigger_token_id: int,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> list[dict[str, float | int]]:
    adapter = get_model_adapter(model)
    base_outputs = adapter.forward(coerce_model_inputs(model, input_ids=input_ids, attention_mask=attention_mask))
    base_score, target_token_id, target_feature_index = resolve_model_score(
        model,
        base_outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    rows: list[dict[str, float | int]] = []
    for pos in range(int(input_ids.shape[1])):
        changed = input_ids.clone()
        changed[:, pos] = int(trigger_token_id)
        outputs = adapter.forward(coerce_model_inputs(model, input_ids=changed, attention_mask=attention_mask))
        score, _, _ = resolve_model_score(
            model,
            outputs,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        rows.append({"position": int(pos), "score": float(score.detach().cpu().item()), "delta": float((score - base_score).detach().cpu().item())})
    rows.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return rows
