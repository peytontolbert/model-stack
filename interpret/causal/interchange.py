from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from interpret.activation_cache import CaptureSpec
from interpret.causal.patching import output_patching
from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter, resolve_model_score
from interpret.tracer import ActivationTracer


@torch.inference_mode()
def interchange_intervention_effect(
    model: nn.Module,
    *,
    source_inputs: Optional[ModelInputs] = None,
    base_inputs: Optional[ModelInputs] = None,
    source_input_ids: Optional[torch.Tensor] = None,
    base_input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    module_name: str,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> dict[str, float | int | None]:
    """Patch one source example's module output into a base example."""

    adapter = get_model_adapter(model)
    source_inputs = source_inputs or coerce_model_inputs(model, input_ids=source_input_ids, attention_mask=attention_mask)
    base_inputs = base_inputs or coerce_model_inputs(model, input_ids=base_input_ids, attention_mask=attention_mask)
    tracer = ActivationTracer(model, spec=CaptureSpec(move_to_cpu=False))
    tracer.add_modules([module_name])
    with tracer.trace() as cache:
        source_outputs = adapter.forward(source_inputs)
    source_value = cache.get(module_name)
    if source_value is None:
        raise KeyError(f"Module was not captured: {module_name}")
    base_outputs = adapter.forward(base_inputs)
    source_score, target_token_id, target_feature_index = resolve_model_score(
        model,
        source_outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    base_score, _, _ = resolve_model_score(
        model,
        base_outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    with output_patching(model, {module_name: source_value}):
        patched_outputs = adapter.forward(base_inputs)
    patched_score, _, _ = resolve_model_score(
        model,
        patched_outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    denom = (source_score - base_score).abs().clamp_min(1e-8)
    return {
        "source_score": float(source_score.detach().cpu().item()),
        "base_score": float(base_score.detach().cpu().item()),
        "patched_score": float(patched_score.detach().cpu().item()),
        "interchange_fraction": float(((patched_score - base_score) / denom).detach().cpu().item()),
        "target_token_id": target_token_id,
        "target_feature_index": target_feature_index,
    }
