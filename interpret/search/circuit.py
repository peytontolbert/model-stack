from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from interpret.activation_cache import CaptureSpec
from interpret.causal.patching import output_patching
from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter, resolve_model_score
from interpret.tracer import ActivationTracer


def _resolve_inputs(
    model: nn.Module,
    inputs: Optional[ModelInputs],
    input_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> ModelInputs:
    if inputs is not None:
        return inputs
    return coerce_model_inputs(model, input_ids=input_ids, attention_mask=attention_mask)


@torch.inference_mode()
def module_recovery_scores(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    candidate_modules: Iterable[str],
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> dict[str, object]:
    adapter = get_model_adapter(model)
    clean_inputs = _resolve_inputs(model, clean_inputs, clean_input_ids, attention_mask)
    corrupted_inputs = _resolve_inputs(model, corrupted_inputs, corrupted_input_ids, attention_mask)
    candidates = [name for name in candidate_modules if name]

    tracer = ActivationTracer(model, spec=CaptureSpec(move_to_cpu=False))
    tracer.add_modules(candidates)
    with tracer.trace() as cache:
        clean_outputs = adapter.forward(clean_inputs)
        clean_replacements = {name: cache.get(name) for name in candidates if cache.get(name) is not None}

    corrupted_outputs = adapter.forward(corrupted_inputs)
    clean_score, target_token_id, target_feature_index = resolve_model_score(
        model,
        clean_outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    corrupted_score, _, _ = resolve_model_score(
        model,
        corrupted_outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    denom = (clean_score - corrupted_score).abs().clamp_min(1e-8)
    scores = torch.zeros(len(candidates), dtype=torch.float32)
    for idx, name in enumerate(candidates):
        replacement = clean_replacements.get(name)
        if replacement is None:
            continue
        with output_patching(model, {name: replacement}):
            patched_outputs = adapter.forward(corrupted_inputs)
        patched_score, _, _ = resolve_model_score(
            model,
            patched_outputs,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        scores[idx] = float(((patched_score - corrupted_score) / denom).detach().cpu().item())

    return {
        "names": candidates,
        "scores": scores,
        "clean_score": float(clean_score.detach().cpu().item()),
        "corrupted_score": float(corrupted_score.detach().cpu().item()),
        "target_token_id": target_token_id,
        "target_feature_index": target_feature_index,
        "clean_replacements": clean_replacements,
    }


@torch.inference_mode()
def greedy_module_circuit(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    candidate_modules: Iterable[str],
    k: int = 5,
    min_gain: float = 0.0,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> dict[str, object]:
    adapter = get_model_adapter(model)
    clean_inputs = _resolve_inputs(model, clean_inputs, clean_input_ids, attention_mask)
    corrupted_inputs = _resolve_inputs(model, corrupted_inputs, corrupted_input_ids, attention_mask)
    table = module_recovery_scores(
        model,
        clean_inputs=clean_inputs,
        corrupted_inputs=corrupted_inputs,
        candidate_modules=candidate_modules,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    candidates = list(table["names"])  # type: ignore[arg-type]
    replacements = dict(table["clean_replacements"])  # type: ignore[arg-type]
    clean_score_value = float(table["clean_score"])
    corrupted_score_value = float(table["corrupted_score"])
    denom = abs(clean_score_value - corrupted_score_value) + 1e-8

    selected: list[str] = []
    curve: list[float] = []
    remaining = set(candidates)
    current = 0.0
    for _ in range(int(k)):
        best_name: str | None = None
        best_score = current
        for name in sorted(remaining):
            active = {module_name: replacements[module_name] for module_name in selected + [name] if module_name in replacements}
            if not active:
                continue
            with output_patching(model, active):
                patched_outputs = adapter.forward(corrupted_inputs)
            patched_score, target_token_id, target_feature_index = resolve_model_score(
                model,
                patched_outputs,
                position=position,
                target_token_id=target_token_id,
                target_feature_index=target_feature_index,
                score_fn=score_fn,
            )
            recovery = (float(patched_score.detach().cpu().item()) - corrupted_score_value) / denom
            if recovery > best_score:
                best_name = name
                best_score = recovery
        if best_name is None or (best_score - current) < float(min_gain):
            break
        selected.append(best_name)
        remaining.remove(best_name)
        current = best_score
        curve.append(current)

    return {
        "selected": selected,
        "curve": curve,
        "individual": table["scores"],
        "candidate_modules": candidates,
        "clean_score": clean_score_value,
        "corrupted_score": corrupted_score_value,
    }


def summarize_module_circuit(result: dict[str, object], *, topk: int = 10) -> list[dict[str, object]]:
    names = list(result.get("candidate_modules", []))
    scores = result.get("individual")
    if not isinstance(scores, torch.Tensor):
        return []
    order = torch.argsort(scores, descending=True)[: int(topk)]
    selected = set(result.get("selected", []))
    return [
        {
            "module": names[int(idx)],
            "individual_recovery": float(scores[int(idx)].item()),
            "selected": names[int(idx)] in selected,
        }
        for idx in order
    ]
