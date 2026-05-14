from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from interpret.activation_cache import CaptureSpec
from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter, resolve_model_score
from interpret.tracer import ActivationTracer


def _coerce_pair(
    model: nn.Module,
    clean_inputs: Optional[ModelInputs],
    corrupted_inputs: Optional[ModelInputs],
    clean_input_ids: Optional[torch.Tensor],
    corrupted_input_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> tuple[ModelInputs, ModelInputs]:
    if clean_inputs is None:
        clean_inputs = coerce_model_inputs(model, input_ids=clean_input_ids, attention_mask=attention_mask)
    if corrupted_inputs is None:
        corrupted_inputs = coerce_model_inputs(model, input_ids=corrupted_input_ids, attention_mask=attention_mask)
    return clean_inputs, corrupted_inputs


def module_attribution_patching(
    model: nn.Module,
    *,
    candidate_modules: Iterable[str],
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> dict[str, object]:
    """First-order attribution patching estimate for module outputs.

    Score estimate per module is ``(clean_activation - corrupted_activation) * grad``
    summed over the module output. It is a fast approximation to activation
    patching and is useful for ranking many candidate nodes before exact patching.
    """

    adapter = get_model_adapter(model)
    names = [name for name in candidate_modules if name]
    clean_inputs, corrupted_inputs = _coerce_pair(model, clean_inputs, corrupted_inputs, clean_input_ids, corrupted_input_ids, attention_mask)

    clean_tracer = ActivationTracer(model, spec=CaptureSpec(move_to_cpu=False))
    clean_tracer.add_modules(names)
    with torch.no_grad():
        with clean_tracer.trace() as clean_cache:
            _ = adapter.forward(clean_inputs)

    modules = dict(model.named_modules())
    corrupted_acts: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(name: str):
        def _hook(_module: nn.Module, _inputs, output):
            if isinstance(output, torch.Tensor):
                if output.requires_grad:
                    output.retain_grad()
                corrupted_acts[name] = output
            return output

        return _hook

    for name in names:
        module = modules.get(name)
        if module is not None:
            handles.append(module.register_forward_hook(_make_hook(name)))
    try:
        model.zero_grad(set_to_none=True)
        outputs = adapter.forward(corrupted_inputs)
        score, target_token_id, target_feature_index = resolve_model_score(
            model,
            outputs,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        score.backward()
    finally:
        for handle in handles:
            handle.remove()

    estimates = torch.zeros(len(names), dtype=torch.float32)
    for idx, name in enumerate(names):
        clean_act = clean_cache.get(name)
        corrupted_act = corrupted_acts.get(name)
        grad = corrupted_act.grad if isinstance(corrupted_act, torch.Tensor) else None
        if clean_act is None or corrupted_act is None or grad is None or clean_act.shape != corrupted_act.shape:
            continue
        delta = clean_act.to(device=corrupted_act.device, dtype=corrupted_act.dtype) - corrupted_act.detach()
        estimates[idx] = float((delta * grad).sum().detach().cpu().item())
    model.zero_grad(set_to_none=True)
    return {
        "names": names,
        "scores": estimates,
        "target_token_id": target_token_id,
        "target_feature_index": target_feature_index,
    }


def summarize_attribution_patching(result: dict[str, object], *, topk: int = 10) -> list[dict[str, object]]:
    names = list(result.get("names", []))
    scores = result.get("scores")
    if not isinstance(scores, torch.Tensor):
        return []
    order = torch.argsort(scores.abs(), descending=True)[: int(topk)]
    return [{"module": names[int(idx)], "score": float(scores[int(idx)].item())} for idx in order]
