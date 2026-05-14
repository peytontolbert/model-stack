from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional

import torch
import torch.nn as nn

from interpret.model_adapter import coerce_model_inputs, get_model_adapter, resolve_model_score


def concept_direction_from_means(positive: torch.Tensor, negative: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    pos = positive.float().reshape(-1, positive.shape[-1]).mean(dim=0)
    neg = negative.float().reshape(-1, negative.shape[-1]).mean(dim=0)
    direction = pos - neg
    if normalize:
        direction = direction / direction.norm().clamp_min(1e-12)
    return direction


def erase_direction(x: torch.Tensor, direction: torch.Tensor, *, strength: float = 1.0) -> torch.Tensor:
    d = direction.to(device=x.device, dtype=x.dtype)
    d = d / d.norm().clamp_min(1e-12)
    projection = (x * d).sum(dim=-1, keepdim=True) * d
    return x - float(strength) * projection


def boost_direction(x: torch.Tensor, direction: torch.Tensor, *, strength: float = 1.0) -> torch.Tensor:
    d = direction.to(device=x.device, dtype=x.dtype)
    d = d / d.norm().clamp_min(1e-12)
    return x + float(strength) * d


@contextmanager
def patch_module_direction(
    model: nn.Module,
    module_name: str,
    direction: torch.Tensor,
    *,
    mode: str = "erase",
    strength: float = 1.0,
):
    modules = dict(model.named_modules())
    module = modules.get(module_name)
    if module is None:
        raise KeyError(f"Unknown module: {module_name}")

    def _hook(_module: nn.Module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            return output
        if mode == "erase":
            return erase_direction(output, direction, strength=strength)
        if mode == "boost":
            return boost_direction(output, direction, strength=strength)
        raise ValueError("mode must be 'erase' or 'boost'")

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def concept_direction_effect(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    module_name: str,
    direction: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    mode: str = "erase",
    strength: float = 1.0,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> dict[str, float | int | None]:
    adapter = get_model_adapter(model)
    inputs = coerce_model_inputs(model, input_ids=input_ids, attention_mask=attention_mask)
    base_outputs = adapter.forward(inputs)
    base_score, target_token_id, target_feature_index = resolve_model_score(
        model,
        base_outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    with patch_module_direction(model, module_name, direction, mode=mode, strength=strength):
        patched_outputs = adapter.forward(inputs)
    patched_score, _, _ = resolve_model_score(
        model,
        patched_outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    return {
        "base_score": float(base_score.detach().cpu().item()),
        "patched_score": float(patched_score.detach().cpu().item()),
        "delta": float((patched_score - base_score).detach().cpu().item()),
        "target_token_id": target_token_id,
        "target_feature_index": target_feature_index,
    }
