from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from interpret.activation_cache import CaptureSpec
from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter
from interpret.tracer import ActivationTracer


def flatten_representation(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        return x.reshape(-1, 1).float()
    return x.reshape(-1, x.shape[-1]).float()


def centered_kernel_alignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = flatten_representation(x)
    y_flat = flatten_representation(y)
    if x_flat.shape[0] != y_flat.shape[0]:
        raise ValueError(f"Representations must have same row count, got {x_flat.shape[0]} and {y_flat.shape[0]}")
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    y_centered = y_flat - y_flat.mean(dim=0, keepdim=True)
    dot_xy = torch.linalg.matrix_norm(x_centered.T @ y_centered, ord="fro") ** 2
    dot_xx = torch.linalg.matrix_norm(x_centered.T @ x_centered, ord="fro")
    dot_yy = torch.linalg.matrix_norm(y_centered.T @ y_centered, ord="fro")
    denom = (dot_xx * dot_yy).clamp_min(1e-12)
    return dot_xy / denom


def representation_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = flatten_representation(x).mean(dim=0)
    y_flat = flatten_representation(y).mean(dim=0)
    if x_flat.numel() != y_flat.numel():
        raise ValueError("Mean representations must have the same width")
    return torch.nn.functional.cosine_similarity(x_flat, y_flat, dim=0)


def cache_representation_similarity(
    cache_a,
    cache_b,
    keys: Optional[Iterable[str]] = None,
) -> dict[str, dict[str, float]]:
    selected = list(keys) if keys is not None else sorted(set(cache_a.keys()) & set(cache_b.keys()))
    out: dict[str, dict[str, float]] = {}
    for key in selected:
        a = cache_a.get(key)
        b = cache_b.get(key)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            continue
        if a.shape[:-1] != b.shape[:-1]:
            continue
        out[key] = {
            "cka": float(centered_kernel_alignment(a, b).detach().cpu().item()),
            "cosine": float(representation_cosine_similarity(a, b).detach().cpu().item()) if a.shape[-1] == b.shape[-1] else 0.0,
            "l2_delta": float((flatten_representation(a) - flatten_representation(b)).norm().detach().cpu().item()) if a.shape == b.shape else 0.0,
        }
    return out


@torch.inference_mode()
def compare_model_representations(
    model_a: nn.Module,
    model_b: nn.Module,
    *,
    module_names: Iterable[str],
    inputs: Optional[ModelInputs] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[str, dict[str, float]]:
    names = [name for name in module_names if name]
    adapter_a = get_model_adapter(model_a)
    adapter_b = get_model_adapter(model_b)
    inputs_a = inputs or coerce_model_inputs(model_a, input_ids=input_ids, attention_mask=attention_mask)
    inputs_b = inputs or coerce_model_inputs(model_b, input_ids=input_ids, attention_mask=attention_mask)
    tracer_a = ActivationTracer(model_a, spec=CaptureSpec(move_to_cpu=True))
    tracer_b = ActivationTracer(model_b, spec=CaptureSpec(move_to_cpu=True))
    tracer_a.add_modules(names)
    tracer_b.add_modules(names)
    with tracer_a.trace() as cache_a:
        _ = adapter_a.forward(inputs_a)
    with tracer_b.trace() as cache_b:
        _ = adapter_b.forward(inputs_b)
    return cache_representation_similarity(cache_a, cache_b, names)
