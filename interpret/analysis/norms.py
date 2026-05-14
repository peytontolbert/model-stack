from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from interpret.activation_cache import CaptureSpec
from interpret.tracer import ActivationTracer


def tensor_norm_summary(x: torch.Tensor, *, dim: int = -1) -> dict[str, float | tuple[int, ...]]:
    norms = x.float().norm(dim=dim) if x.ndim > 0 else x.float().abs()
    return {
        "shape": tuple(x.shape),
        "mean": float(norms.mean().item()),
        "std": float(norms.std(unbiased=False).item()),
        "min": float(norms.min().item()),
        "max": float(norms.max().item()),
    }


def module_parameter_norms(model: nn.Module, *, module_names: Iterable[str] | None = None) -> dict[str, dict[str, float]]:
    wanted = set(module_names) if module_names is not None else None
    out: dict[str, dict[str, float]] = {}
    for name, module in model.named_modules():
        if wanted is not None and name not in wanted:
            continue
        params = [p.detach().float().flatten() for p in module.parameters(recurse=False) if p is not None]
        if not params:
            continue
        flat = torch.cat(params)
        out[name] = {
            "numel": float(flat.numel()),
            "l2": float(flat.norm().item()),
            "mean_abs": float(flat.abs().mean().item()),
            "max_abs": float(flat.abs().max().item()),
        }
    return out


@torch.inference_mode()
def module_activation_norms(
    model: nn.Module,
    input_args: tuple = (),
    input_kwargs: dict | None = None,
    *,
    module_names: Iterable[str],
) -> dict[str, dict[str, float | tuple[int, ...]]]:
    tracer = ActivationTracer(model, spec=CaptureSpec(move_to_cpu=True))
    names = [name for name in module_names if name]
    tracer.add_modules(names)
    with tracer.trace() as cache:
        _ = model(*input_args, **(input_kwargs or {}))
    return {name: tensor_norm_summary(value) for name in names if isinstance((value := cache.get(name)), torch.Tensor)}


def residual_update_ratio(residual_pre: torch.Tensor, residual_post: torch.Tensor) -> torch.Tensor:
    if residual_pre.shape != residual_post.shape:
        raise ValueError("residual_pre and residual_post must have the same shape")
    update = residual_post.float() - residual_pre.float()
    return update.norm(dim=-1) / residual_pre.float().norm(dim=-1).clamp_min(1e-12)
