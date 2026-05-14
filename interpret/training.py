from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn


def snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def parameter_drift_summary(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, old in before.items():
        new = after.get(name)
        if not isinstance(new, torch.Tensor) or old.shape != new.shape:
            continue
        delta = new.detach().float() - old.detach().float()
        base = old.detach().float()
        out[name] = {
            "delta_l2": float(delta.norm().item()),
            "param_l2": float(base.norm().item()),
            "relative_delta": float((delta.norm() / base.norm().clamp_min(1e-12)).item()),
            "max_abs_delta": float(delta.abs().max().item()),
        }
    return out


def gradient_norm_summary(model: nn.Module) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        data = param.detach().float()
        out[name] = {
            "grad_l2": float(grad.norm().item()),
            "param_l2": float(data.norm().item()),
            "grad_to_param_ratio": float((grad.norm() / data.norm().clamp_min(1e-12)).item()),
            "grad_max_abs": float(grad.abs().max().item()),
        }
    return out


def fisher_diagonal_from_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.grad.detach().float().pow(2).cpu() for name, param in model.named_parameters() if param.grad is not None}


@contextmanager
def capture_activation_gradients(model: nn.Module, module_names: Iterable[str]):
    modules = dict(model.named_modules())
    records: dict[str, dict[str, torch.Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_forward_hook(name: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            if not isinstance(output, torch.Tensor):
                return output
            records.setdefault(name, {})["activation"] = output.detach()
            if output.requires_grad:
                output.retain_grad()

                def _grad_hook(grad: torch.Tensor, *, _name=name):
                    records.setdefault(_name, {})["gradient"] = grad.detach()

                output.register_hook(_grad_hook)
            return output

        return _hook

    for name in module_names:
        module = modules.get(name)
        if module is not None:
            handles.append(module.register_forward_hook(_make_forward_hook(name)))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def activation_gradient_summary(records: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, values in records.items():
        row: dict[str, float] = {}
        activation = values.get("activation")
        gradient = values.get("gradient")
        if isinstance(activation, torch.Tensor):
            row["activation_norm"] = float(activation.float().norm(dim=-1).mean().item()) if activation.ndim > 1 else float(activation.float().norm().item())
        if isinstance(gradient, torch.Tensor):
            row["gradient_norm"] = float(gradient.float().norm(dim=-1).mean().item()) if gradient.ndim > 1 else float(gradient.float().norm().item())
        if isinstance(activation, torch.Tensor) and isinstance(gradient, torch.Tensor) and activation.shape == gradient.shape:
            row["grad_x_activation"] = float((activation.float() * gradient.float()).sum().item())
        out[name] = row
    return out


def training_step_diagnostics(model: nn.Module, before: dict[str, torch.Tensor] | None = None) -> dict[str, object]:
    diagnostics: dict[str, object] = {"gradients": gradient_norm_summary(model), "fisher_diagonal": fisher_diagonal_from_grads(model)}
    if before is not None:
        diagnostics["parameter_drift"] = parameter_drift_summary(before, snapshot_parameters(model))
    return diagnostics
