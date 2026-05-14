from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn

from interpret.activation_cache import ActivationCache, CaptureSpec


@dataclass(frozen=True)
class MoETarget:
    name: str
    module: nn.Module
    router: nn.Module
    num_experts: int
    k: int


def find_moe_targets(model: nn.Module) -> list[MoETarget]:
    out: list[MoETarget] = []
    for name, module in model.named_modules():
        router = getattr(module, "router", None)
        experts = getattr(module, "experts", None)
        num_experts = int(getattr(module, "num_experts", len(experts) if experts is not None else 0) or 0)
        if isinstance(router, nn.Module) and experts is not None and num_experts > 0:
            out.append(MoETarget(name=name, module=module, router=router, num_experts=num_experts, k=int(getattr(module, "k", 1))))
    return out


@contextmanager
def capture_moe_router_logits(model: nn.Module, *, spec: CaptureSpec | None = None):
    cache = ActivationCache()
    handles: list[torch.utils.hooks.RemovableHandle] = []
    spec = spec or CaptureSpec()
    for target in find_moe_targets(model):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, _target=target) -> None:
            if isinstance(output, torch.Tensor):
                cache.store(f"{_target.name}.router_logits", output, spec)

        handles.append(target.router.register_forward_hook(_hook))
    try:
        yield cache
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass


def expert_usage_from_logits(logits: torch.Tensor, *, k: int = 1) -> dict[str, torch.Tensor]:
    if logits.ndim < 2:
        raise ValueError("router logits must have expert dimension")
    num_experts = int(logits.shape[-1])
    top = torch.topk(logits.float(), k=min(int(k), num_experts), dim=-1).indices
    counts = torch.zeros(num_experts, dtype=torch.float32, device=logits.device)
    for expert_idx in range(num_experts):
        counts[expert_idx] = (top == expert_idx).float().sum()
    usage = counts / counts.sum().clamp_min(1.0)
    probs = torch.softmax(logits.float(), dim=-1).reshape(-1, num_experts).mean(dim=0)
    return {"counts": counts.detach().cpu(), "usage": usage.detach().cpu(), "mean_probs": probs.detach().cpu(), "top_indices": top.detach().cpu()}


def summarize_moe_router_usage(cache: ActivationCache, targets: Iterable[MoETarget]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in targets:
        logits = cache.get(f"{target.name}.router_logits")
        if logits is None:
            continue
        usage = expert_usage_from_logits(logits, k=target.k)
        rows.append(
            {
                "name": target.name,
                "num_experts": target.num_experts,
                "k": target.k,
                "usage": usage["usage"].tolist(),
                "mean_probs": usage["mean_probs"].tolist(),
            }
        )
    return rows
