from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from interpret.activation_cache import CaptureSpec
from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter
from interpret.tracer import ActivationTracer


def block_internal_keys(model: nn.Module, *, stack: Optional[str] = None) -> list[str]:
    adapter = get_model_adapter(model)
    keys: list[str] = []
    for block in adapter.block_targets(stack=stack):
        keys.extend([block.name, f"{block.name}.n1", f"{block.name}.attn", f"{block.name}.n2", f"{block.name}.mlp"])
    return keys


@torch.inference_mode()
def block_internal_norms(
    model: nn.Module,
    *,
    inputs: Optional[ModelInputs] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    adapter = get_model_adapter(model)
    inputs = inputs or coerce_model_inputs(model, input_ids=input_ids, attention_mask=attention_mask)
    tracer = ActivationTracer(model, spec=CaptureSpec(move_to_cpu=True))
    tracer.add_residual_streams(stack=stack)
    names = block_internal_keys(model, stack=stack)
    tracer.add_modules(names)
    with tracer.trace() as cache:
        _ = adapter.forward(inputs)
    rows: dict[str, dict[str, float]] = {}
    for block in adapter.block_targets(stack=stack):
        pre = cache.get(f"{block.name}.resid_pre")
        post = cache.get(f"{block.name}.resid_post")
        attn = cache.get(f"{block.name}.attn")
        mlp = cache.get(f"{block.name}.mlp")
        row: dict[str, float] = {}
        if isinstance(pre, torch.Tensor):
            row["resid_pre_norm"] = float(pre.float().norm(dim=-1).mean().item())
        if isinstance(post, torch.Tensor):
            row["resid_post_norm"] = float(post.float().norm(dim=-1).mean().item())
        if isinstance(pre, torch.Tensor) and isinstance(post, torch.Tensor) and pre.shape == post.shape:
            row["block_update_ratio"] = float(((post - pre).float().norm(dim=-1) / pre.float().norm(dim=-1).clamp_min(1e-12)).mean().item())
        if isinstance(attn, torch.Tensor):
            row["attn_out_norm"] = float(attn.float().norm(dim=-1).mean().item())
        if isinstance(mlp, torch.Tensor):
            row["mlp_out_norm"] = float(mlp.float().norm(dim=-1).mean().item())
        rows[block.name] = row
    return rows


def residual_stream_delta_table(cache, *, block_names: list[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name in block_names:
        pre = cache.get(f"{name}.resid_pre")
        post = cache.get(f"{name}.resid_post")
        if not isinstance(pre, torch.Tensor) or not isinstance(post, torch.Tensor) or pre.shape != post.shape:
            continue
        delta = post.float() - pre.float()
        rows.append(
            {
                "block": name,
                "pre_norm": float(pre.float().norm(dim=-1).mean().item()),
                "post_norm": float(post.float().norm(dim=-1).mean().item()),
                "delta_norm": float(delta.norm(dim=-1).mean().item()),
                "delta_to_pre_ratio": float((delta.norm(dim=-1) / pre.float().norm(dim=-1).clamp_min(1e-12)).mean().item()),
            }
        )
    return rows
