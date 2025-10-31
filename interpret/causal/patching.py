from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


@contextmanager
def output_patching(model: nn.Module, replacements: Dict[str, torch.Tensor]):
    """Context manager that replaces outputs of specified modules during forward.

    replacements: mapping from module name (as in model.named_modules()) to full replacement tensor.
    The replacement tensor must match the module's output shape for the actual inputs.
    """
    name_to_module: Dict[str, nn.Module] = dict(model.named_modules())
    handles: list = []

    def make_hook(name: str, rep: torch.Tensor):
        def _hook(_mod: nn.Module, _inputs, _output):
            return rep
        return _hook

    for name, rep in replacements.items():
        mod = name_to_module.get(name)
        if mod is None:
            continue
        handles.append(mod.register_forward_hook(make_hook(name, rep)))
    try:
        yield
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


@torch.inference_mode()
def causal_trace_restore_fraction(
    model: nn.Module,
    *,
    clean_input_ids: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    patch_points: Iterable[str],
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Measure how much of the clean logit at position is restored by patching.

    Runs three forwards:
      - clean: f(clean)
      - corrupted: f(corrupted)
      - patched: f(corrupted) with outputs at patch_points replaced by those from f(clean)

    Returns a 1D tensor of shape [V] with the fraction of clean-correct-logit recovered (for the last token).
    """
    from interpret.tracer import ActivationTracer

    model_was_training = model.training
    model.eval()

    # Clean run with capture
    tracer = ActivationTracer(model)
    tracer.add_modules(patch_points)
    with tracer.trace() as cache:
        clean_logits = model(clean_input_ids, attn_mask)
        clean_h = {k: cache.get(k) for k in patch_points}

    # Corrupted baseline
    corrupted_logits = model(corrupted_input_ids, attn_mask)

    # Patched run
    with output_patching(model, clean_h):
        patched_logits = model(corrupted_input_ids, attn_mask)

    # Compare on target position
    clean_pos = clean_logits[:, position, :]
    corrupted_pos = corrupted_logits[:, position, :]
    patched_pos = patched_logits[:, position, :]

    # Recovery fraction per vocab entry: (patched - corrupted) / (clean - corrupted)
    denom = (clean_pos - corrupted_pos)
    eps = 1e-8
    frac = (patched_pos - corrupted_pos) / (denom.abs() + eps)
    frac = frac[0].detach().cpu()

    if model_was_training:
        model.train()
    return frac


