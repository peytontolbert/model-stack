from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter, resolve_model_score


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
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    patch_points: Iterable[str],
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    score_fn=None,
) -> torch.Tensor:
    """Measure how much of the clean logit at position is restored by patching.

    Runs three forwards:
      - clean: f(clean)
      - corrupted: f(corrupted)
      - patched: f(corrupted) with outputs at patch_points replaced by those from f(clean)

    Returns a 1D tensor of shape [V] with the fraction of clean-correct-logit recovered (for the last token).
    """
    from interpret.tracer import ActivationTracer
    adapter = get_model_adapter(model)
    if clean_inputs is None:
        clean_inputs = coerce_model_inputs(
            model,
            clean_input_ids,
            attn_mask,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
        )
    if corrupted_inputs is None:
        corrupted_inputs = coerce_model_inputs(
            model,
            corrupted_input_ids,
            attn_mask,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
        )

    model_was_training = model.training
    model.eval()

    # Clean run with capture
    tracer = ActivationTracer(model)
    tracer.add_modules(patch_points)
    with tracer.trace() as cache:
        clean_logits = adapter.forward(clean_inputs)
        clean_h = {k: cache.get(k) for k in patch_points}

    # Corrupted baseline
    corrupted_logits = adapter.forward(corrupted_inputs)

    # Patched run
    with output_patching(model, clean_h):
        patched_logits = adapter.forward(corrupted_inputs)

    if adapter.output_module() is not None and score_fn is None:
        clean_pos = clean_logits[:, position, :]
        corrupted_pos = corrupted_logits[:, position, :]
        patched_pos = patched_logits[:, position, :]
        denom = (clean_pos - corrupted_pos)
        frac = (patched_pos - corrupted_pos) / (denom.abs() + 1e-8)
        frac = frac[0].detach().cpu()
    else:
        clean_score, target_token_id, target_feature_index = resolve_model_score(
            model,
            clean_logits,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        corrupted_score, _, _ = resolve_model_score(
            model,
            corrupted_logits,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        patched_score, _, _ = resolve_model_score(
            model,
            patched_logits,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        frac = (((patched_score - corrupted_score) / ((clean_score - corrupted_score).abs() + 1e-8))).view(1).detach().cpu()

    if model_was_training:
        model.train()
    return frac

