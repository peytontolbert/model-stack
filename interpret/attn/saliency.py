from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from runtime.attention_modules import EagerAttention
from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter, resolve_model_score


@torch.no_grad()
def _infer_target_token_id(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    position: int,
    attn_mask: Optional[torch.Tensor],
    *,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    target_feature_index: Optional[int] = None,
    score_fn=None,
):
    adapter = get_model_adapter(model)
    inputs = coerce_model_inputs(
        model,
        input_ids,
        attn_mask,
        enc_input_ids=enc_input_ids,
        dec_input_ids=dec_input_ids,
        enc_padding_mask=enc_padding_mask,
        dec_self_mask=dec_self_mask,
    )
    outputs = adapter.forward(inputs)
    _, target_token_id, target_feature_index = resolve_model_score(
        model,
        outputs,
        position=position,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    return int(target_token_id if target_token_id is not None else target_feature_index)


def head_grad_saliencies(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    *,
    inputs: Optional[ModelInputs] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    attn_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    kind: str = "self",
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    score_fn=None,
) -> torch.Tensor:
    """Compute per-head grad×output saliency (L,H) for the target logit.

    Returns a tensor (L,H) of summed |grad * output| over (B,T,Dh) for each head.
    Works with EagerAttention.
    """
    adapter = get_model_adapter(model)
    if inputs is None:
        inputs = coerce_model_inputs(
            model,
            input_ids,
            attn_mask,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
        )
    was_training = model.training
    model.train(False)
    # Determine target if not specified
    if target_token_id is None and target_feature_index is None and score_fn is None:
        inferred = _infer_target_token_id(
            model,
            input_ids,
            position,
            attn_mask,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        if adapter.output_module() is not None:
            target_token_id = inferred
        else:
            target_feature_index = inferred

    # Capture live per-head outputs before the final output projection.
    records: dict[int, torch.Tensor] = {}
    from interpret.causal.head_patching import _wrap_forward_capture_heads  # reuse helper

    wrappers = []
    targets = adapter.attention_targets(stack=stack, kind=kind)  # type: ignore[arg-type]
    for li, target in enumerate(targets):
        attn = target.module
        if isinstance(attn, EagerAttention):
            orig, new = _wrap_forward_capture_heads(attn, records, li, detach=False, move_to_cpu=False)
            wrappers.append((attn, orig))
            attn.forward = new  # type: ignore

    try:
        outputs = adapter.forward(inputs)
        score, target_token_id, target_feature_index = resolve_model_score(
            model,
            outputs,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        model.zero_grad(set_to_none=True)
        score.backward()
    finally:
        for attn, orig in wrappers:
            attn.forward = orig  # type: ignore

    L = len(targets)
    H = max((int(getattr(target.module, "n_heads", 0)) for target in targets), default=0)
    sal = torch.zeros(L, H)
    for li, out in records.items():
        if out.grad is None:
            continue
        # Reduce abs(grad * out) over B,T,Dh per head -> (H,)
        g = out.grad.detach().float()
        y = out.detach().float()
        s = (g * y).abs().sum(dim=(0, 2, 3))  # (H,)
        # If needed, move to CPU
        sal[li, : s.numel()] = s.cpu()

    if was_training:
        model.train(True)
    return sal
