from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from interpret.activation_cache import CaptureSpec
from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter
from interpret.tracer import ActivationTracer
from tensor.numerics import masked_mean


@torch.inference_mode()
def residual_norms(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    *,
    inputs: Optional[ModelInputs] = None,
    attn_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute per-layer residual L2 norms before and after each block.

    Returns dict with keys: pre (L vector), post (L vector), tokenwise_pre/post (B,T,L).
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
    resolved_stack = stack or ("causal" if adapter.kind == "causal" else "encoder" if adapter.kind == "encoder" else "decoder")
    seq_source = adapter.sequence_tokens(inputs, stack=resolved_stack)
    if seq_source is None:
        raise ValueError("A token sequence is required to compute residual norms")
    row_mask = None
    if adapter.kind == "causal":
        row_mask = inputs.attention_mask
    elif adapter.kind == "encoder":
        row_mask = inputs.attention_mask
    elif resolved_stack == "encoder":
        row_mask = inputs.enc_padding_mask
    else:
        row_mask = inputs.dec_self_mask
    if row_mask is not None and row_mask.ndim != 2:
        row_mask = None

    was_training = model.training
    model.eval()
    tracer = ActivationTracer(model, spec=CaptureSpec(move_to_cpu=False))
    targets = adapter.block_targets(stack=resolved_stack)
    tracer.add_residual_streams(stack=resolved_stack)
    with tracer.trace() as cache:
        _ = adapter.forward(inputs)

    pre_list = []
    post_list = []
    tok_pre = []
    tok_post = []
    token_mask = (row_mask == 0) if row_mask is not None else None
    for target in targets:
        pre_x = cache.get(f"{target.name}.resid_pre")
        post_x = cache.get(f"{target.name}.resid_post")
        if pre_x is None or post_x is None:
            continue
        pre = pre_x.norm(dim=-1)
        post = post_x.norm(dim=-1)
        tok_pre.append(pre)
        tok_post.append(post)
        pre_list.append(masked_mean(pre, mask=token_mask, dim=-1).mean(dim=0))
        post_list.append(masked_mean(post, mask=token_mask, dim=-1).mean(dim=0))
    out = {
        "pre": torch.stack(pre_list),
        "post": torch.stack(post_list),
        "tokenwise_pre": torch.stack(tok_pre, dim=-1),
        "tokenwise_post": torch.stack(tok_post, dim=-1),
    }
    if was_training:
        model.train()
    return out
