from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter, resolve_model_score
from tensor.masking import build_sliding_window_causal_mask, build_block_causal_mask, build_dilated_causal_mask, broadcast_mask


@torch.inference_mode()
def logit_change_with_mask(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    *,
    inputs: Optional[ModelInputs] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    attn_mask_type: str = "sliding",  # "sliding"|"block"|"dilated"
    window: int = 128,
    block: int = 128,
    dilation: int = 2,
    stack: Optional[str] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    score_fn=None,
) -> float:
    """Apply a what-if attention mask and report target logit delta vs baseline.

    Builds a boolean causal mask (T,T), expands to (B,H,T,S), and forwards.
    """
    adapter = get_model_adapter(model)
    if inputs is None:
        inputs = coerce_model_inputs(
            model,
            input_ids,
            None,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=None,
        )
    resolved_stack = stack or ("causal" if adapter.kind == "causal" else "decoder")
    if resolved_stack not in {"causal", "decoder"}:
        raise ValueError("Alternative causal mask analysis only applies to causal or decoder self-attention stacks")
    seq_source = adapter.sequence_tokens(inputs, stack=resolved_stack)
    if seq_source is None:
        raise ValueError("A decoder-side token sequence is required to compare mask effects")
    was_training = model.training
    model.eval()

    outputs = adapter.forward(inputs)
    base, target_token_id, target_feature_index = resolve_model_score(
        model,
        outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    base = base.float()

    B, T = seq_source.shape
    # infer heads
    H = getattr(adapter.attention_target(0, stack=resolved_stack, kind="self").module, "n_heads", 1)
    if attn_mask_type == "sliding":
        cm = build_sliding_window_causal_mask(T, window, device=seq_source.device)
    elif attn_mask_type == "block":
        cm = build_block_causal_mask(T, block, device=seq_source.device)
    elif attn_mask_type == "dilated":
        cm = build_dilated_causal_mask(T, window, dilation, device=seq_source.device)
    else:
        raise ValueError("attn_mask_type must be 'sliding'|'block'|'dilated'")
    full_mask = broadcast_mask(batch_size=B, num_heads=int(H), tgt_len=T, src_len=T, causal_mask=cm, padding_mask=None)
    alt_inputs = ModelInputs(
        input_ids=inputs.input_ids,
        attention_mask=(full_mask if adapter.kind == "causal" else inputs.attention_mask),
        enc_input_ids=inputs.enc_input_ids,
        dec_input_ids=inputs.dec_input_ids,
        enc_padding_mask=inputs.enc_padding_mask,
        dec_self_mask=(full_mask if adapter.kind == "encoder_decoder" else inputs.dec_self_mask),
    )
    l2 = adapter.forward(alt_inputs)
    val, _, _ = resolve_model_score(
        model,
        l2,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    val = val.float()
    if was_training:
        model.train()
    return float(val - base)

