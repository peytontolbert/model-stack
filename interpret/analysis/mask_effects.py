from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from tensor.masking import build_sliding_window_causal_mask, build_block_causal_mask, build_dilated_causal_mask, broadcast_mask


@torch.inference_mode()
def logit_change_with_mask(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    position: int = -1,
    target_token_id: Optional[int] = None,
    attn_mask_type: str = "sliding",  # "sliding"|"block"|"dilated"
    window: int = 128,
    block: int = 128,
    dilation: int = 2,
) -> float:
    """Apply a what-if attention mask and report target logit delta vs baseline.

    Builds a boolean causal mask (T,T), expands to (B,H,T,S), and forwards.
    """
    was_training = model.training
    model.eval()

    logits = model(input_ids, None)
    if target_token_id is None:
        target_token_id = int(logits[0, position].argmax().item())
    base = logits[0, position, target_token_id].float()

    B, T = input_ids.shape
    # infer heads
    H = getattr(getattr(model.blocks[0], "attn", None), "n_heads", 1)
    if attn_mask_type == "sliding":
        cm = build_sliding_window_causal_mask(T, window, device=input_ids.device)
    elif attn_mask_type == "block":
        cm = build_block_causal_mask(T, block, device=input_ids.device)
    elif attn_mask_type == "dilated":
        cm = build_dilated_causal_mask(T, window, dilation, device=input_ids.device)
    else:
        raise ValueError("attn_mask_type must be 'sliding'|'block'|'dilated'")
    full_mask = broadcast_mask(batch_size=B, num_heads=int(H), tgt_len=T, src_len=T, causal_mask=cm, padding_mask=None)
    l2 = model(input_ids, full_mask)
    val = l2[0, position, target_token_id].float()
    if was_training:
        model.train()
    return float(val - base)


