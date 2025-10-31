from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from tensor.numerics import masked_mean


@torch.inference_mode()
def residual_norms(model: nn.Module, input_ids: torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Compute per-layer residual L2 norms before and after each block.

    Returns dict with keys: pre (L vector), post (L vector), tokenwise_pre/post (B,T,L).
    """
    was_training = model.training
    model.eval()

    x = model.embed(input_ids)
    L = len(model.blocks)
    B, T, D = x.shape
    pre_list = []
    post_list = []
    tok_pre = []
    tok_post = []
    for i, blk in enumerate(model.blocks):
        pre = x.norm(dim=-1)  # (B,T)
        tok_pre.append(pre)
        pre_list.append(masked_mean(pre, mask=(attn_mask == 0) if attn_mask is not None else None, dim=-1).mean(dim=0))  # (B)->scalar
        x = blk(x, attn_mask, None)
        post = x.norm(dim=-1)
        tok_post.append(post)
        post_list.append(masked_mean(post, mask=(attn_mask == 0) if attn_mask is not None else None, dim=-1).mean(dim=0))
    out = {
        "pre": torch.stack(pre_list),
        "post": torch.stack(post_list),
        "tokenwise_pre": torch.stack(tok_pre, dim=-1),
        "tokenwise_post": torch.stack(tok_post, dim=-1),
    }
    if was_training:
        model.train()
    return out


