from __future__ import annotations

from typing import Optional

import torch

from tensor.numerics import masked_log_softmax


@torch.inference_mode()
def token_surprisal(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dim: int = -1,
) -> torch.Tensor:
    """Per-position surprisal: -log p(target).

    logits: (B,T,V), target_ids: (B,T), attn_mask: optional (B,T) with 1=token, 0=pad
    Returns: (B,T) surprisal, masked positions set to 0 if attn_mask provided.
    """
    logp = masked_log_softmax(logits, mask=None, dim=dim)
    s = -logp.gather(dim, target_ids.unsqueeze(dim)).squeeze(dim)
    if attn_mask is not None:
        m = (attn_mask == 0)
        s = s.masked_fill(m, 0.0)
    return s


@torch.inference_mode()
def token_entropy(
    logits: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dim: int = -1,
) -> torch.Tensor:
    """Per-position entropy H(p(.|context)).

    Returns: (B,T) entropy, masked positions set to 0 if attn_mask provided.
    """
    probs = torch.softmax(logits.float(), dim=dim)
    ent = -(probs * torch.log(probs.clamp_min(1e-45))).sum(dim=dim)
    if attn_mask is not None:
        m = (attn_mask == 0)
        ent = ent.masked_fill(m, 0.0)
    return ent.to(dtype=logits.dtype)


