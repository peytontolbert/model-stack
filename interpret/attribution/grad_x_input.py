from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def grad_x_input_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    target_token_id: Optional[int] = None,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gradient x input attribution on token embeddings; returns scores [T].

    Uses embedding output as "input". If target_token_id is None, uses argmax at position.
    """
    assert hasattr(model, "embed"), "Model must have an embedding layer 'embed'"
    was_training = model.training
    model.eval()

    name_to_module = dict(model.named_modules())
    embed = name_to_module.get("embed", getattr(model, "embed"))

    logits = model(input_ids, attn_mask)
    if target_token_id is None:
        target_token_id = int(logits[0, position].argmax().item())

    grads = {}
    feats = {}

    def hook(_m, _inp, out):
        out.retain_grad()
        feats["emb"] = out
        return out

    h = embed.register_forward_hook(hook)
    try:
        logits = model(input_ids, attn_mask)
        score = logits[0, position, target_token_id]
        model.zero_grad(set_to_none=True)
        score.backward()
        grad = feats["emb"].grad.detach()  # [B,T,D]
        emb = feats["emb"].detach()
    finally:
        h.remove()

    scores = (emb * grad).sum(dim=-1)[0].cpu()
    if was_training:
        model.train()
    return scores


