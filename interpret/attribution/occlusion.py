from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn


def _infer_target_token_id(model: nn.Module, input_ids: torch.Tensor, position: int, attn_mask: Optional[torch.Tensor]) -> int:
    with torch.no_grad():
        logits = model(input_ids, attn_mask)
        return int(logits[0, position].argmax().item())


def _occlude_positions_hook(positions: Iterable[int]):
    pos = sorted({int(p) for p in positions})

    def hook(_m, _inp, out: torch.Tensor):
        if not isinstance(out, torch.Tensor) or out.ndim != 3:
            return out
        B, T, D = out.shape
        mask = torch.zeros(B, T, 1, device=out.device, dtype=out.dtype)
        for p in pos:
            if -T <= p < T:
                mask[:, p if p >= 0 else (T + p)] = 1
        return out * (1 - mask)

    return hook


@torch.inference_mode()
def token_occlusion_importance(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    positions: Optional[Iterable[int]] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    attn_mask: Optional[torch.Tensor] = None,
    mode: str = "logit",  # "logit" | "prob" | "nll"
) -> torch.Tensor:
    """Score importance of tokens by zeroing their embedding outputs.

    Returns a vector [T] with delta in the selected score when each position is occluded.
    - mode="logit": target logit drop (clean - occluded)
      "prob": target probability drop
      "nll": increase in -log p(target)
    If positions is None, scores all positions individually; otherwise only those.
    """
    was_training = model.training
    model.eval()

    # Determine target if not provided
    if target_token_id is None:
        target_token_id = _infer_target_token_id(model, input_ids, position, attn_mask)

    # Baseline score
    logits = model(input_ids, attn_mask)
    base_logit = logits[0, position, target_token_id].float()
    if mode == "prob" or mode == "nll":
        probs = torch.softmax(logits.float(), dim=-1)
        base_prob = probs[0, position, target_token_id]
        base_nll = -torch.log(base_prob.clamp_min(1e-45))

    # Decide which positions to evaluate
    T = int(input_ids.shape[1])
    eval_positions = list(range(T)) if positions is None else [int(p) for p in positions]
    scores = torch.zeros(T, device=input_ids.device, dtype=torch.float32)

    # Embedding module
    name_to_module = dict(model.named_modules())
    embed = name_to_module.get("embed", getattr(model, "embed"))

    for p in eval_positions:
        hook = embed.register_forward_hook(_occlude_positions_hook([p]))
        try:
            l = model(input_ids, attn_mask)
        finally:
            hook.remove()
        if mode == "logit":
            occl = l[0, position, target_token_id].float()
            scores[p] = (base_logit - occl).detach()
        elif mode == "prob":
            pr = torch.softmax(l.float(), dim=-1)[0, position, target_token_id]
            scores[p] = (base_prob - pr).detach()
        elif mode == "nll":
            pr = torch.softmax(l.float(), dim=-1)[0, position, target_token_id]
            nll = -torch.log(pr.clamp_min(1e-45))
            scores[p] = (nll - base_nll).detach()
        else:
            raise ValueError("mode must be 'logit' | 'prob' | 'nll'")

    if was_training:
        model.train()
    return scores.cpu()


