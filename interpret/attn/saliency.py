from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from attn.eager import EagerAttention


@torch.no_grad()
def _infer_target_token_id(model: nn.Module, input_ids: torch.Tensor, position: int, attn_mask: Optional[torch.Tensor]):
    logits = model(input_ids, attn_mask)
    return int(logits[0, position].argmax().item())


def head_grad_saliencies(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    position: int = -1,
    target_token_id: Optional[int] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-head grad×output saliency (L,H) for the target logit.

    Returns a tensor (L,H) of summed |grad * output| over (B,T,Dh) for each head.
    Works with EagerAttention.
    """
    was_training = model.training
    model.train(False)
    # Determine target if not specified
    if target_token_id is None:
        target_token_id = _infer_target_token_id(model, input_ids, position, attn_mask)

    # Register hooks to capture per-head output tensors with grad
    records = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_m, _inp, out):
            # out: (B,H,T,Dh) after SDPA in our eager path; if not, we skip
            if out is None or not isinstance(out, torch.Tensor) or out.ndim != 4:
                return out
            out.retain_grad()
            records[layer_idx] = out
            return out
        return hook

    # We instrument by wrapping EagerAttention forward to insert a hook point after SDPA.
    # EagerAttention exposes only the final merged output; to get head outputs, we piggy-back
    # on the head capture wrapper in causal/head_patching by calling scaled_dot_product_attention directly.
    from interpret.causal.head_patching import _wrap_forward_capture_heads  # reuse helper

    wrappers = []
    for li, blk in enumerate(model.blocks):
        attn = getattr(blk, "attn", None)
        if isinstance(attn, EagerAttention):
            orig, new = _wrap_forward_capture_heads(attn, records, li)
            wrappers.append((attn, orig))
            attn.forward = new  # type: ignore

    try:
        logits = model(input_ids, attn_mask)
        score = logits[0, position, target_token_id]
        model.zero_grad(set_to_none=True)
        score.backward()
    finally:
        for attn, orig in wrappers:
            attn.forward = orig  # type: ignore

    L = len(model.blocks)
    H = getattr(getattr(model.blocks[0], "attn", None), "n_heads", 0) if L > 0 else 0
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


