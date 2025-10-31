from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def predict_argmax(model: nn.Module, input_ids: torch.Tensor, position: int = -1, attn_mask: Optional[torch.Tensor] = None) -> int:
    logits = model(input_ids, attn_mask)
    return int(logits[0, position].argmax().item())


def integrated_gradients_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    target_token_id: Optional[int] = None,
    steps: int = 20,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Integrated gradients attribution over input token embeddings.

    Returns attribution scores per token: shape [T], using zero baseline and linear path scaling of embedding output.
    """
    assert hasattr(model, "embed"), "Model must have an embedding layer 'embed'"
    was_training = model.training
    device = input_ids.device
    model.eval()

    if target_token_id is None:
        target_token_id = predict_argmax(model, input_ids, position=position, attn_mask=attn_mask)

    # We will scale the embedding output by alpha in [0,1]
    name_to_module = dict(model.named_modules())
    embed = name_to_module.get("embed", getattr(model, "embed"))

    def run_with_alpha(alpha: float) -> torch.Tensor:
        grads = {}

        def hook(_m, _inp, out):
            out_alpha = out * alpha
            out_alpha.retain_grad()

            def save_grad(grad):
                grads["emb_grad"] = grad
            out_alpha.register_hook(save_grad)
            return out_alpha

        handle = embed.register_forward_hook(hook)
        try:
            logits = model(input_ids, attn_mask)
            score = logits[0, position, target_token_id]
            model.zero_grad(set_to_none=True)
            score.backward()
        finally:
            handle.remove()
        return grads["emb_grad"], None  # grad wrt scaled embed output

    total_grad = None
    for s in range(1, steps + 1):
        alpha = float(s) / float(steps)
        grad, _ = run_with_alpha(alpha)
        grad = grad.detach()  # [B, T, D]
        total_grad = grad if total_grad is None else total_grad + grad

    # Average gradient along path and multiply by (emb - baseline) = emb
    with torch.no_grad():
        # Obtain actual embedding output once (alpha=1 path already computed above, but recompute cleanly)
        emb_out = []
        def capture(_m, _inp, out):
            emb_out.append(out.detach())
            return out
        h2 = embed.register_forward_hook(capture)
        try:
            _ = model(input_ids, attn_mask)
        finally:
            h2.remove()
        emb = emb_out[0]  # [B, T, D]

    avg_grad = total_grad / float(steps)
    attrib = (emb * avg_grad).sum(dim=-1)  # [B, T]
    scores = attrib[0].detach().cpu()

    if was_training:
        model.train()
    return scores


