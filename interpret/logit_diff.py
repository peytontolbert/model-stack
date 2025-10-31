from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

from .tracer import ActivationTracer


@torch.inference_mode()
def logit_diff_lens(
    model,
    input_ids: torch.Tensor,
    target_token_id: int,
    baseline_token_id: int,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    layer_ids: Optional[Iterable[int]] = None,
    position: int = -1,
) -> Dict[int, float]:
    """Compute per-layer contribution to logit difference via residual projections.

    Projects hidden states (block outputs) onto W[target] - W[baseline] and reports scalar contributions.
    """
    model_was_training = model.training
    model.eval()

    tracer = ActivationTracer(model)
    tracer.add_block_outputs(prefix="blocks.")
    with tracer.trace():
        _ = model(input_ids, attn_mask)
        cache = tracer._cache

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W = model.lm_head.weight  # (V,D)
    elif hasattr(model, "embed") and hasattr(model, "embed") and hasattr(model.embed, "weight"):
        W = model.embed.weight
    else:
        raise AttributeError("Model has neither lm_head.weight nor embed.weight")

    v = (W[target_token_id] - W[baseline_token_id]).to(dtype=next(model.parameters()).dtype)  # (D,)

    if layer_ids is None:
        layer_ids = sorted(
            {
                int(name.split(".")[1])
                for name in cache.keys()
                if name.startswith("blocks.") and "." not in name[len("blocks.") :]
            }
        )

    scores: Dict[int, float] = {}
    for i in layer_ids:
        key = f"blocks.{i}"
        hid = cache.get(key)
        if hid is None:
            continue
        h = hid[:, position, :]  # (B,D)
        contrib = torch.einsum("bd,d->b", h.float(), v.float())
        scores[i] = float(contrib[0].item())

    if model_was_training:
        model.train()
    return scores


