from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

from interpret.tracer import ActivationTracer


@torch.inference_mode()
def mlp_lens(
    model,
    input_ids: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    layer_ids: Optional[Iterable[int]] = None,
    position: int = -1,
    topk: int = 10,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Project MLP outputs through lm_head to get top-k token predictions per layer.

    This focuses on the MLP submodule output to isolate feedforward contributions.
    Returns layer -> (topk_indices, topk_values) for the selected position.
    """
    model_was_training = model.training
    model.eval()

    tracer = ActivationTracer(model)
    tracer.add_block_residual_streams(prefix="blocks.")  # captures .mlp outputs too
    with tracer.trace():
        _ = model(input_ids, attn_mask)
        cache = tracer._cache

    if layer_ids is None:
        layer_ids = sorted({int(name.split(".")[1]) for name in cache.keys() if name.startswith("blocks.") and name.endswith(".mlp")})

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W = model.lm_head.weight
    elif hasattr(model, "embed") and hasattr(model.embed, "weight"):
        W = model.embed.weight
    else:
        raise AttributeError("Model has neither lm_head.weight nor embed.weight for MLP lens projection")

    results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for i in layer_ids:
        key = f"blocks.{i}.mlp"
        hid = cache.get(key)
        if hid is None:
            continue
        h_pos = hid[:, position, :]
        logits = torch.matmul(h_pos, W.t().to(h_pos.dtype))
        values, indices = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)
        results[i] = (indices[0].cpu(), values[0].cpu())

    if model_was_training:
        model.train()
    return results


