from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .tracer import ActivationTracer


def _get_norm_fn(model) -> callable:
    norm = getattr(model, "norm", None)
    if norm is None:
        return lambda x: x
    return norm


def _get_lm_proj_weight(model) -> torch.Tensor:
    # Prefer explicit lm_head weight
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    # Fallback to tied embedding if present
    if hasattr(model, "embed") and hasattr(model.embed, "weight"):
        return model.embed.weight
    raise AttributeError("Model has neither lm_head.weight nor embed.weight for logit lens projection")


@torch.inference_mode()
def logit_lens(
    model,
    input_ids: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    layer_ids: Optional[Iterable[int]] = None,
    position: int = -1,
    topk: int = 10,
    apply_norm: bool = True,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute logit-lens top-k for selected layers.

    Returns a dict mapping layer index -> (topk_indices, topk_values) for the token position.

    - position: which sequence position to analyze (default last token, -1)
    - apply_norm: if True, apply model.final_norm before projection
    - topk: number of tokens to return per layer
    """
    model_was_training = model.training
    model.eval()

    tracer = ActivationTracer(model)
    hooked = tracer.add_block_outputs(prefix="blocks.")
    if not hooked:
        # Fallback: try to hook every module that looks like a block output
        tracer.add_block_residual_streams(prefix="blocks.")

    with tracer.trace():
        _ = model(input_ids, attn_mask)
        cache = tracer._cache

    norm_fn = _get_norm_fn(model) if apply_norm else (lambda x: x)
    W = _get_lm_proj_weight(model)  # [V, d_model]

    # Determine layer ids to report
    if layer_ids is None:
        # Infer from captured names: blocks.{i}
        layer_ids = sorted(
            {
                int(name.split(".")[1])
                for name in cache.keys()
                if name.startswith("blocks.") and "." not in name[len("blocks.") :]
            }
        )

    results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for i in layer_ids:
        key = f"blocks.{i}"
        hid = cache.get(key)
        if hid is None:
            # Try mlp output if block output missing
            hid = cache.get(f"blocks.{i}.mlp")
            if hid is None:
                continue
        # hid: [B, T, D]
        h = norm_fn(hid)
        h_pos = h[:, position, :]  # [B, D]
        # Project to logits using transpose: [B, V]
        logits = torch.matmul(h_pos, W.t().to(h_pos.dtype))
        values, indices = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)
        # Return only the first batch element by default
        results[i] = (indices[0].cpu(), values[0].cpu())

    if model_was_training:
        model.train()
    return results

