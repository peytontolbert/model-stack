from __future__ import annotations

from typing import Iterable, Optional

import torch

from .weights import attention_weights_for_layer


@torch.inference_mode()
def attention_rollout(
    model,
    input_ids: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    layers: Optional[Iterable[int]] = None,
    residual: bool = True,
    head_agg: str = "mean",
) -> torch.Tensor:
    """Compute attention rollout (B, T, T) across selected layers.

    - residual: if True, use (A + I)/2 per layer (common in literature)
    - head_agg: 'mean' or 'max' across heads
    """
    B = input_ids.shape[0]
    num_layers = len(model.blocks)
    if layers is None:
        layers = list(range(num_layers))
    layers = list(layers)

    # Initialize rollout matrix R as identity per batch
    T = int(input_ids.shape[1])
    R = torch.eye(T, device=input_ids.device, dtype=torch.float32).unsqueeze(0).expand(B, T, T).clone()

    for li in layers:
        probs = attention_weights_for_layer(model, input_ids, layer_index=int(li), attn_mask=attn_mask)  # (B,H,T,S)
        if head_agg == "mean":
            A = probs.mean(dim=1)  # (B,T,S)
        elif head_agg == "max":
            A = probs.max(dim=1).values
        else:
            raise ValueError("head_agg must be 'mean' or 'max'")
        if residual:
            I = torch.eye(T, device=A.device, dtype=A.dtype).unsqueeze(0)
            A = (A + I) / 2.0
        # Compose: queries attend to sources; combine layer maps
        R = A @ R  # (B,T,S) x (B,S,S)

    return R


