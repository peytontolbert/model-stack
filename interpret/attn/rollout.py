from __future__ import annotations

from typing import Iterable, Optional

import torch

from interpret.model_adapter import coerce_model_inputs, get_model_adapter
from .weights import attention_weights_for_layer


@torch.inference_mode()
def attention_rollout(
    model,
    input_ids: Optional[torch.Tensor],
    *,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    kind: str = "self",
    layers: Optional[Iterable[int]] = None,
    residual: bool = True,
    head_agg: str = "mean",
) -> torch.Tensor:
    """Compute attention rollout (B, T, T) across selected layers.

    - residual: if True, use (A + I)/2 per layer (common in literature)
    - head_agg: 'mean' or 'max' across heads
    """
    adapter = get_model_adapter(model)
    inputs = coerce_model_inputs(
        model,
        input_ids,
        attn_mask,
        enc_input_ids=enc_input_ids,
        dec_input_ids=dec_input_ids,
        enc_padding_mask=enc_padding_mask,
        dec_self_mask=dec_self_mask,
    )
    seq_source = adapter.sequence_tokens(inputs, stack=stack)
    if seq_source is None:
        raise ValueError("A token sequence is required to compute attention rollout")
    B = seq_source.shape[0]
    num_layers = len(adapter.attention_targets(stack=stack, kind=kind))  # type: ignore[arg-type]
    if layers is None:
        layers = list(range(num_layers))
    layers = list(layers)
    if kind == "cross" and len(layers) > 1:
        raise ValueError("Cross-attention rollout is only defined for a single layer in this implementation")

    # Initialize rollout matrix R as identity per batch
    T = int(seq_source.shape[1])
    R = torch.eye(T, device=seq_source.device, dtype=torch.float32).unsqueeze(0).expand(B, T, T).clone()

    for li in layers:
        probs = attention_weights_for_layer(
            model,
            input_ids,
            layer_index=int(li),
            attn_mask=attn_mask,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
            stack=stack,
            kind=kind,
        )  # (B,H,T,S)
        if head_agg == "mean":
            A = probs.mean(dim=1)  # (B,T,S)
        elif head_agg == "max":
            A = probs.max(dim=1).values
        else:
            raise ValueError("head_agg must be 'mean' or 'max'")
        if A.shape[-2] != T or A.shape[-1] != T:
            raise ValueError("Attention rollout currently requires square attention maps; use self-attention targets")
        if residual:
            I = torch.eye(T, device=A.device, dtype=A.dtype).unsqueeze(0)
            A = (A + I) / 2.0
        # Compose: queries attend to sources; combine layer maps
        R = A @ R  # (B,T,S) x (B,S,S)

    return R
