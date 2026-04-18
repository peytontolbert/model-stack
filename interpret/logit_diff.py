from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

from .tracer import ActivationTracer
from .model_adapter import coerce_model_inputs, get_model_adapter
from runtime.ops import resolve_linear_module_tensors as runtime_resolve_linear_module_tensors


@torch.inference_mode()
def logit_diff_lens(
    model,
    input_ids: Optional[torch.Tensor],
    target_token_id: int,
    baseline_token_id: int,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    layer_ids: Optional[Iterable[int]] = None,
    position: int = -1,
    stack: Optional[str] = None,
    kind: Optional[str] = None,
) -> Dict[int, float]:
    """Compute per-layer contribution to logit difference via residual projections.

    Projects hidden states (block outputs) onto W[target] - W[baseline] and reports scalar contributions.
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
    resolved_stack = stack or ("causal" if adapter.kind == "causal" else "encoder" if adapter.kind == "encoder" else "decoder")
    resolved_kind = kind
    if adapter.kind == "encoder_decoder" and resolved_stack == "decoder" and resolved_kind is None:
        resolved_kind = "cross"
    model_was_training = model.training
    model.eval()

    tracer = ActivationTracer(model)
    tracer.add_residual_streams(stack=resolved_stack)
    with tracer.trace() as cache:
        _ = adapter.forward(inputs)

    if hasattr(model, "lm_head"):
        W, _ = runtime_resolve_linear_module_tensors(model.lm_head)
    else:
        embed = adapter.embedding_module(stack=resolved_stack)
        if hasattr(embed, "weight"):
            W = embed.weight
        else:
            raise AttributeError("Model has neither lm_head.weight nor embed.weight")

    v = (W[target_token_id] - W[baseline_token_id]).to(dtype=next(model.parameters()).dtype)  # (D,)
    targets = adapter.block_targets(stack=resolved_stack)
    if resolved_kind is not None:
        targets = [target for target in targets if target.kind == resolved_kind]

    if layer_ids is None:
        layer_ids = [target.layer_index for target in targets]

    scores: Dict[int, float] = {}
    target_by_layer = {int(target.layer_index): target for target in targets}
    for i in layer_ids:
        target = target_by_layer.get(int(i))
        if target is None:
            continue
        key = f"{target.name}.resid_post"
        hid = cache.get(key)
        if hid is None:
            continue
        h = hid[:, position, :]  # (B,D)
        contrib = torch.einsum("bd,d->b", h.float(), v.float())
        scores[i] = float(contrib[0].item())

    if model_was_training:
        model.train()
    return scores
