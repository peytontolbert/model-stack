from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .tracer import ActivationTracer
from .model_adapter import coerce_model_inputs, get_model_adapter
from runtime.ops import resolve_linear_module_tensors as runtime_resolve_linear_module_tensors


def _get_norm_fn(model, *, stack: Optional[str] = None) -> callable:
    adapter = get_model_adapter(model)
    norm = adapter.final_norm(stack=stack)
    if norm is None:
        return lambda x: x
    return norm


def _get_lm_proj_weight(model, reference: Optional[torch.Tensor] = None, *, stack: Optional[str] = None) -> torch.Tensor:
    # Prefer explicit lm_head weight
    if hasattr(model, "lm_head"):
        weight, _ = runtime_resolve_linear_module_tensors(model.lm_head, reference=reference)
        return weight
    # Fallback to tied embedding if present
    adapter = get_model_adapter(model)
    embed = adapter.embedding_module(stack=stack)
    if hasattr(embed, "weight"):
        weight = embed.weight
        if reference is not None:
            weight = weight.to(device=reference.device, dtype=reference.dtype)
        return weight
    raise AttributeError("Model has neither lm_head.weight nor embed.weight for logit lens projection")


@torch.inference_mode()
def logit_lens(
    model,
    input_ids: Optional[torch.Tensor],
    *,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    layer_ids: Optional[Iterable[int]] = None,
    position: int = -1,
    topk: int = 10,
    apply_norm: bool = True,
    stack: Optional[str] = None,
    kind: Optional[str] = None,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute logit-lens top-k for selected layers.

    Returns a dict mapping layer index -> (topk_indices, topk_values) for the token position.

    - position: which sequence position to analyze (default last token, -1)
    - apply_norm: if True, apply model.final_norm before projection
    - topk: number of tokens to return per layer
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

    norm_fn = _get_norm_fn(model, stack=resolved_stack) if apply_norm else (lambda x: x)
    targets = adapter.block_targets(stack=resolved_stack)
    if resolved_kind is not None:
        targets = [target for target in targets if target.kind == resolved_kind]
    # Determine layer ids to report
    if layer_ids is None:
        layer_ids = [target.layer_index for target in targets]

    results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    target_by_layer = {int(target.layer_index): target for target in targets}
    for i in layer_ids:
        target = target_by_layer.get(int(i))
        if target is None:
            continue
        key = f"{target.name}.resid_post"
        hid = cache.get(key)
        if hid is None:
            continue
        # hid: [B, T, D]
        h = norm_fn(hid)
        h_pos = h[:, position, :]  # [B, D]
        W = _get_lm_proj_weight(model, reference=h_pos, stack=resolved_stack)  # [V, D]
        # Project to logits using transpose: [B, V]
        logits = torch.matmul(h_pos, W.t())
        values, indices = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)
        # Return only the first batch element by default
        results[i] = (indices[0].cpu(), values[0].cpu())

    if model_was_training:
        model.train()
    return results
