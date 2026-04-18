from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

from interpret.tracer import ActivationTracer
from interpret.model_adapter import coerce_model_inputs, get_model_adapter
from runtime.ops import resolve_linear_module_tensors as runtime_resolve_linear_module_tensors


@torch.inference_mode()
def mlp_lens(
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
    stack: Optional[str] = None,
    kind: Optional[str] = None,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Project MLP outputs through lm_head to get top-k token predictions per layer.

    This focuses on the MLP submodule output to isolate feedforward contributions.
    Returns layer -> (topk_indices, topk_values) for the selected position.
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
    tracer.add_mlp_surfaces(stack=resolved_stack, kind=resolved_kind, include=("mlp_out",))
    with tracer.trace() as cache:
        _ = adapter.forward(inputs)

    targets = adapter.mlp_targets(stack=resolved_stack, kind=resolved_kind)

    if layer_ids is None:
        layer_ids = [target.layer_index for target in targets]

    results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    target_by_layer = {int(target.layer_index): target for target in targets}
    for i in layer_ids:
        target = target_by_layer.get(int(i))
        if target is None:
            continue
        key = f"{target.name}.mlp_out"
        hid = cache.get(key)
        if hid is None:
            continue
        h_pos = hid[:, position, :]
        if hasattr(model, "lm_head"):
            W, _ = runtime_resolve_linear_module_tensors(model.lm_head, reference=h_pos)
        else:
            embed = adapter.embedding_module(stack=resolved_stack)
            if not hasattr(embed, "weight"):
                raise AttributeError("Model has neither lm_head.weight nor embed.weight for MLP lens projection")
            W = embed.weight.to(device=h_pos.device, dtype=h_pos.dtype)
        logits = torch.matmul(h_pos, W.t())
        values, indices = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)
        results[i] = (indices[0].cpu(), values[0].cpu())

    if model_was_training:
        model.train()
    return results
