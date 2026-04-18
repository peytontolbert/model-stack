from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from runtime.attention_modules import EagerAttention

from interpret.model_adapter import AttentionSnapshot, ModelInputs, coerce_model_inputs, eager_attention_forward, get_model_adapter, resolve_model_score


def _wrap_forward_capture_heads(
    attn: EagerAttention,
    sink: Dict[int, torch.Tensor],
    layer_idx: int,
    *,
    detach: bool = True,
    move_to_cpu: bool = True,
):
    orig_forward = attn.forward

    def _capture(snapshot: AttentionSnapshot) -> None:
        if snapshot.head_out is None:
            return
        out = snapshot.head_out
        if not detach and not move_to_cpu:
            sink[layer_idx] = out
            return
        if detach:
            out = out.detach()
        if move_to_cpu and out.device.type != "cpu":
            out = out.to("cpu")
        sink[layer_idx] = out.clone()

    def forward(q, k, v, mask, cache=None, *, position_embeddings=None, position_ids=None):
        return eager_attention_forward(
            attn,
            q,
            k,
            v,
            mask,
            cache,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            capture=_capture,
            keep_grad=(not detach and not move_to_cpu),
        )

    return orig_forward, forward


def _wrap_forward_patch_heads(
    attn: EagerAttention,
    source: Dict[int, torch.Tensor],
    layer_idx: int,
    heads: Optional[Iterable[int]] = None,
    *,
    time_index: Optional[int] = None,
):
    selected_heads = None if heads is None else {int(h) for h in heads}
    orig_forward = attn.forward

    def _patch(head_out: torch.Tensor) -> torch.Tensor:
        clean = source.get(layer_idx)
        if clean is None:
            return head_out
        patched = head_out.clone()
        clean_live = clean.to(device=patched.device, dtype=patched.dtype)
        head_indices = range(patched.shape[1]) if selected_heads is None else selected_heads
        for h in head_indices:
            if not (0 <= h < patched.shape[1]):
                continue
            if time_index is None:
                patched[:, h].copy_(clean_live[:, h])
            else:
                patched[:, h, time_index].copy_(clean_live[:, h, time_index])
        return patched

    def forward(q, k, v, mask, cache=None, *, position_embeddings=None, position_ids=None):
        return eager_attention_forward(
            attn,
            q,
            k,
            v,
            mask,
            cache,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            patch_heads=_patch,
        )

    return orig_forward, forward


@torch.inference_mode()
def causal_trace_heads_restore_table(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    stack: Optional[str] = None,
    kind: str = "self",
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    score_fn=None,
) -> torch.Tensor:
    """Return (L, H) restored score fractions by patching one head at a time."""
    adapter = get_model_adapter(model)
    if clean_inputs is None:
        clean_inputs = coerce_model_inputs(
            model,
            clean_input_ids,
            attn_mask,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
        )
    if corrupted_inputs is None:
        corrupted_inputs = coerce_model_inputs(
            model,
            corrupted_input_ids,
            attn_mask,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
        )
    targets = adapter.attention_targets(stack=stack, kind=kind)  # type: ignore[arg-type]
    was_training = model.training
    model.eval()

    outputs_clean = adapter.forward(clean_inputs)
    clean_score, target_token_id, target_feature_index = resolve_model_score(
        model,
        outputs_clean,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )

    clean_heads: Dict[int, torch.Tensor] = {}
    wrappers = []
    for idx, target in enumerate(targets):
        attn = target.module
        if not isinstance(attn, EagerAttention):
            continue
        orig, new = _wrap_forward_capture_heads(attn, clean_heads, idx)
        wrappers.append((attn, orig))
        attn.forward = new  # type: ignore[assignment]
    try:
        _ = adapter.forward(clean_inputs)
    finally:
        for attn, orig in wrappers:
            attn.forward = orig  # type: ignore[assignment]

    outputs_corrupted = adapter.forward(corrupted_inputs)
    base_score, _, _ = resolve_model_score(
        model,
        outputs_corrupted,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )

    num_layers = len(targets)
    num_heads = max((int(getattr(target.module, "n_heads", 0)) for target in targets), default=0)
    table = torch.zeros(num_layers, num_heads)

    denom = (clean_score - base_score).abs() + 1e-8
    for li, target in enumerate(targets):
        attn = target.module
        if not isinstance(attn, EagerAttention):
            continue
        layer_heads = clean_heads.get(li)
        if layer_heads is None:
            continue
        for h in range(int(layer_heads.shape[1])):
            orig, new = _wrap_forward_patch_heads(attn, clean_heads, li, heads=[h])
            attn.forward = new  # type: ignore[assignment]
            try:
                outputs_patch = adapter.forward(corrupted_inputs)
            finally:
                attn.forward = orig  # type: ignore[assignment]
            patched_score, _, _ = resolve_model_score(
                model,
                outputs_patch,
                position=position,
                target_token_id=target_token_id,
                target_feature_index=target_feature_index,
                score_fn=score_fn,
            )
            table[li, h] = ((patched_score - base_score) / denom).detach().cpu()

    if was_training:
        model.train()
    return table


__all__ = [
    "_wrap_forward_capture_heads",
    "_wrap_forward_patch_heads",
    "causal_trace_heads_restore_table",
]
