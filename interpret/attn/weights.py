from __future__ import annotations

from typing import Optional

import torch

from tensor.numerics import entropy_from_probs

from interpret.model_adapter import AttentionSnapshot, coerce_model_inputs, get_model_adapter, patched_attention


@torch.inference_mode()
def attention_snapshot_for_layer(
    model,
    input_ids: Optional[torch.Tensor] = None,
    layer_index: int = 0,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    kind: str = "self",
) -> AttentionSnapshot:
    """Capture a runtime-faithful attention snapshot for one layer.

    The snapshot is collected by wrapping the real attention module used during the
    model forward pass, so the prepared masks and resolved module tensors match the
    actual runtime path.
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
    target = adapter.attention_target(int(layer_index), stack=stack, kind=kind)  # type: ignore[arg-type]
    if not hasattr(target.module, "forward"):
        raise TypeError(f"Target attention module at layer {layer_index} has no forward method")

    captured: dict[str, AttentionSnapshot] = {}

    def _capture(snapshot: AttentionSnapshot) -> None:
        captured["snapshot"] = AttentionSnapshot(
            q=(None if snapshot.q is None else snapshot.q.detach().cpu()),
            k=(None if snapshot.k is None else snapshot.k.detach().cpu()),
            v=(None if snapshot.v is None else snapshot.v.detach().cpu()),
            attn_mask=(None if snapshot.attn_mask is None else snapshot.attn_mask.detach().cpu()),
            logits=(None if snapshot.logits is None else snapshot.logits.detach().cpu()),
            probs=(None if snapshot.probs is None else snapshot.probs.detach().cpu()),
            head_out=(None if snapshot.head_out is None else snapshot.head_out.detach().cpu()),
            output=(None if snapshot.output is None else snapshot.output.detach().cpu()),
        )

    with patched_attention(target.module, capture=_capture, capture_logits=True, capture_probs=True):
        _ = adapter.forward(inputs)

    if "snapshot" not in captured:
        raise RuntimeError(f"Failed to capture attention snapshot for layer {layer_index}")
    return captured["snapshot"]


@torch.inference_mode()
def attention_weights_for_layer(
    model,
    input_ids: Optional[torch.Tensor],
    layer_index: int,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    kind: str = "self",
) -> torch.Tensor:
    """Return attention probabilities (B, H, T, S) for a given layer."""
    snapshot = attention_snapshot_for_layer(
        model,
        input_ids,
        layer_index,
        attn_mask=attn_mask,
        enc_input_ids=enc_input_ids,
        dec_input_ids=dec_input_ids,
        enc_padding_mask=enc_padding_mask,
        dec_self_mask=dec_self_mask,
        stack=stack,
        kind=kind,
    )
    if snapshot.probs is None:
        raise RuntimeError(f"Attention probabilities were not captured for layer {layer_index}")
    return snapshot.probs


@torch.inference_mode()
def attention_entropy_for_layer(
    model,
    input_ids: Optional[torch.Tensor],
    layer_index: int,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    kind: str = "self",
) -> torch.Tensor:
    """Return per-head attention entropy (B, H, T) at the target layer."""
    probs = attention_weights_for_layer(
        model,
        input_ids,
        layer_index,
        attn_mask=attn_mask,
        enc_input_ids=enc_input_ids,
        dec_input_ids=dec_input_ids,
        enc_padding_mask=enc_padding_mask,
        dec_self_mask=dec_self_mask,
        stack=stack,
        kind=kind,
    )
    return entropy_from_probs(probs, dim=-1)


__all__ = [
    "attention_snapshot_for_layer",
    "attention_weights_for_layer",
    "attention_entropy_for_layer",
]
