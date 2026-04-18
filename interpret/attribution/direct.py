from __future__ import annotations

from contextlib import ExitStack
from typing import Dict, Optional

import torch
import torch.nn as nn

from runtime.attention_modules import EagerAttention
from runtime.ops import activation as runtime_activation
from runtime.ops import head_output_projection as runtime_head_output_projection
from runtime.ops import linear as runtime_linear
from runtime.ops import resolve_linear_module_tensors as runtime_resolve_linear_module_tensors

from interpret.features.sae import SparseAutoencoder
from interpret.model_adapter import AttentionSnapshot, MLPSnapshot, coerce_model_inputs, get_model_adapter, patched_attention, patched_embedding_output, patched_mlp, resolve_model_score


def _capture_output_with_grad(store: dict, key: str):
    def hook(_module: nn.Module, _inputs, output: torch.Tensor):
        if isinstance(output, torch.Tensor):
            output.retain_grad()
            store[key] = output
        return output

    return hook


def _final_residual_module(adapter, *, stack: Optional[str]) -> nn.Module:
    targets = adapter.block_targets(stack=stack)
    if not targets:
        raise RuntimeError("Model adapter has no residual blocks to trace")
    return targets[-1].module


def component_logit_attribution(
    model: nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    stack: Optional[str] = None,
    score_fn=None,
) -> Dict[str, float]:
    """Gradient-based direct logit attribution over additive residual components."""
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

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    stores: dict[str, torch.Tensor] = {}
    handles = []
    try:
        def _capture_embed(output: torch.Tensor) -> None:
            stores["embed"] = output

        component_specs: list[tuple[str, float]] = []
        with ExitStack() as stack_ctx:
            for block in adapter.block_targets(stack=resolved_stack):
                scale = float(getattr(getattr(block.module, "bc", None), "residual_scale", 1.0))
                attn_module = getattr(block.module, "attn", None)
                if attn_module is None:
                    attn_module = getattr(block.module, "cross", None)
                if isinstance(attn_module, EagerAttention):
                    key = f"{block.name}.attn"

                    def _capture_attn(snapshot: AttentionSnapshot, *, _key=key) -> None:
                        if snapshot.output is not None:
                            stores[_key] = snapshot.output

                    stack_ctx.enter_context(patched_attention(attn_module, capture=_capture_attn))
                    component_specs.append((key, scale))
                mlp_module = getattr(block.module, "mlp", None)
                if isinstance(mlp_module, nn.Module):
                    key = f"{block.name}.mlp"
                    handles.append(mlp_module.register_forward_hook(_capture_output_with_grad(stores, key)))
                    component_specs.append((key, scale))

            final_residual_key = "final_residual"
            final_module = _final_residual_module(adapter, stack=resolved_stack)
            handles.append(final_module.register_forward_hook(_capture_output_with_grad(stores, final_residual_key)))

            stack_ctx.enter_context(
                patched_embedding_output(
                    adapter,
                    inputs=inputs,
                    stack=resolved_stack,
                    capture=_capture_embed,
                    keep_grad=True,
                )
            )
            outputs = adapter.forward(inputs)
            score, target_token_id, target_feature_index = resolve_model_score(
                model,
                outputs,
                position=position,
                target_token_id=target_token_id,
                target_feature_index=target_feature_index,
                score_fn=score_fn,
            )
            score.backward()

        final_residual = stores[final_residual_key]
        grad = final_residual.grad[0, position].detach()
        out: Dict[str, float] = {}
        out["embed"] = float(torch.dot(grad.float(), stores["embed"][0, position].detach().float()).item())

        for key, scale in component_specs:
            if key in stores:
                out[key] = float(scale * torch.dot(grad.float(), stores[key][0, position].detach().float()).item())
        return out
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()


def head_logit_attribution(
    model: nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    *,
    layer_index: int,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    stack: Optional[str] = None,
    kind: str = "self",
    score_fn=None,
) -> torch.Tensor:
    """Per-head direct logit attribution at one layer."""
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
    target = adapter.attention_target(int(layer_index), stack=resolved_stack, kind=kind)  # type: ignore[arg-type]

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    stores: dict[str, object] = {}
    handles = []
    try:
        final_module = _final_residual_module(adapter, stack=resolved_stack)
        handles.append(final_module.register_forward_hook(_capture_output_with_grad(stores, "final_residual")))

        def _capture(snapshot: AttentionSnapshot) -> None:
            stores["attn_snapshot"] = snapshot

        with patched_attention(target.module, capture=_capture):
            outputs = adapter.forward(inputs)
        score, target_token_id, target_feature_index = resolve_model_score(
            model,
            outputs,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        score.backward()

        final_grad = stores["final_residual"].grad[0, position].detach()  # type: ignore[index]
        snapshot = stores["attn_snapshot"]
        if not isinstance(snapshot, AttentionSnapshot) or snapshot.head_out is None:
            raise RuntimeError(f"Failed to capture head outputs for layer {layer_index}")
        head_out = snapshot.head_out.detach()
        attn = target.module
        block_kind = kind if adapter.kind == "encoder_decoder" and resolved_stack == "decoder" else None
        scale = float(getattr(getattr(adapter.block_target(int(layer_index), stack=resolved_stack, kind=block_kind).module, "bc", None), "residual_scale", 1.0))

        o_weight, _ = runtime_resolve_linear_module_tensors(attn.w_o, reference=head_out)
        out = torch.zeros(head_out.shape[1], dtype=torch.float32)
        for h in range(head_out.shape[1]):
            head_only = torch.zeros_like(head_out)
            head_only[:, h] = head_out[:, h]
            projected = runtime_head_output_projection(head_only, o_weight, None)
            out[h] = scale * torch.dot(final_grad.float(), projected[0, position].detach().float())
        return out.cpu()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()


def mlp_neuron_logit_attribution(
    model: nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    *,
    layer_index: int,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    stack: Optional[str] = None,
    kind: Optional[str] = None,
    score_fn=None,
) -> torch.Tensor:
    """Per-neuron direct logit attribution for one MLP block."""
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
    if adapter.kind == "encoder_decoder" and resolved_stack == "decoder" and kind is None:
        raise ValueError("kind must be 'self' or 'cross' for decoder MLP attribution on encoder-decoder models")
    mlp = adapter.mlp_module(int(layer_index), stack=resolved_stack, kind=kind)

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    stores: dict[str, object] = {}
    handles = []
    try:
        final_module = _final_residual_module(adapter, stack=resolved_stack)
        handles.append(final_module.register_forward_hook(_capture_output_with_grad(stores, "final_residual")))

        def _capture(snapshot: MLPSnapshot) -> None:
            stores["mlp_snapshot"] = snapshot

        with patched_mlp(mlp, capture=_capture):
            outputs = adapter.forward(inputs)
        score, target_token_id, target_feature_index = resolve_model_score(
            model,
            outputs,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        score.backward()

        final_grad = stores["final_residual"].grad[0, position].detach()  # type: ignore[index]
        snapshot = stores["mlp_snapshot"]
        if not isinstance(snapshot, MLPSnapshot) or snapshot.mlp_mid is None:
            raise RuntimeError(f"Failed to capture MLP activations for layer {layer_index}")
        mid = snapshot.mlp_mid.detach()[0, position]
        w_out, _ = runtime_resolve_linear_module_tensors(mlp.w_out, reference=snapshot.mlp_mid.detach())
        scale = float(getattr(getattr(adapter.block_target(int(layer_index), stack=resolved_stack, kind=kind).module, "bc", None), "residual_scale", 1.0))
        proj = torch.matmul(final_grad.float(), w_out.float())
        return (scale * mid.float() * proj).cpu()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()


def sae_feature_logit_attribution(
    model: nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    *,
    layer_index: int,
    sae: SparseAutoencoder,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    stack: Optional[str] = None,
    module: str = "block",
    kind: Optional[str] = None,
    score_fn=None,
) -> torch.Tensor:
    """Attribute a target logit to SAE decoder features at one hooked activation site."""
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
    if adapter.kind == "encoder_decoder" and resolved_stack == "decoder" and kind is None:
        raise ValueError("kind must be 'self' or 'cross' for decoder SAE attribution on encoder-decoder models")

    if module == "mlp":
        target_module = adapter.mlp_module(int(layer_index), stack=resolved_stack, kind=kind)
    else:
        target_module = adapter.block_target(int(layer_index), stack=resolved_stack, kind=kind).module

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    stores: dict[str, torch.Tensor] = {}
    handle = target_module.register_forward_hook(_capture_output_with_grad(stores, "features"))
    try:
        outputs = adapter.forward(inputs)
        score, target_token_id, target_feature_index = resolve_model_score(
            model,
            outputs,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        score.backward()

        feats = stores["features"]
        x = feats[0, position].detach()
        grad = feats.grad[0, position].detach()
        codes = runtime_activation(runtime_linear(x.unsqueeze(0), sae.encoder.weight, sae.encoder.bias), "relu").squeeze(0)
        decoder_weight, _ = runtime_resolve_linear_module_tensors(sae.decoder, reference=x.unsqueeze(0))
        basis_scores = torch.matmul(grad.float(), decoder_weight.float())
        return (codes.float() * basis_scores).cpu()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()


__all__ = [
    "component_logit_attribution",
    "head_logit_attribution",
    "mlp_neuron_logit_attribution",
    "sae_feature_logit_attribution",
]
