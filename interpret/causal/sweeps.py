from __future__ import annotations

from contextlib import ExitStack
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from interpret.causal.head_patching import _wrap_forward_capture_heads, _wrap_forward_patch_heads
from interpret.causal.slice_patching import output_patching_slice
from interpret.model_adapter import MLPSnapshot, ModelInputs, coerce_model_inputs, get_model_adapter, patched_mlp


def _clean_target_fraction(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, patched_logits: torch.Tensor, *, position: int, target_token_id: Optional[int]) -> float:
    if target_token_id is None:
        target_token_id = int(clean_logits[0, position].argmax().item())
    clean = clean_logits[0, position, target_token_id]
    base = corrupted_logits[0, position, target_token_id]
    patched = patched_logits[0, position, target_token_id]
    return float(((patched - base) / ((clean - base).abs() + 1e-8)).item())


@torch.inference_mode()
def block_output_patch_sweep(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    target_token_id: Optional[int] = None,
    stack: Optional[str] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Patch one residual-stream position at a time for each block target."""
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
    targets = adapter.block_targets(stack=stack)
    names = [target.name for target in targets]

    clean_cache: Dict[str, torch.Tensor] = {}
    handles = []
    try:
        for target in targets:
            def _hook(_module, _inputs, output, *, _name=target.name):
                if isinstance(output, torch.Tensor):
                    clean_cache[_name] = output.detach().cpu().clone()
                return output

            handles.append(target.module.register_forward_hook(_hook))
        clean_logits = adapter.forward(clean_inputs)
    finally:
        for handle in handles:
            handle.remove()
    corrupted_logits = adapter.forward(corrupted_inputs)

    seq_source = clean_inputs.input_ids if clean_inputs.input_ids is not None else clean_inputs.dec_input_ids
    if seq_source is None:
        raise ValueError("A decoder-side token sequence is required to determine sweep length")
    seq_len = int(seq_source.shape[1])
    scores = torch.zeros(len(names), seq_len)
    for row, name in enumerate(names):
        replacement = clean_cache.get(name)
        if replacement is None:
            continue
        for t in range(seq_len):
            with output_patching_slice(model, {name: replacement}, time_slice=slice(t, t + 1)):
                patched_logits = adapter.forward(corrupted_inputs)
            scores[row, t] = _clean_target_fraction(
                clean_logits,
                corrupted_logits,
                patched_logits,
                position=position,
                target_token_id=target_token_id,
            )
    return {"names": names, "scores": scores}


@torch.inference_mode()
def head_patch_sweep(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    target_token_id: Optional[int] = None,
    stack: Optional[str] = None,
    kind: str = "self",
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Patch one head and one source position at a time."""
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
    names = [target.name for target in targets]

    clean_heads: Dict[int, torch.Tensor] = {}
    wrappers = []
    for idx, target in enumerate(targets):
        orig, new = _wrap_forward_capture_heads(target.module, clean_heads, idx)
        wrappers.append((target.module, orig))
        target.module.forward = new  # type: ignore[assignment]
    try:
        clean_logits = adapter.forward(clean_inputs)
    finally:
        for attn, orig in wrappers:
            attn.forward = orig  # type: ignore[assignment]

    corrupted_logits = adapter.forward(corrupted_inputs)
    seq_source = clean_inputs.input_ids if clean_inputs.input_ids is not None else clean_inputs.dec_input_ids
    if seq_source is None:
        raise ValueError("A decoder-side token sequence is required to determine sweep length")
    seq_len = int(seq_source.shape[1])
    head_dim = max((tensor.shape[1] for tensor in clean_heads.values()), default=0)
    scores = torch.zeros(len(names), head_dim, seq_len)

    for row, target in enumerate(targets):
        clean = clean_heads.get(row)
        if clean is None:
            continue
        num_heads = int(clean.shape[1])
        for h in range(num_heads):
            for t in range(seq_len):
                orig, new = _wrap_forward_patch_heads(target.module, clean_heads, row, heads=[h], time_index=t)
                target.module.forward = new  # type: ignore[assignment]
                try:
                    patched_logits = adapter.forward(corrupted_inputs)
                finally:
                    target.module.forward = orig  # type: ignore[assignment]
                scores[row, h, t] = _clean_target_fraction(
                    clean_logits,
                    corrupted_logits,
                    patched_logits,
                    position=position,
                    target_token_id=target_token_id,
                )
    return {"names": names, "scores": scores}


@torch.inference_mode()
def cross_attention_head_patch_sweep(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    target_token_id: Optional[int] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    return head_patch_sweep(
        model,
        clean_inputs=clean_inputs,
        corrupted_inputs=corrupted_inputs,
        clean_input_ids=clean_input_ids,
        corrupted_input_ids=corrupted_input_ids,
        position=position,
        attn_mask=attn_mask,
        target_token_id=target_token_id,
        stack="decoder",
        kind="cross",
        enc_input_ids=enc_input_ids,
        dec_input_ids=dec_input_ids,
        enc_padding_mask=enc_padding_mask,
        dec_self_mask=dec_self_mask,
    )


@torch.inference_mode()
def mlp_neuron_patch_sweep(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    neurons: Iterable[int],
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    target_token_id: Optional[int] = None,
    stack: Optional[str] = None,
    kind: Optional[str] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Patch selected MLP neurons one position at a time."""
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

    neuron_ids = [int(n) for n in neurons]
    if adapter.kind == "encoder_decoder" and kind is None and (stack or "decoder") == "decoder":
        kinds = ["self", "cross"]
    else:
        kinds = [kind]

    targets = []
    block_refs = adapter.block_targets(stack=stack)
    max_layer = (max((ref.layer_index for ref in block_refs), default=-1) + 1)
    for layer_idx in range(max_layer):
        for block_kind in kinds:
            try:
                mlp = adapter.mlp_module(layer_idx, stack=stack, kind=block_kind)
                targets.append((f"{stack or adapter.kind}.{layer_idx}.mlp" if block_kind is None else f"{stack or adapter.kind}.{layer_idx}.{block_kind}.mlp", mlp))
            except Exception:
                continue

    clean_mid: Dict[str, torch.Tensor] = {}
    with ExitStack() as stack_ctx:
        for name, mlp in targets:
            def _capture(snapshot: MLPSnapshot, *, _name=name):
                if snapshot.mlp_mid is not None:
                    clean_mid[_name] = snapshot.mlp_mid.detach().cpu().clone()

            stack_ctx.enter_context(patched_mlp(mlp, capture=_capture))
        clean_logits = adapter.forward(clean_inputs)

    corrupted_logits = adapter.forward(corrupted_inputs)
    seq_source = clean_inputs.input_ids if clean_inputs.input_ids is not None else clean_inputs.dec_input_ids
    if seq_source is None:
        raise ValueError("A decoder-side token sequence is required to determine sweep length")
    seq_len = int(seq_source.shape[1])
    scores = torch.zeros(len(targets), len(neuron_ids), seq_len)
    for row, (name, mlp) in enumerate(targets):
        source = clean_mid.get(name)
        if source is None:
            continue
        for ni, neuron in enumerate(neuron_ids):
            for t in range(seq_len):
                def _patch(mid: torch.Tensor, *, _source=source, _neuron=neuron, _t=t):
                    patched = mid.clone()
                    clean_live = _source.to(device=patched.device, dtype=patched.dtype)
                    patched[:, _t, _neuron].copy_(clean_live[:, _t, _neuron])
                    return patched

                with patched_mlp(mlp, patch_mid=_patch):
                    patched_logits = adapter.forward(corrupted_inputs)
                scores[row, ni, t] = _clean_target_fraction(
                    clean_logits,
                    corrupted_logits,
                    patched_logits,
                    position=position,
                    target_token_id=target_token_id,
                )
    return {"names": [name for name, _ in targets], "neurons": neuron_ids, "scores": scores}


@torch.inference_mode()
def path_patch_effect(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    source_module: str,
    receiver_module: str,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    target_token_id: Optional[int] = None,
    time_slice: Optional[slice] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Patch one source activation and measure downstream receiver recovery."""
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
    name_to_module = adapter.named_modules()
    source = name_to_module.get(source_module)
    receiver = name_to_module.get(receiver_module)
    if source is None or receiver is None:
        raise KeyError("source_module and receiver_module must resolve via model.named_modules()")

    captured: Dict[str, torch.Tensor] = {}
    handles = []
    try:
        def _source_hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured["source_clean"] = output.detach().cpu().clone()
            return output

        def _receiver_clean_hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured["receiver_clean"] = output.detach().cpu().clone()
            return output

        handles.append(source.register_forward_hook(_source_hook))
        handles.append(receiver.register_forward_hook(_receiver_clean_hook))
        clean_logits = adapter.forward(clean_inputs)
    finally:
        for handle in handles:
            handle.remove()

    handles = []
    try:
        def _receiver_cor_hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured["receiver_corrupted"] = output.detach().cpu().clone()
            return output

        handles.append(receiver.register_forward_hook(_receiver_cor_hook))
        corrupted_logits = adapter.forward(corrupted_inputs)
    finally:
        for handle in handles:
            handle.remove()

    handles = []
    try:
        def _receiver_patch_hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured["receiver_patched"] = output.detach().cpu().clone()
            return output

        handles.append(receiver.register_forward_hook(_receiver_patch_hook))
        if time_slice is None:
            patched_ctx = output_patching_slice(model, {source_module: captured["source_clean"]}, time_slice=slice(None))
        else:
            patched_ctx = output_patching_slice(model, {source_module: captured["source_clean"]}, time_slice=time_slice)
        with patched_ctx:
            patched_logits = adapter.forward(corrupted_inputs)
    finally:
        for handle in handles:
            handle.remove()

    clean_recv = captured["receiver_clean"].float()
    base_recv = captured["receiver_corrupted"].float()
    patched_recv = captured["receiver_patched"].float()
    base_dist = torch.norm(base_recv - clean_recv)
    patched_dist = torch.norm(patched_recv - clean_recv)
    receiver_restore = float(((base_dist - patched_dist) / (base_dist + 1e-8)).item())
    target_restore = _clean_target_fraction(
        clean_logits,
        corrupted_logits,
        patched_logits,
        position=position,
        target_token_id=target_token_id,
    )
    return {
        "source_module": source_module,
        "receiver_module": receiver_module,
        "receiver_restore_fraction": receiver_restore,
        "target_logit_restore_fraction": target_restore,
    }


@torch.inference_mode()
def path_patch_sweep(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    source_modules: Iterable[str],
    receiver_modules: Iterable[str],
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    target_token_id: Optional[int] = None,
    time_slice: Optional[slice] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Evaluate path patching across a source/receiver module grid."""
    sources = [str(name) for name in source_modules]
    receivers = [str(name) for name in receiver_modules]
    if not sources or not receivers:
        raise ValueError("source_modules and receiver_modules must both be non-empty")

    target_restore = torch.zeros(len(sources), len(receivers))
    receiver_restore = torch.zeros(len(sources), len(receivers))
    effects = []
    for i, source_module in enumerate(sources):
        for j, receiver_module in enumerate(receivers):
            effect = path_patch_effect(
                model,
                clean_inputs=clean_inputs,
                corrupted_inputs=corrupted_inputs,
                clean_input_ids=clean_input_ids,
                corrupted_input_ids=corrupted_input_ids,
                source_module=source_module,
                receiver_module=receiver_module,
                position=position,
                attn_mask=attn_mask,
                target_token_id=target_token_id,
                time_slice=time_slice,
                enc_input_ids=enc_input_ids,
                dec_input_ids=dec_input_ids,
                enc_padding_mask=enc_padding_mask,
                dec_self_mask=dec_self_mask,
            )
            target_restore[i, j] = float(effect["target_logit_restore_fraction"])
            receiver_restore[i, j] = float(effect["receiver_restore_fraction"])
            effects.append(effect)
    return {
        "source_modules": sources,
        "receiver_modules": receivers,
        "target_restore": target_restore,
        "receiver_restore": receiver_restore,
        "effects": effects,
    }


__all__ = [
    "block_output_patch_sweep",
    "cross_attention_head_patch_sweep",
    "head_patch_sweep",
    "mlp_neuron_patch_sweep",
    "path_patch_effect",
    "path_patch_sweep",
]
