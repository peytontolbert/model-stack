from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter, resolve_model_score


@torch.inference_mode()
def module_importance_scan(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    *,
    inputs: Optional[ModelInputs] = None,
    modules: Optional[Iterable[str]] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    attn_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    score_fn=None,
    mode: str = "logit",  # "logit" | "prob" | "nll"
) -> List[Tuple[str, float]]:
    """Rank modules by importance via output ablation (set output to zero) and measuring score drop.

    Returns list of (module_name, delta) sorted descending by delta (higher -> more important).
    """
    adapter = get_model_adapter(model)
    if inputs is None:
        inputs = coerce_model_inputs(
            model,
            input_ids,
            attn_mask,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
        )
    was_training = model.training
    model.eval()

    # Baseline
    outputs = adapter.forward(inputs)
    base_score, target_token_id, target_feature_index = resolve_model_score(
        model,
        outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    base_logit = base_score.float()
    if adapter.output_module() is not None:
        base_prob = torch.softmax(outputs.float(), dim=-1)[0, position, target_token_id]
        base_nll = -torch.log(base_prob.clamp_min(1e-45))
    else:
        base_prob = torch.tensor(0.0, device=base_logit.device)
        base_nll = torch.tensor(0.0, device=base_logit.device)

    # Candidate modules: default to leaf blocks and their submodules
    name_to_module = dict(model.named_modules())
    if modules is None:
        mods = []
        for block in adapter.block_targets(stack=stack):
            mods.append(block.name)
            attn_module = getattr(block.module, "attn", None)
            if attn_module is None:
                attn_module = getattr(block.module, "cross", None)
            if isinstance(attn_module, nn.Module):
                mods.append(f"{block.name}.attn")
            if isinstance(getattr(block.module, "mlp", None), nn.Module):
                mods.append(f"{block.name}.mlp")
        modules = mods

    results: List[Tuple[str, float]] = []

    def zero_hook(_m, _inp, out):
        return torch.zeros_like(out) if isinstance(out, torch.Tensor) else out

    for name in modules:
        mod = name_to_module.get(name)
        if mod is None:
            continue
        h = mod.register_forward_hook(lambda m, i, o: zero_hook(m, i, o))
        try:
            l = adapter.forward(inputs)
        finally:
            h.remove()
        if mode == "logit":
            occl, _, _ = resolve_model_score(
                model,
                l,
                position=position,
                target_token_id=target_token_id,
                target_feature_index=target_feature_index,
                score_fn=score_fn,
            )
            occl = occl.float()
            delta = (base_logit - occl).item()
        elif mode == "prob":
            if adapter.output_module() is None:
                raise ValueError("mode='prob' and mode='nll' require token logits")
            pr = torch.softmax(l.float(), dim=-1)[0, position, target_token_id]
            delta = (base_prob - pr).item()
        elif mode == "nll":
            if adapter.output_module() is None:
                raise ValueError("mode='prob' and mode='nll' require token logits")
            pr = torch.softmax(l.float(), dim=-1)[0, position, target_token_id]
            nll = -torch.log(pr.clamp_min(1e-45))
            delta = (nll - base_nll).item()
        else:
            raise ValueError("mode must be 'logit' | 'prob' | 'nll'")
        results.append((name, float(delta)))

    if was_training:
        model.train()
    results.sort(key=lambda x: x[1], reverse=True)
    return results
