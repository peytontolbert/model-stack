from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


@torch.inference_mode()
def module_importance_scan(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    modules: Optional[Iterable[str]] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    attn_mask: Optional[torch.Tensor] = None,
    mode: str = "logit",  # "logit" | "prob" | "nll"
) -> List[Tuple[str, float]]:
    """Rank modules by importance via output ablation (set output to zero) and measuring score drop.

    Returns list of (module_name, delta) sorted descending by delta (higher -> more important).
    """
    was_training = model.training
    model.eval()

    # Baseline
    logits = model(input_ids, attn_mask)
    if target_token_id is None:
        target_token_id = int(logits[0, position].argmax().item())
    base_logit = logits[0, position, target_token_id].float()
    base_prob = torch.softmax(logits.float(), dim=-1)[0, position, target_token_id]
    base_nll = -torch.log(base_prob.clamp_min(1e-45))

    # Candidate modules: default to leaf blocks and their submodules
    name_to_module = dict(model.named_modules())
    if modules is None:
        mods = []
        for name, mod in name_to_module.items():
            # skip root and top-level model
            if name == "" or name.count(".") == 0:
                continue
            # consider attention/MLP submodules and block outputs
            if name.endswith(".attn") or name.endswith(".mlp"):
                mods.append(name)
            elif name.startswith("blocks.") and name.count(".") == 1:
                mods.append(name)
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
            l = model(input_ids, attn_mask)
        finally:
            h.remove()
        if mode == "logit":
            occl = l[0, position, target_token_id].float()
            delta = (base_logit - occl).item()
        elif mode == "prob":
            pr = torch.softmax(l.float(), dim=-1)[0, position, target_token_id]
            delta = (base_prob - pr).item()
        elif mode == "nll":
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


