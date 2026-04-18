from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from interpret.causal.head_patching import causal_trace_heads_restore_table
from interpret.causal.head_patching import _wrap_forward_capture_heads, _wrap_forward_patch_heads
from interpret.model_adapter import ModelInputs, coerce_model_inputs, get_model_adapter, resolve_model_score
from runtime.attention_modules import EagerAttention


def _evaluate_selected_heads(
    *,
    selected: List[Tuple[int, int]],
    targets,
    clean_heads: Dict[int, torch.Tensor],
    adapter,
    corrupted_inputs: ModelInputs,
    model: nn.Module,
    position: int,
    target_token_id: Optional[int],
    target_feature_index: Optional[int],
    score_fn,
    base_score: torch.Tensor,
    denom: torch.Tensor,
) -> float:
    if not selected:
        return 0.0
    mapping: Dict[int, List[int]] = {}
    for li, h in selected:
        mapping.setdefault(int(li), []).append(int(h))
    restored = []
    try:
        for li, heads in mapping.items():
            if not (0 <= li < len(targets)):
                continue
            attn = targets[li].module
            if not isinstance(attn, EagerAttention):
                continue
            orig, new = _wrap_forward_patch_heads(attn, clean_heads, li, heads=heads)
            restored.append((attn, orig))
            attn.forward = new  # type: ignore[assignment]
        outputs_patch = adapter.forward(corrupted_inputs)
    finally:
        for attn, orig in restored:
            attn.forward = orig  # type: ignore[assignment]
    patched_score, _, _ = resolve_model_score(
        model,
        outputs_patch,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    return float(((patched_score - base_score) / denom).item())


@torch.inference_mode()
def greedy_head_recovery(
    model: nn.Module,
    *,
    clean_inputs: Optional[ModelInputs] = None,
    corrupted_inputs: Optional[ModelInputs] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    corrupted_input_ids: Optional[torch.Tensor] = None,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    k: int = 5,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    stack: Optional[str] = None,
    kind: str = "self",
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    score_fn=None,
) -> Dict[str, object]:
    """Greedy selection of heads to patch for maximal target-logit recovery.

    Returns dict with keys: selected (list of (layer, head)), table ((L,H) recovery),
    and curve (list of recovery after each addition).
    """
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
    table = causal_trace_heads_restore_table(
        model,
        clean_inputs=clean_inputs,
        corrupted_inputs=corrupted_inputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        stack=stack,
        kind=kind,
        score_fn=score_fn,
    )  # (L,H)
    L, H = table.shape
    # Flatten candidates sorted by recovery
    vals = []
    for li in range(L):
        for h in range(H):
            vals.append(((li, h), float(table[li, h].item())))
    vals.sort(key=lambda x: x[1], reverse=True)

    selected: List[Tuple[int, int]] = []
    curve: List[float] = []
    outputs_clean = adapter.forward(clean_inputs)
    clean_score, target_token_id, target_feature_index = resolve_model_score(
        model,
        outputs_clean,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    outputs_cor = adapter.forward(corrupted_inputs)
    base_score, _, _ = resolve_model_score(
        model,
        outputs_cor,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    denom = (clean_score - base_score).abs() + 1e-8

    clean_heads: Dict[int, torch.Tensor] = {}
    wrappers = []
    for li, target in enumerate(targets):
        attn = target.module
        if not isinstance(attn, EagerAttention):
            continue
        orig, new = _wrap_forward_capture_heads(attn, clean_heads, li)
        wrappers.append((attn, orig))
        attn.forward = new  # type: ignore[assignment]
    try:
        _ = adapter.forward(clean_inputs)
    finally:
        for attn, orig in wrappers:
            attn.forward = orig  # type: ignore[assignment]

    for (li, h), _v in vals:
        if len(selected) >= int(k):
            break
        selected.append((li, h))
        curve.append(
            _evaluate_selected_heads(
                selected=selected,
                targets=targets,
                clean_heads=clean_heads,
                adapter=adapter,
                corrupted_inputs=corrupted_inputs,
                model=model,
                position=position,
                target_token_id=target_token_id,
                target_feature_index=target_feature_index,
                score_fn=score_fn,
                base_score=base_score,
                denom=denom,
            )
        )

    return {"selected": selected, "table": table, "curve": curve}
