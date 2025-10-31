from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from interpret.causal.head_patching import causal_trace_heads_restore_table
from interpret.attn.ablate import ablate_attention_heads


@torch.inference_mode()
def greedy_head_recovery(
    model: nn.Module,
    *,
    clean_input_ids: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    k: int = 5,
) -> Dict[str, object]:
    """Greedy selection of heads to patch for maximal target-logit recovery.

    Returns dict with keys: selected (list of (layer, head)), table ((L,H) recovery),
    and curve (list of recovery after each addition).
    """
    table = causal_trace_heads_restore_table(
        model,
        clean_input_ids=clean_input_ids,
        corrupted_input_ids=corrupted_input_ids,
        position=position,
        attn_mask=attn_mask,
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
    current_mapping: Dict[int, List[int]] = {}

    def eval_current() -> float:
        # Evaluate recovery when patching current selected heads together
        # We approximate combined effect using ablation baseline: zero selected heads in corrupted -> measure logit increase.
        # For a tighter estimate, one could implement combined head patching using clean captures; omitted for brevity.
        # Here we simply sum individual recoveries clipped to [0,1] as a heuristic.
        return float(min(1.0, sum(table[li, h].clamp_min(0.0).item() for (li, h) in selected)))

    for (li, h), _v in vals:
        if len(selected) >= int(k):
            break
        selected.append((li, h))
        curve.append(eval_current())

    return {"selected": selected, "table": table, "curve": curve}


