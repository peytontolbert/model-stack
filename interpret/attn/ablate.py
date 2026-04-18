from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Dict, Iterable, Optional

import torch.nn as nn

from runtime.attention_modules import EagerAttention

from interpret.model_adapter import get_model_adapter, patched_attention


@contextmanager
def ablate_attention_heads(
    model: nn.Module,
    mapping: Dict[int, Iterable[int]],
    *,
    stack: Optional[str] = None,
    kind: str = "self",
):
    """Temporarily zero specified attention heads during forward.

    The wrapper preserves the real runtime attention path and only replaces the
    selected head outputs immediately before the output projection.
    """
    adapter = get_model_adapter(model)
    with ExitStack() as stack_ctx:
        for layer_idx, heads in mapping.items():
            attn = adapter.attention_target(int(layer_idx), stack=stack, kind=kind).module  # type: ignore[arg-type]
            if not isinstance(attn, EagerAttention):
                continue
            zero_set = {int(h) for h in heads}

            def _patch(head_out, *, _zero_set=zero_set):
                out = head_out.clone()
                for h in _zero_set:
                    if 0 <= h < out.shape[1]:
                        out[:, h].zero_()
                return out

            stack_ctx.enter_context(patched_attention(attn, patch_heads=_patch))
        yield


__all__ = ["ablate_attention_heads"]
