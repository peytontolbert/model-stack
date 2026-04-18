from __future__ import annotations

from typing import Dict

import torch

from interpret.model_adapter import get_model_adapter
from tensor.shard import attn_flops, mlp_flops, estimate_activation_bytes_per_token


def _get_cfg(model) -> object:
    return getattr(model, "cfg", None)


def estimate_layer_costs(
    model,
    *,
    seq_len: int,
    batch_size: int = 1,
    dtype: str = "bf16",
    stack: str | None = None,
    source_seq_len: int | None = None,
) -> Dict[str, object]:
    """Return per-layer estimated FLOPs and activation bytes using model cfg fields.

    Requires model.cfg with d_model, n_heads, d_ff. Uses rough analytical formulas.
    """
    cfg = _get_cfg(model)
    if cfg is None:
        raise AttributeError("model.cfg not found; cannot estimate costs")
    d_model = int(getattr(cfg, "d_model"))
    n_heads = int(getattr(cfg, "n_heads"))
    d_ff = int(getattr(cfg, "d_ff"))
    head_dim = d_model // n_heads
    expand = max(d_ff // max(d_model, 1), 1)
    adapter = get_model_adapter(model)
    resolved_stack = stack or ("causal" if adapter.kind == "causal" else "encoder" if adapter.kind == "encoder" else "decoder")
    if adapter.kind == "encoder_decoder" and resolved_stack == "decoder":
        L = len(getattr(model, "decoder", []))
    else:
        L = len(adapter.block_targets(stack=resolved_stack))
    src_len = int(source_seq_len if source_seq_len is not None else seq_len)

    per_layer = []
    total_flops = 0
    bytes_per_token = estimate_activation_bytes_per_token(D=d_model, H=n_heads, expand=expand, dtype=dtype)
    for i in range(L):
        if adapter.kind == "encoder_decoder" and resolved_stack == "decoder":
            self_attn = attn_flops(B=batch_size, H=n_heads, T=seq_len, S=seq_len, D=head_dim)
            cross_attn = attn_flops(B=batch_size, H=n_heads, T=seq_len, S=src_len, D=head_dim)
            attn = self_attn + cross_attn
        else:
            attn = attn_flops(B=batch_size, H=n_heads, T=seq_len, S=src_len, D=head_dim)
        mlp = mlp_flops(B=batch_size, T=seq_len, D=d_model, expand=expand)
        fl = attn + mlp
        total_flops += fl
        per_layer.append({"layer": i, "attn_flops": attn, "mlp_flops": mlp, "total_flops": fl, "act_bytes_per_token": bytes_per_token})
    return {"per_layer": per_layer, "total_flops": total_flops, "bytes_per_token": bytes_per_token}
