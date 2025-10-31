from __future__ import annotations

from typing import Dict

import torch

from tensor.shard import attn_flops, mlp_flops, estimate_activation_bytes_per_token


def _get_cfg(model) -> object:
    return getattr(model, "cfg", None)


def estimate_layer_costs(model, *, seq_len: int, batch_size: int = 1, dtype: str = "bf16") -> Dict[str, object]:
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
    L = len(model.blocks)

    per_layer = []
    total_flops = 0
    bytes_per_token = estimate_activation_bytes_per_token(D=d_model, H=n_heads, expand=expand, dtype=dtype)
    for i in range(L):
        attn = attn_flops(B=batch_size, H=n_heads, T=seq_len, S=seq_len, D=head_dim)
        mlp = mlp_flops(B=batch_size, T=seq_len, D=d_model, expand=expand)
        fl = attn + mlp
        total_flops += fl
        per_layer.append({"layer": i, "attn_flops": attn, "mlp_flops": mlp, "total_flops": fl, "act_bytes_per_token": bytes_per_token})
    return {"per_layer": per_layer, "total_flops": total_flops, "bytes_per_token": bytes_per_token}


