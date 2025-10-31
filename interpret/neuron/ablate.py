from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor.mlp import MLP
from tensor.activations import silu as act_silu, gelu as act_gelu


def _wrap_mlp_forward_zero_channels(mlp: MLP, channels: List[int]):
    orig_forward = mlp.forward
    ch = torch.tensor(sorted(set(int(c) for c in channels)), dtype=torch.long)

    def forward(x: torch.Tensor) -> torch.Tensor:
        if mlp.gated:
            x_proj = mlp.w_in(x)
            a, b = x_proj.chunk(2, dim=-1)
            name = mlp.activation_name.lower()
            if name in ("swiglu", "gated-silu"):
                x_mid = act_silu(a) * b
            elif name == "geglu":
                x_mid = act_gelu(a) * b
            elif name == "reglu":
                x_mid = F.relu(a) * b
            else:
                x_mid = act_silu(a) * b
        else:
            x_mid = mlp._act(mlp.w_in(x))
        # Zero selected intermediate channels
        if ch.numel() > 0:
            x_mid[..., ch] = 0
        x_mid = mlp.dropout(x_mid)
        return mlp.w_out(x_mid)

    return orig_forward, forward


@contextmanager
def ablate_mlp_channels(model: nn.Module, mapping: Dict[int, Iterable[int]]):
    """Temporarily zero selected MLP channels in `blocks.{layer}.mlp` forward.

    mapping: {layer_index: [channels...]}
    """
    origs = []
    try:
        for li, chans in mapping.items():
            blk = model.blocks[int(li)]
            mlp = getattr(blk, "mlp", None)
            if not isinstance(mlp, MLP):
                continue
            orig, new = _wrap_mlp_forward_zero_channels(mlp, list(chans))
            origs.append((mlp, orig))
            mlp.forward = new  # type: ignore
        yield
    finally:
        for mlp, orig in origs:
            mlp.forward = orig  # type: ignore


