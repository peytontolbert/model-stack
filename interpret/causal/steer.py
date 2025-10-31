from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable

import torch
import torch.nn as nn


@contextmanager
def steer_residual(model: nn.Module, mapping: Dict[int, torch.Tensor], scale: float = 1.0):
    """Add a residual direction at specified block outputs during forward.

    mapping: {layer_index: direction} where direction is (D,) or (1,1,D) or (B,T,D).
    Returns a context manager that adds `scale * direction` to block output.
    """
    handles = []
    try:
        for li, vec in mapping.items():
            blk = model.blocks[int(li)]
            v = vec
            def hook(_m: nn.Module, _inp, out: torch.Tensor):
                nonlocal v
                if not isinstance(out, torch.Tensor):
                    return out
                if v.ndim == 1:
                    add = v.view(1, 1, -1).to(device=out.device, dtype=out.dtype)
                else:
                    add = v.to(device=out.device, dtype=out.dtype)
                return out + float(scale) * add
            handles.append(blk.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


