from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


@contextmanager
def output_patching_slice(model: nn.Module, replacements: Dict[str, torch.Tensor], *, time_slice: slice):
    """Patch only a slice of time positions in module outputs.

    replacements: {module_name: tensor} with full output to pull from for the given slice.
    """
    name_to_module: Dict[str, nn.Module] = dict(model.named_modules())
    handles: list = []

    def make_hook(rep: torch.Tensor):
        def _hook(_mod: nn.Module, _inputs, output: torch.Tensor):
            if not isinstance(output, torch.Tensor):
                return output
            x = output.clone()
            x[:, time_slice] = rep[:, time_slice].to(device=x.device, dtype=x.dtype)
            return x
        return _hook

    for name, rep in replacements.items():
        mod = name_to_module.get(name)
        if mod is None:
            continue
        handles.append(mod.register_forward_hook(make_hook(rep)))
    try:
        yield
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


