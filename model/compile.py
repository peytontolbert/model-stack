from __future__ import annotations

import torch


def maybe_compile(model: torch.nn.Module, *, backend: str = "inductor", mode: str | None = None, fullgraph: bool = False):
    try:
        compiled = torch.compile(model, backend=backend, mode=mode, fullgraph=fullgraph)  # type: ignore[attr-defined]
        return compiled
    except Exception:
        return model


