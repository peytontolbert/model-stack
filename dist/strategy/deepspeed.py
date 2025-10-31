from __future__ import annotations

from typing import Any

import torch


def wrap_deepspeed(model: torch.nn.Module, *, precision: str = "bf16", zero_stage: int = 2) -> torch.nn.Module:
    import deepspeed  # type: ignore

    cfg: dict[str, Any] = {
        "train_batch_size": 1,
        "zero_optimization": {"stage": int(zero_stage)},
        "bf16": {"enabled": (precision == "bf16")},
        "fp16": {"enabled": (precision == "fp16")},
    }
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=cfg)
    return engine


