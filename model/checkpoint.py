from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
import torch

from specs.config import ModelConfig
from tensor.io_safetensors import safetensor_dump, safetensor_load


def save_pretrained(model: torch.nn.Module, cfg: ModelConfig, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    # Save config
    cfg_path = os.path.join(outdir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    # Save weights as safetensors
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    safetensor_dump(weights, os.path.join(outdir, "model.safetensors"))
    return outdir


def load_pretrained(model: torch.nn.Module, indir: str, *, strict: bool = True) -> torch.nn.Module:
    path = os.path.join(indir, "model.safetensors")
    tensors = safetensor_load(path)
    model.load_state_dict(tensors, strict=strict)
    return model


def load_config(indir: str) -> ModelConfig:
    with open(os.path.join(indir, "config.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    return ModelConfig(**d)


