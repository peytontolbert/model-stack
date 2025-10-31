from __future__ import annotations

import json
import os
from typing import Dict

import torch

from . import utils
from tensor.io_safetensors import safetensor_dump, safetensor_load


def save_sharded(model: torch.nn.Module, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    rank = utils.get_rank(0)
    path = os.path.join(outdir, f"model-rank{rank:05d}.safetensors")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    safetensor_dump(state, path)
    utils.barrier()
    if utils.is_primary():
        # Write an index file listing shards
        world = utils.get_world_size(1)
        index = {"format": "safetensors_sharded", "shards": [f"model-rank{i:05d}.safetensors" for i in range(world)]}
        with open(os.path.join(outdir, "model.index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    utils.barrier()
    return outdir


def load_any(model: torch.nn.Module, indir: str, *, strict: bool = True) -> torch.nn.Module:
    # Prefer non-sharded if present
    full = os.path.join(indir, "model.safetensors")
    if os.path.exists(full):
        tensors = safetensor_load(full)
        model.load_state_dict(tensors, strict=strict)
        return model
    # Try local shard
    shard = os.path.join(indir, f"model-rank{utils.get_rank(0):05d}.safetensors")
    if os.path.exists(shard):
        tensors = safetensor_load(shard)
        model.load_state_dict(tensors, strict=strict)
        return model
    # Fallback: if index exists and single-rank load env, just load first shard
    idx = os.path.join(indir, "model.index.json")
    if os.path.exists(idx):
        with open(idx, "r", encoding="utf-8") as f:
            meta: Dict[str, object] = json.load(f)
        shards = meta.get("shards", [])  # type: ignore[assignment]
        if isinstance(shards, list) and len(shards) > 0:
            first = os.path.join(indir, str(shards[0]))
            if os.path.exists(first):
                tensors = safetensor_load(first)
                model.load_state_dict(tensors, strict=False)
                return model
    raise FileNotFoundError(f"No checkpoint found in {indir}")


