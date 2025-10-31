from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from specs.config import ModelConfig
from model.factory import build_model
from model.checkpoint import load_config, load_pretrained
from attn.kv_cache import init_kv_cache


@dataclass
class RuntimeConfig:
    model_dir: str
    device: Optional[str] = None  # e.g., "cuda", "cpu"
    dtype: Optional[str] = None   # "float16", "bfloat16", "float32"
    kv_pagesize: int = 512


class ModelRuntime:
    def __init__(self, cfg: ModelConfig, model: torch.nn.Module, device: torch.device, dtype: torch.dtype, kv_pagesize: int) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.dtype = dtype
        self.kv_pagesize = int(kv_pagesize)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_dir(cls, model_dir: Optional[str] = None, *, device: Optional[str] = None, dtype: Optional[str] = None, kv_pagesize: int = 512) -> "ModelRuntime":
        indir = model_dir or os.environ.get("MODEL_DIR")
        if not indir:
            raise ValueError("MODEL_DIR environment variable not set and model_dir not provided")
        cfg = load_config(indir)
        model = build_model(cfg)
        model = load_pretrained(model, indir)
        # Resolve device/dtype
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dmap = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dt = dmap.get(dtype or cfg.dtype, torch.bfloat16)
        model = model.to(device=dev, dtype=dt)
        return cls(cfg, model, dev, dt, kv_pagesize)

    def allocate_cache(self, batch_size: int):
        # Derive heads and head_dim
        n_layers = int(self.cfg.n_layers)
        n_kv_heads = int(getattr(self.cfg, "n_kv_heads", self.cfg.n_heads))
        head_dim = int(self.cfg.d_model) // int(self.cfg.n_heads)
        return init_kv_cache(
            batch=int(batch_size),
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            pagesize=self.kv_pagesize,
            dtype=self.dtype,
            device=self.device,
        )


