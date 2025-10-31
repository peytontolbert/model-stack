import torch.nn as nn

from specs.config import ModelConfig
from .interfaces import Attention
from .eager import EagerAttention
from .flash import FlashAttention
from .triton import TritonAttention
from .xformers import XFormersAttention


def build_attention(cfg: ModelConfig, **overrides) -> Attention:
    # Route by cfg.attn_impl; explicit classes for clarity, all share EagerAttention core
    impl = getattr(cfg, "attn_impl", "eager")
    name = str(impl).lower()
    if name in ("eager", "torch", "sdpa"):
        return EagerAttention(cfg, **overrides)
    if name in ("flash", "flash2"):
        return FlashAttention(cfg, **overrides)
    if name in ("triton",):
        return TritonAttention(cfg, **overrides)
    if name in ("xformers", "xformers2"):
        return XFormersAttention(cfg, **overrides)
    return EagerAttention(cfg, **overrides)


