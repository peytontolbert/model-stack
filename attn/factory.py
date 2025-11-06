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
    # Map requested impl to backend override for EagerAttention
    backend_override = None
    if name in ("torch", "sdpa"):
        backend_override = "torch"
    elif name in ("flash", "flash2"):
        backend_override = "flash2"
    elif name in ("xformers", "xformers2"):
        backend_override = "xformers"
    elif name in ("triton",):
        backend_override = "triton"
    if name in ("eager", "torch", "sdpa", "flash", "flash2", "xformers", "xformers2", "triton"):
        return EagerAttention(cfg, backend_override=backend_override, **overrides)
    if name in ("flash", "flash2"):
        return FlashAttention(cfg, **overrides)
    if name in ("triton",):
        return TritonAttention(cfg, **overrides)
    if name in ("xformers", "xformers2"):
        return XFormersAttention(cfg, **overrides)
    return EagerAttention(cfg, **overrides)


