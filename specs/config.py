# specs/config.py
from dataclasses import dataclass
from typing import Literal, Optional

DType = Literal["float16", "bfloat16", "float32"]

@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    vocab_size: int
    attn_impl: Literal["eager", "flash", "triton", "xformers", "sdpa", "torch", "flash2", "xformers2"] = "eager"
    rope_theta: float = 1e6
    dtype: DType = "bfloat16"
    kv_cache_paged: bool = True
    version: int = 1
    # tensor-backed op selections (resolved via specs.ops)
    activation: str = "silu"
    norm: str = "rmsnorm"
    positional: str = "apply_rotary"
    masking: str = "build_causal_mask"
    softmax: str = "safe_softmax"
    residual: str = "prenorm"