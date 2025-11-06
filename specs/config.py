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
    head_dim: int | None = None
    attn_impl: Literal["eager", "flash", "triton", "xformers", "sdpa", "torch", "flash2", "xformers2"] = "eager"
    rope_theta: float = 1e6
    dtype: DType = "bfloat16"
    kv_cache_paged: bool = True
    # attention options
    attention_bias: bool = False
    attn_dropout: float = 0.0
    # optional sliding-window length for KV cache during generation
    sliding_window: Optional[int] = None
    version: int = 1
    # tensor-backed op selections (resolved via specs.ops)
    activation: str = "silu"
    norm: str = "rmsnorm"
    positional: str = "apply_rotary"
    masking: str = "build_causal_mask"
    softmax: str = "safe_softmax"
    residual: str = "prenorm"
    # numeric knobs
    rms_norm_eps: float = 1e-6
    # HF rope scaling (optional)
    rope_scaling_type: Optional[str] = None
    rope_scaling_factor: Optional[float] = None
    rope_scaling_original_max_position_embeddings: Optional[int] = None
    rope_scaling_low_freq_factor: Optional[float] = None
    rope_scaling_high_freq_factor: Optional[float] = None
    # HF rotary attention scaling (applied to cos/sin)
    rope_attention_scaling: Optional[float] = None
    # Token ids
    pad_token_id: Optional[int] = None