from dataclasses import dataclass
from typing import Literal, Optional

from specs.config import ModelConfig


@dataclass
class BlockConfig:
    d_model: int
    d_ff: int
    n_heads: int
    n_kv_heads: Optional[int] = None  # None -> same as n_heads (MHA)

    # Wiring policy and components
    norm_policy: Literal["prenorm", "postnorm"] = "prenorm"
    norm_type: Literal["rms", "layer"] = "rms"
    activation: Literal["gelu", "silu", "swiglu", "geglu", "reglu"] = "swiglu"

    # Dropouts
    resid_dropout: float = 0.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0

    # Positional bias/injection
    use_rope: bool = True
    rope_theta: float = 1e6
    use_alibi: bool = False
    use_rpb: bool = False
    rpb_max_distance: int = 128

    # Misc
    checkpoint_forward: bool = False
    residual_scale: float = 1.0


def build_block_config_from_model(cfg: ModelConfig, **overrides) -> BlockConfig:
    """Construct a BlockConfig from a ModelConfig with optional overrides.

    Defaults map to common LLaMA-style pre-norm + RMSNorm + SwiGLU.
    """
    n_kv = overrides.pop("n_kv_heads", None)
    bc = BlockConfig(
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        n_heads=cfg.n_heads,
        n_kv_heads=n_kv,
        norm_policy=overrides.pop("norm_policy", "prenorm"),
        norm_type=overrides.pop("norm_type", "rms"),
        activation=overrides.pop("activation", "swiglu"),
        resid_dropout=float(overrides.pop("resid_dropout", 0.0)),
        attn_dropout=float(overrides.pop("attn_dropout", 0.0)),
        mlp_dropout=float(overrides.pop("mlp_dropout", 0.0)),
        use_rope=bool(overrides.pop("use_rope", True)),
        rope_theta=float(overrides.pop("rope_theta", getattr(cfg, "rope_theta", 1e6))),
        use_alibi=bool(overrides.pop("use_alibi", False)),
        use_rpb=bool(overrides.pop("use_rpb", False)),
        rpb_max_distance=int(overrides.pop("rpb_max_distance", 128)),
        checkpoint_forward=bool(overrides.pop("checkpoint_forward", False)),
        residual_scale=float(overrides.pop("residual_scale", 1.0)),
    )
    if overrides:
        # Leftover unknown keys provided by caller
        extra = ", ".join(sorted(overrides.keys()))
        raise ValueError(f"Unknown BlockConfig override keys: {extra}")
    return bc


