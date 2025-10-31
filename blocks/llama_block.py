import torch.nn as nn

from specs.config import ModelConfig
from .config import build_block_config_from_model, BlockConfig
from .transformer_block import TransformerBlock
from attn.factory import build_attention


class LlamaBlock(TransformerBlock):
    def __init__(self, cfg: ModelConfig, n_kv_heads: int | None = None, **overrides):
        bc: BlockConfig = build_block_config_from_model(
            cfg,
            n_kv_heads=n_kv_heads,
            norm_policy=overrides.pop("norm_policy", "prenorm"),
            norm_type=overrides.pop("norm_type", "rms"),
            activation=overrides.pop("activation", "swiglu"),
            resid_dropout=overrides.pop("resid_dropout", 0.0),
            attn_dropout=overrides.pop("attn_dropout", 0.0),
            mlp_dropout=overrides.pop("mlp_dropout", 0.0),
            use_rope=overrides.pop("use_rope", True),
            rope_theta=overrides.pop("rope_theta", getattr(cfg, "rope_theta", 1e6)),
            use_alibi=overrides.pop("use_alibi", False),
            checkpoint_forward=overrides.pop("checkpoint_forward", False),
        )
        attn = build_attention(cfg, n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads, attn_dropout=bc.attn_dropout, use_rope=bc.use_rope, rope_theta=bc.rope_theta)
        drop_path = float(overrides.pop("drop_path", 0.0))
        super().__init__(cfg, attn, block_cfg=bc, drop_path=drop_path)


