from __future__ import annotations

import torch

from runtime.attention_factory import build_attention
from runtime.block_config import BlockConfig, build_block_config_from_model
from runtime.block_shared import CausalSelfAttentionBlockBase
from runtime.blocks import prepare_banded_attention_mask
from specs.config import ModelConfig


class BandedAttentionBlock(CausalSelfAttentionBlockBase):
    def __init__(self, cfg: ModelConfig, bandwidth: int, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        bc = block_cfg or build_block_config_from_model(cfg)
        self.bandwidth = int(bandwidth)
        super().__init__(cfg, bc, drop_path)
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        attn_mask = prepare_banded_attention_mask(
            x,
            mask,
            num_heads=self.cfg.n_heads,
            bandwidth=self.bandwidth,
        )
        return super().forward(x, attn_mask, cache)


__all__ = [
    "BandedAttentionBlock",
]
