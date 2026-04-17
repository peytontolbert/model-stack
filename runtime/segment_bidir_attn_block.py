from __future__ import annotations

import torch

from runtime.attention_factory import build_attention
from runtime.block_config import BlockConfig, build_block_config_from_model
from runtime.block_shared import CausalSelfAttentionBlockBase
from runtime.blocks import prepare_segment_bidir_attention_mask
from specs.config import ModelConfig


class SegmentBidirAttentionBlock(CausalSelfAttentionBlockBase):
    """Bidirectional within segments; expects per-batch segment IDs in forward."""

    def __init__(self, cfg: ModelConfig, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        bc = block_cfg or build_block_config_from_model(cfg)
        super().__init__(cfg, bc, drop_path)
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor, mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        base = prepare_segment_bidir_attention_mask(
            x,
            segment_ids,
            mask,
            num_heads=self.cfg.n_heads,
        )
        return super().forward(x, base, cache)


__all__ = [
    "SegmentBidirAttentionBlock",
]
