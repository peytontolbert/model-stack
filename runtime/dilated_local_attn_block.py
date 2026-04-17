from __future__ import annotations

import torch

from runtime.attention_factory import build_attention
from runtime.block_config import BlockConfig, build_block_config_from_model
from runtime.block_shared import CausalSelfAttentionBlockBase
from runtime.blocks import prepare_dilated_attention_mask
from specs.config import ModelConfig


class DilatedLocalAttentionBlock(CausalSelfAttentionBlockBase):
    def __init__(self, cfg: ModelConfig, window: int = 128, dilation: int = 2, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        bc = block_cfg or build_block_config_from_model(cfg)
        self.window = int(window)
        self.dilation = int(dilation)
        super().__init__(cfg, bc, drop_path)
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        base = prepare_dilated_attention_mask(
            x,
            mask,
            num_heads=self.cfg.n_heads,
            window=self.window,
            dilation=self.dilation,
        )
        return super().forward(x, base, cache)


__all__ = [
    "DilatedLocalAttentionBlock",
]
