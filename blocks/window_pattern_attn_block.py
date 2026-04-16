import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

from specs.config import ModelConfig
from runtime.blocks import prepare_window_pattern_attention_mask

from .config import BlockConfig, build_block_config_from_model
from .shared import CausalSelfAttentionBlockBase
from attn.factory import build_attention


class WindowPatternAttentionBlock(CausalSelfAttentionBlockBase):
    def __init__(self, cfg: ModelConfig, spans: List[Tuple[int, int]], block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        bc = block_cfg or build_block_config_from_model(cfg)
        self.spans = list(spans)
        super().__init__(cfg, bc, drop_path)
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        base = prepare_window_pattern_attention_mask(
            x,
            mask,
            num_heads=self.cfg.n_heads,
            spans=self.spans,
        )
        return super().forward(x, base, cache)
