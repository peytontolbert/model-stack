import torch
import torch.nn as nn

from specs.config import ModelConfig
from runtime.blocks import default_block_sparse_pattern, prepare_block_sparse_attention_mask

from .config import BlockConfig, build_block_config_from_model
from .shared import CausalSelfAttentionBlockBase
from attn.factory import build_attention


class BlockSparseAttentionBlock(CausalSelfAttentionBlockBase):
    def __init__(self, cfg: ModelConfig, block_size: int = 64, pattern: torch.Tensor | None = None, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        bc = block_cfg or build_block_config_from_model(cfg)
        self.block_size = int(block_size)
        super().__init__(cfg, bc, drop_path)
        self.register_buffer("pattern", pattern if pattern is not None else torch.tensor([]), persistent=False)
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )

    def _pattern(self, T: int, device) -> torch.Tensor:
        if self.pattern.numel() == 0:
            return default_block_sparse_pattern(T, self.block_size, device=device)
        return self.pattern.to(device=device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        attn_mask = prepare_block_sparse_attention_mask(
            x,
            mask,
            num_heads=self.cfg.n_heads,
            block_size=self.block_size,
            pattern=self._pattern(x.shape[1], x.device),
        )
        return super().forward(x, attn_mask, cache)
