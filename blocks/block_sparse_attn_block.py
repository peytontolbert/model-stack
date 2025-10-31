import torch
import torch.nn as nn

from specs.config import ModelConfig
from tensor.masking import build_block_sparse_mask, broadcast_mask

from .config import BlockConfig, build_block_config_from_model
from .shared import CausalSelfAttentionBlockBase
from attn.factory import build_attention


def _default_pattern(seq_len: int, block: int, device=None) -> torch.Tensor:
    # Allow only same-block attention (diagonal blocks)
    N = (seq_len + block - 1) // block
    pat = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(N):
        pat[i, i] = True
    return pat


class BlockSparseAttentionBlock(CausalSelfAttentionBlockBase):
    def __init__(self, cfg: ModelConfig, block_size: int = 64, pattern: torch.Tensor | None = None, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        bc = block_cfg or build_block_config_from_model(cfg)
        self.cfg = cfg
        self.bc = bc
        self.block_size = int(block_size)
        self.register_buffer("pattern", pattern if pattern is not None else torch.tensor([]), persistent=False)
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )
        super().__init__(cfg, bc, drop_path)

    def _pattern(self, T: int, device) -> torch.Tensor:
        if self.pattern.numel() == 0:
            return _default_pattern(T, self.block_size, device=device)
        return self.pattern.to(device=device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        B, T, _ = x.shape
        sparse = build_block_sparse_mask(T, self.block_size, self._pattern(T, x.device), device=x.device, dtype=torch.bool)
        attn_mask = broadcast_mask(batch_size=B, num_heads=self.cfg.n_heads, tgt_len=T, src_len=T, causal_mask=sparse)
        if mask is not None:
            attn_mask = attn_mask | mask
        return super().forward(x, attn_mask, cache)


