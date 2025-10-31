import torch
import torch.nn as nn

from specs.config import ModelConfig
from tensor.masking import build_strided_mask, broadcast_mask

from .config import BlockConfig, build_block_config_from_model
from .shared import CausalSelfAttentionBlockBase
from attn.factory import build_attention


class StridedAttentionBlock(CausalSelfAttentionBlockBase):
    def __init__(self, cfg: ModelConfig, stride: int = 4, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        bc = block_cfg or build_block_config_from_model(cfg)
        self.cfg = cfg
        self.bc = bc
        self.stride = int(stride)
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )
        super().__init__(cfg, bc, drop_path)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        B, T, _ = x.shape
        st = build_strided_mask(T, self.stride, device=x.device, dtype=torch.bool)
        base = broadcast_mask(batch_size=B, num_heads=self.cfg.n_heads, tgt_len=T, src_len=T, causal_mask=st)
        if mask is not None:
            base = base | mask
        return super().forward(x, base, cache)


