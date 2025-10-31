import torch
import torch.nn as nn

from specs.config import ModelConfig
from tensor.masking import build_segment_bidir_mask

from .config import BlockConfig, build_block_config_from_model
from .shared import CausalSelfAttentionBlockBase
from attn.factory import build_attention


class SegmentBidirAttentionBlock(CausalSelfAttentionBlockBase):
    """Bidirectional within segments; expects per-batch segment IDs in forward.

    Forward signature accepts `segment_ids: (B,T)`.
    """

    def __init__(self, cfg: ModelConfig, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        bc = block_cfg or build_block_config_from_model(cfg)
        self.cfg = cfg
        self.bc = bc
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )
        super().__init__(cfg, bc, drop_path)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor, mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        B, T, _ = x.shape
        seg = build_segment_bidir_mask(segment_ids.to(device=x.device))  # (B,T,T) True means masked across segments
        base = seg.unsqueeze(1).expand(B, self.cfg.n_heads, T, T)  # (B,H,T,T)
        if mask is not None:
            base = base | mask
        return super().forward(x, base, cache)


