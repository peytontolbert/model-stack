import torch
import torch.nn as nn

from specs.config import ModelConfig
from .config import BlockConfig, build_block_config_from_model
from .transformer_block import TransformerBlock
from .cross_attn_block import CrossAttentionBlock


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        super().__init__()
        bc = block_cfg or build_block_config_from_model(cfg)
        self.self_attn_block = TransformerBlock(cfg, attn=None, block_cfg=bc, drop_path=drop_path)  # attn will be built inside override
        # Build self-attn via factory inside a temp Llama/GPT pattern? Simpler: reuse TransformerBlock but pass attention via factory at call time
        # To keep API consistent, replace attn with build_attention on first forward
        self._attn_built = False
        self.cross_block = CrossAttentionBlock(cfg, block_cfg=bc, drop_path=drop_path)

    def _ensure_self_attn(self):
        if not self._attn_built:
            from attn.factory import build_attention
            cfg = self.self_attn_block.cfg
            bc = self.self_attn_block.bc
            self.self_attn_block.attn = build_attention(cfg, n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads, attn_dropout=bc.attn_dropout, use_rope=bc.use_rope, rope_theta=bc.rope_theta, is_causal=True)
            self._attn_built = True

    def forward(self, x: torch.Tensor, enc: torch.Tensor, self_mask: torch.Tensor | None = None, enc_mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        self._ensure_self_attn()
        x = self.self_attn_block(x, self_mask, cache)
        x = self.cross_block(x, enc, attn_mask=self_mask, enc_mask=enc_mask)
        return x


