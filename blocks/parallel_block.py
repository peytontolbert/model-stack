import torch
import torch.nn as nn

from specs.config import ModelConfig
from tensor.norms import RMSNorm
from tensor.mlp import MLP
from tensor.regularization import StochasticDepth
from runtime.blocks import execute_parallel_attention_mlp_block

from .config import BlockConfig, build_block_config_from_model
from attn.factory import build_attention


class ParallelTransformerBlock(nn.Module):
    """Parallel residual wiring: compute attn and mlp on same input, sum outputs.

    x_out = x + drop(a(x_norm)) + drop(m(x_norm))
    """

    def __init__(self, cfg: ModelConfig, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        super().__init__()
        bc = block_cfg or build_block_config_from_model(cfg)
        self.cfg = cfg
        self.bc = bc
        Norm = RMSNorm if bc.norm_type == "rms" else nn.LayerNorm
        self.n = Norm(cfg.d_model)
        self.attn = build_attention(cfg, n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads, attn_dropout=bc.attn_dropout, use_rope=bc.use_rope, rope_theta=bc.rope_theta)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, activation=bc.activation, dropout_p=bc.mlp_dropout)
        self.resid_dropout = nn.Dropout(bc.resid_dropout) if bc.resid_dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, cache=None) -> torch.Tensor:
        return execute_parallel_attention_mlp_block(
            x,
            attn_fn=lambda y: self.attn.forward(y, None, None, mask, cache),
            mlp_fn=self.mlp,
            norm=self.n,
            resid_dropout=self.resid_dropout,
            drop_path=self.drop_path,
            residual_scale=1.0,
        )
