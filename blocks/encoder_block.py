import torch
import torch.nn as nn

from specs.config import ModelConfig
from tensor.norms import RMSNorm
from tensor.mlp import MLP
from tensor.regularization import StochasticDepth
from runtime.blocks import execute_attention_mlp_block, prepare_encoder_attention_mask

from .config import BlockConfig, build_block_config_from_model
from attn.factory import build_attention


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        super().__init__()
        bc = block_cfg or build_block_config_from_model(cfg, use_rope=False)
        self.cfg = cfg
        self.bc = bc
        Norm = RMSNorm if bc.norm_type == "rms" else nn.LayerNorm
        self.n1 = Norm(cfg.d_model)
        self.n2 = Norm(cfg.d_model)
        self.attn = build_attention(cfg, n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads, attn_dropout=bc.attn_dropout, use_rope=False, is_causal=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, activation=bc.activation, dropout_p=bc.mlp_dropout)
        self.resid_dropout = nn.Dropout(bc.resid_dropout) if bc.resid_dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        if getattr(self.bc, "use_rpb", False):
            D = int(self.bc.rpb_max_distance)
            self.rpb_table = nn.Parameter(torch.zeros(cfg.n_heads, 2 * D - 1))
        else:
            self.rpb_table = None

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_mask = prepare_encoder_attention_mask(
            x,
            padding_mask,
            num_heads=self.cfg.n_heads,
            rpb_table=self.rpb_table,
            rpb_max_distance=(int(self.bc.rpb_max_distance) if self.rpb_table is not None else None),
        )
        return execute_attention_mlp_block(
            x,
            attn_fn=lambda y: self.attn.forward(y, None, None, attn_mask, None),
            mlp_fn=self.mlp,
            n1=self.n1,
            n2=self.n2,
            resid_dropout=self.resid_dropout,
            drop_path=self.drop_path,
            residual_scale=self.bc.residual_scale,
            norm_policy=self.bc.norm_policy,
        )
