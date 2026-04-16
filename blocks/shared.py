import torch
import torch.nn as nn

from specs.config import ModelConfig
from tensor.norms import RMSNorm
from tensor.mlp import MLP
from tensor.regularization import StochasticDepth
from runtime.blocks import apply_attention_biases, execute_attention_mlp_block

from .config import BlockConfig


class CausalSelfAttentionBlockBase(nn.Module):
    """Shared implementation for causal self-attention blocks that accept a precomputed boolean mask.

    Children should set `self.attn` and may store additional fields (e.g., window sizes),
    then call `super().__init__(cfg, bc, drop_path)`.
    Forward expects `base_mask` broadcastable to (B,H,T,S) when converted or added.
    """

    def __init__(self, cfg: ModelConfig, bc: BlockConfig, drop_path: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.bc = bc
        Norm = RMSNorm if bc.norm_type == "rms" else nn.LayerNorm
        self.n1 = Norm(cfg.d_model)
        self.n2 = Norm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, activation=bc.activation, dropout_p=bc.mlp_dropout)
        self.resid_dropout = nn.Dropout(bc.resid_dropout) if bc.resid_dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        if getattr(self.bc, "use_rpb", False):
            D = int(self.bc.rpb_max_distance)
            self.rpb_table = nn.Parameter(torch.zeros(cfg.n_heads, 2 * D - 1))
        else:
            self.rpb_table = None

    def _apply_pos_bias(self, x: torch.Tensor, base_mask: torch.Tensor | None) -> torch.Tensor | None:
        return apply_attention_biases(
            x,
            base_mask,
            num_heads=self.cfg.n_heads,
            use_alibi=bool(self.bc.use_alibi),
            rpb_table=self.rpb_table,
            rpb_max_distance=(int(self.bc.rpb_max_distance) if self.rpb_table is not None else None),
        )

    def forward(self, x: torch.Tensor, base_mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        attn_mask = self._apply_pos_bias(x, base_mask)
        return execute_attention_mlp_block(
            x,
            attn_fn=lambda y: self.attn.forward(y, None, None, attn_mask, cache),
            mlp_fn=self.mlp,
            n1=self.n1,
            n2=self.n2,
            resid_dropout=self.resid_dropout,
            drop_path=self.drop_path,
            residual_scale=self.bc.residual_scale,
            norm_policy=self.bc.norm_policy,
        )
