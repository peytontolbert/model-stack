import torch
import torch.nn as nn

from specs.config import ModelConfig
from tensor.norms import RMSNorm
from tensor.mlp import MLP
from tensor.regularization import StochasticDepth
from tensor.masking import to_additive_mask
from tensor.positional import build_alibi_bias, build_relative_position_indices, relative_position_bias_from_table

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
        attn_mask = base_mask
        B, T, _ = x.shape
        if self.bc.use_alibi:
            bias = build_alibi_bias(self.cfg.n_heads, seq_len=T, device=x.device).to(dtype=x.dtype)
            if attn_mask is None:
                attn_mask = bias
            else:
                if attn_mask.dtype == torch.bool:
                    add = to_additive_mask(attn_mask)
                else:
                    add = attn_mask
                attn_mask = add + bias
        if self.rpb_table is not None:
            idx = build_relative_position_indices(T, T, int(self.bc.rpb_max_distance), device=x.device)
            rpb = relative_position_bias_from_table(idx, self.rpb_table).to(dtype=x.dtype)
            if attn_mask is None:
                attn_mask = rpb
            else:
                if attn_mask.dtype == torch.bool:
                    add = to_additive_mask(attn_mask)
                else:
                    add = attn_mask
                attn_mask = add + rpb
        return attn_mask

    def forward(self, x: torch.Tensor, base_mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        attn_mask = self._apply_pos_bias(x, base_mask)
        if self.bc.norm_policy == "prenorm":
            a = self.attn.forward(self.n1(x), None, None, attn_mask, cache)
            x = x + self.bc.residual_scale * self.drop_path(self.resid_dropout(a))
            m = self.mlp(self.n2(x))
            x = x + self.bc.residual_scale * self.drop_path(self.resid_dropout(m))
            return x
        a = self.attn.forward(x, None, None, attn_mask, cache)
        x = self.n1(x + self.bc.residual_scale * self.drop_path(self.resid_dropout(a)))
        m = self.mlp(x)
        x = self.n2(x + self.bc.residual_scale * self.drop_path(self.resid_dropout(m)))
        return x


