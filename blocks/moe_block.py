import torch
import torch.nn as nn

from specs.config import ModelConfig
from tensor.norms import RMSNorm
from tensor.regularization import StochasticDepth
from attn import topk_router, combine_expert_outputs, load_balance_loss

from .config import BlockConfig, build_block_config_from_model
from .transformer_block import TransformerBlock
from attn.factory import build_attention
from tensor.mlp import MLP


class MoEMLP(nn.Module):
    def __init__(self, hidden_size: int, ff_size: int, num_experts: int, k: int = 1, dropout_p: float = 0.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.ff_size = int(ff_size)
        self.num_experts = int(num_experts)
        self.k = int(k)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(hidden_size, ff_size) for _ in range(num_experts)])
        self.dropout = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # router logits -> top-k assignments
        logits = self.router(x)
        routes = topk_router(logits, k=self.k)  # expected to return (indices, weights)
        expert_out = []
        for i, expert in enumerate(self.experts):
            expert_out.append(expert(x))
        y = combine_expert_outputs(expert_out, routes)
        y = self.dropout(y)
        # Optional: load balance loss if provided by router util
        try:
            l_aux = load_balance_loss(logits, routes, num_experts=self.num_experts)
        except Exception:
            l_aux = None
        return y, l_aux


class MoEBlock(TransformerBlock):
    def __init__(self, cfg: ModelConfig, num_experts: int = 4, k: int = 1, **overrides):
        bc: BlockConfig = build_block_config_from_model(cfg, **overrides)
        attn = build_attention(cfg, n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads, attn_dropout=bc.attn_dropout, use_rope=bc.use_rope, rope_theta=bc.rope_theta)
        drop_path = float(overrides.pop("drop_path", 0.0))
        super().__init__(cfg, attn, block_cfg=bc, drop_path=drop_path)
        # Replace MLP with MoE
        self.moe = MoEMLP(cfg.d_model, cfg.d_ff, num_experts=num_experts, k=k, dropout_p=bc.mlp_dropout)

    def _forward_core(self, x: torch.Tensor, mask: torch.Tensor | None, cache=None) -> torch.Tensor:
        if self.bc.norm_policy == "prenorm":
            a = self.attn.forward(self.n1(x), None, None, mask, cache)
            x = x + self.drop_path(self.resid_dropout(a))
            m, _ = self.moe(self.n2(x))
            x = x + self.drop_path(self.resid_dropout(m))
            return x
        a = self.attn.forward(x, None, None, mask, cache)
        x = self.n1(x + self.drop_path(self.resid_dropout(a)))
        m, _ = self.moe(x)
        x = self.n2(x + self.drop_path(self.resid_dropout(m)))
        return x


