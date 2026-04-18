from __future__ import annotations

import torch
import torch.nn as nn

from runtime.attention_factory import build_attention
from runtime.attention_interfaces import Attention
from runtime.block_config import BlockConfig, build_block_config_from_model
from runtime.blocks import (
    apply_attention_biases,
    execute_attention_mlp_block,
    execute_parallel_attention_mlp_block,
    prepare_cross_attention_mask,
    prepare_encoder_attention_mask,
)
from runtime.moe import combine_expert_outputs, load_balance_loss, topk_router
from runtime.ops import linear_module as runtime_linear_module
from specs.config import ModelConfig
from tensor.mlp import MLP
from tensor.norms import RMSNorm
from tensor.regularization import StochasticDepth


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, attn: Attention, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        super().__init__()
        bc = block_cfg or build_block_config_from_model(cfg)
        self.cfg = cfg
        self.bc = bc
        Norm = RMSNorm if bc.norm_type == "rms" else nn.LayerNorm
        if Norm is RMSNorm:
            self.n1 = Norm(cfg.d_model, eps=getattr(cfg, "rms_norm_eps", 1e-6))
            self.n2 = Norm(cfg.d_model, eps=getattr(cfg, "rms_norm_eps", 1e-6))
        else:
            self.n1 = Norm(cfg.d_model)
            self.n2 = Norm(cfg.d_model)
        self.attn = attn
        self.mlp = MLP(
            cfg.d_model,
            cfg.d_ff,
            activation=bc.activation,
            dropout_p=bc.mlp_dropout,
            bias=(bc.mlp_bias if bc.mlp_bias is not None else True),
        )
        self.resid_dropout = nn.Dropout(bc.resid_dropout) if bc.resid_dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        if getattr(self.bc, "use_rpb", False):
            D = int(self.bc.rpb_max_distance)
            self.rpb_table = nn.Parameter(torch.zeros(cfg.n_heads, 2 * D - 1))
        else:
            self.rpb_table = None

    def _forward_core(self, x: torch.Tensor, mask: torch.Tensor | None, cache=None, position_embeddings=None, position_ids=None) -> torch.Tensor:
        attn_mask = apply_attention_biases(
            x,
            mask,
            num_heads=self.cfg.n_heads,
            use_alibi=bool(self.bc.use_alibi),
            rpb_table=self.rpb_table,
            rpb_max_distance=(int(self.bc.rpb_max_distance) if self.rpb_table is not None else None),
        )
        return execute_attention_mlp_block(
            x,
            attn_fn=lambda y: self.attn.forward(
                y,
                None,
                None,
                attn_mask,
                cache,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            ),
            mlp_fn=self.mlp,
            n1=self.n1,
            n2=self.n2,
            resid_dropout=self.resid_dropout,
            drop_path=self.drop_path,
            residual_scale=self.bc.residual_scale,
            norm_policy=self.bc.norm_policy,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, cache=None, position_embeddings=None, position_ids=None) -> torch.Tensor:
        if self.bc.checkpoint_forward and self.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(lambda y: self._forward_core(y, mask, cache, position_embeddings, position_ids), x, use_reentrant=False)
        return self._forward_core(x, mask, cache, position_embeddings, position_ids)


class LlamaBlock(TransformerBlock):
    def __init__(self, cfg: ModelConfig, n_kv_heads: int | None = None, **overrides):
        bc: BlockConfig = build_block_config_from_model(
            cfg,
            n_kv_heads=n_kv_heads,
            norm_policy=overrides.pop("norm_policy", "prenorm"),
            norm_type=overrides.pop("norm_type", "rms"),
            activation=overrides.pop("activation", "swiglu"),
            resid_dropout=overrides.pop("resid_dropout", 0.0),
            attn_dropout=overrides.pop("attn_dropout", 0.0),
            mlp_dropout=overrides.pop("mlp_dropout", 0.0),
            mlp_bias=overrides.pop("mlp_bias", False),
            use_rope=overrides.pop("use_rope", True),
            rope_theta=overrides.pop("rope_theta", getattr(cfg, "rope_theta", 1e6)),
            use_alibi=overrides.pop("use_alibi", False),
            use_rpb=overrides.pop("use_rpb", False),
            rpb_max_distance=overrides.pop("rpb_max_distance", 128),
            checkpoint_forward=overrides.pop("checkpoint_forward", False),
        )
        attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )
        drop_path = float(overrides.pop("drop_path", 0.0))
        super().__init__(cfg, attn, block_cfg=bc, drop_path=drop_path)


class GPTBlock(TransformerBlock):
    def __init__(self, cfg: ModelConfig, **overrides):
        bc: BlockConfig = build_block_config_from_model(
            cfg,
            n_kv_heads=overrides.pop("n_kv_heads", None),
            norm_policy=overrides.pop("norm_policy", "postnorm"),
            norm_type=overrides.pop("norm_type", "layer"),
            activation=overrides.pop("activation", "gelu"),
            resid_dropout=overrides.pop("resid_dropout", 0.0),
            attn_dropout=overrides.pop("attn_dropout", 0.0),
            mlp_dropout=overrides.pop("mlp_dropout", 0.0),
            use_rope=overrides.pop("use_rope", False),
            rope_theta=overrides.pop("rope_theta", getattr(cfg, "rope_theta", 1e6)),
            use_alibi=overrides.pop("use_alibi", False),
            use_rpb=overrides.pop("use_rpb", False),
            rpb_max_distance=overrides.pop("rpb_max_distance", 128),
            checkpoint_forward=overrides.pop("checkpoint_forward", False),
        )
        attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )
        drop_path = float(overrides.pop("drop_path", 0.0))
        super().__init__(cfg, attn, block_cfg=bc, drop_path=drop_path)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        super().__init__()
        bc = block_cfg or build_block_config_from_model(cfg, use_rope=False)
        self.cfg = cfg
        self.bc = bc
        Norm = RMSNorm if bc.norm_type == "rms" else nn.LayerNorm
        self.n1 = Norm(cfg.d_model)
        self.n2 = Norm(cfg.d_model)
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=False,
            is_causal=False,
        )
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


class CrossAttentionBlock(nn.Module):
    """Decoder cross-attention block.

    Expects decoder hidden states as `x` and encoder states as `enc`, with optional masks.
    """

    def __init__(self, cfg: ModelConfig, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        super().__init__()
        bc = block_cfg or build_block_config_from_model(cfg)
        self.cfg = cfg
        self.bc = bc
        Norm = RMSNorm if bc.norm_type == "rms" else nn.LayerNorm
        self.n1 = Norm(cfg.d_model)
        self.n2 = Norm(cfg.d_model)
        self.cross = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=False,
            rope_theta=bc.rope_theta,
            is_causal=False,
        )
        self.mlp = MLP(cfg.d_model, cfg.d_ff, activation=bc.activation, dropout_p=bc.mlp_dropout)
        self.resid_dropout = nn.Dropout(bc.resid_dropout) if bc.resid_dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prepared_enc_mask = prepare_cross_attention_mask(
            x,
            enc,
            enc_mask,
            num_heads=self.cfg.n_heads,
        )
        return execute_attention_mlp_block(
            x,
            attn_fn=lambda y: self.cross.forward(y, enc, enc, prepared_enc_mask),
            mlp_fn=self.mlp,
            n1=self.n1,
            n2=self.n2,
            resid_dropout=self.resid_dropout,
            drop_path=self.drop_path,
            residual_scale=self.bc.residual_scale,
            norm_policy=self.bc.norm_policy,
        )


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, block_cfg: BlockConfig | None = None, drop_path: float = 0.0):
        super().__init__()
        bc = block_cfg or build_block_config_from_model(cfg)
        self.self_attn_block = TransformerBlock(cfg, attn=None, block_cfg=bc, drop_path=drop_path)
        self._attn_built = False
        self.cross_block = CrossAttentionBlock(cfg, block_cfg=bc, drop_path=drop_path)

    def _ensure_self_attn(self):
        if not self._attn_built:
            cfg = self.self_attn_block.cfg
            bc = self.self_attn_block.bc
            self.self_attn_block.attn = build_attention(
                cfg,
                n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
                attn_dropout=bc.attn_dropout,
                use_rope=bc.use_rope,
                rope_theta=bc.rope_theta,
                is_causal=True,
            )
            self._attn_built = True

    def forward(self, x: torch.Tensor, enc: torch.Tensor, self_mask: torch.Tensor | None = None, enc_mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        self._ensure_self_attn()
        x = self.self_attn_block(x, self_mask, cache)
        x = self.cross_block(x, enc, attn_mask=self_mask, enc_mask=enc_mask)
        return x


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
        self.attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )
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
        logits = runtime_linear_module(x, self.router)
        routes = topk_router(logits, k=self.k)
        expert_out = []
        for expert in self.experts:
            expert_out.append(expert(x))
        y = combine_expert_outputs(expert_out, routes)
        y = self.dropout(y)
        try:
            l_aux = load_balance_loss(logits, routes, num_experts=self.num_experts)
        except Exception:
            l_aux = None
        return y, l_aux


class MoEBlock(TransformerBlock):
    def __init__(self, cfg: ModelConfig, num_experts: int = 4, k: int = 1, **overrides):
        drop_path = float(overrides.pop("drop_path", 0.0))
        bc: BlockConfig = build_block_config_from_model(cfg, **overrides)
        attn = build_attention(
            cfg,
            n_kv_heads=bc.n_kv_heads if bc.n_kv_heads is not None else cfg.n_heads,
            attn_dropout=bc.attn_dropout,
            use_rope=bc.use_rope,
            rope_theta=bc.rope_theta,
        )
        super().__init__(cfg, attn, block_cfg=bc, drop_path=drop_path)
        self.moe = MoEMLP(cfg.d_model, cfg.d_ff, num_experts=num_experts, k=k, dropout_p=bc.mlp_dropout)

    def _forward_core(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        cache=None,
        position_embeddings=None,
        position_ids=None,
    ) -> torch.Tensor:
        attn_mask = apply_attention_biases(
            x,
            mask,
            num_heads=self.cfg.n_heads,
            use_alibi=bool(self.bc.use_alibi),
            rpb_table=self.rpb_table,
            rpb_max_distance=(int(self.bc.rpb_max_distance) if self.rpb_table is not None else None),
        )
        return execute_attention_mlp_block(
            x,
            attn_fn=lambda y: self.attn.forward(
                y,
                None,
                None,
                attn_mask,
                cache,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            ),
            mlp_fn=lambda y: self.moe(y)[0],
            n1=self.n1,
            n2=self.n2,
            resid_dropout=self.resid_dropout,
            drop_path=self.drop_path,
            residual_scale=1.0,
            norm_policy=self.bc.norm_policy,
        )


__all__ = [
    "TransformerBlock",
    "LlamaBlock",
    "GPTBlock",
    "EncoderBlock",
    "CrossAttentionBlock",
    "DecoderBlock",
    "ParallelTransformerBlock",
    "MoEMLP",
    "MoEBlock",
]
