import torch.nn as nn

from tensor.init import (
    init_rmsnorm_,
    xavier_uniform_linear,
    init_qkv_proj_,
    init_out_proj_scaled_,
    zero_out_proj_bias,
)
from tensor.norms import RMSNorm


def _init_llama_block(block: nn.Module, n_layers: int | None = None) -> None:
    for m in block.modules():
        if isinstance(m, RMSNorm):
            init_rmsnorm_(m, eps=getattr(m, "eps", 1e-6), gain=1.0)
    # Attention projections
    attn = getattr(block, "attn", None)
    if attn is not None:
        if hasattr(attn, "w_q"):
            init_qkv_proj_(attn.w_q, block.cfg.d_model, block.cfg.n_heads)
        if hasattr(attn, "w_k"):
            init_qkv_proj_(attn.w_k, block.cfg.d_model, getattr(attn, "n_kv_heads", block.cfg.n_heads))
        if hasattr(attn, "w_v"):
            init_qkv_proj_(attn.w_v, block.cfg.d_model, getattr(attn, "n_kv_heads", block.cfg.n_heads))
        if hasattr(attn, "w_o"):
            if n_layers is not None:
                init_out_proj_scaled_(attn.w_o, n_layers)
            else:
                xavier_uniform_linear(attn.w_o)
            zero_out_proj_bias(attn.w_o)
    # MLP
    mlp = getattr(block, "mlp", None)
    if mlp is not None and hasattr(mlp, "w_in") and hasattr(mlp, "w_out"):
        xavier_uniform_linear(mlp.w_in)
        xavier_uniform_linear(mlp.w_out)


def _init_gpt_block(block: nn.Module, n_layers: int | None = None) -> None:
    # GPT often uses LayerNorm; PyTorch default init is fine, adjust linear layers
    attn = getattr(block, "attn", None)
    if attn is not None:
        if hasattr(attn, "w_q"):
            xavier_uniform_linear(attn.w_q)
        if hasattr(attn, "w_k"):
            xavier_uniform_linear(attn.w_k)
        if hasattr(attn, "w_v"):
            xavier_uniform_linear(attn.w_v)
        if hasattr(attn, "w_o"):
            if n_layers is not None:
                init_out_proj_scaled_(attn.w_o, n_layers)
            else:
                xavier_uniform_linear(attn.w_o)
            zero_out_proj_bias(attn.w_o)
    mlp = getattr(block, "mlp", None)
    if mlp is not None and hasattr(mlp, "w_in") and hasattr(mlp, "w_out"):
        xavier_uniform_linear(mlp.w_in)
        xavier_uniform_linear(mlp.w_out)


def apply_block_init(block: nn.Module, recipe: str = "llama", n_layers: int | None = None) -> nn.Module:
    name = recipe.lower()
    if name == "llama":
        _init_llama_block(block, n_layers=n_layers)
    elif name == "gpt":
        _init_gpt_block(block, n_layers=n_layers)
    else:
        _init_llama_block(block, n_layers=n_layers)
    return block


def init_transformer_stack(blocks: nn.ModuleList, recipe: str = "llama") -> nn.ModuleList:
    n_layers = len(blocks)
    for b in blocks:
        apply_block_init(b, recipe=recipe, n_layers=n_layers)
    return blocks


