import torch.nn as nn

from specs.config import ModelConfig
from .llama_block import LlamaBlock
from .gpt_block import GPTBlock
from .parallel_block import ParallelTransformerBlock
from .local_attn_block import LocalAttentionBlock
from .prefix_lm_block import PrefixLMBlock
from .banded_attn_block import BandedAttentionBlock
from .dilated_local_attn_block import DilatedLocalAttentionBlock
from .block_sparse_attn_block import BlockSparseAttentionBlock
from .strided_attn_block import StridedAttentionBlock
from .window_pattern_attn_block import WindowPatternAttentionBlock
from .segment_bidir_attn_block import SegmentBidirAttentionBlock
from .encoder_block import EncoderBlock
from .decoder_block import DecoderBlock
from .cross_attn_block import CrossAttentionBlock
from .moe_block import MoEBlock
from .registry import get_block_builder, register_block
from .schedules import drop_path_linear
from .init import init_transformer_stack
from tensor.init import deepnet_residual_scale


def build_block(variant: str, cfg: ModelConfig, **overrides) -> nn.Module:
    name = variant.lower()
    if name in ("llama", "llm-llama"):
        return LlamaBlock(cfg, **overrides)
    if name in ("gpt", "gpt2", "gptj", "gpt-neox"):
        return GPTBlock(cfg, **overrides)
    if name in ("parallel", "parallel-residual"):
        return ParallelTransformerBlock(cfg, **overrides)
    if name in ("local", "local-attn"):
        return LocalAttentionBlock(cfg, **overrides)
    if name in ("prefix", "prefix-lm"):
        return PrefixLMBlock(cfg, **overrides)
    if name in ("banded", "banded-attn"):
        return BandedAttentionBlock(cfg, **overrides)
    if name in ("dilated", "dilated-local"):
        return DilatedLocalAttentionBlock(cfg, **overrides)
    if name in ("blocksparse", "block-sparse"):
        return BlockSparseAttentionBlock(cfg, **overrides)
    if name in ("strided", "stride"):
        return StridedAttentionBlock(cfg, **overrides)
    if name in ("window", "window-pattern", "windowed"):
        return WindowPatternAttentionBlock(cfg, **overrides)
    if name in ("segment", "segment-bidir", "seg"):
        return SegmentBidirAttentionBlock(cfg, **overrides)
    if name in ("encoder",):
        return EncoderBlock(cfg, **overrides)
    if name in ("decoder",):
        return DecoderBlock(cfg, **overrides)
    if name in ("cross", "cross-attn"):
        return CrossAttentionBlock(cfg, **overrides)
    if name in ("moe",):
        return MoEBlock(cfg, **overrides)
    # Registry fallback
    try:
        builder = get_block_builder(name)
        return builder(cfg, **overrides)
    except Exception:
        pass
    raise ValueError(f"Unknown block variant: {variant}")


def build_block_stack(
    cfg: ModelConfig,
    variant: str = "llama",
    drop_path_max: float = 0.0,
    init_recipe: str | None = None,
    residual_policy: str | None = None,
    **overrides,
) -> nn.ModuleList:
    blocks: list[nn.Module] = []
    schedule = drop_path_linear(cfg.n_layers, max_drop=float(drop_path_max))
    # DeepNet residual scaling support
    if residual_policy and residual_policy.lower() == "deepnet":
        overrides = {**overrides, "block_cfg": None, "residual_scale": deepnet_residual_scale(cfg.n_layers)}
    for i in range(cfg.n_layers):
        b = build_block(variant, cfg, drop_path=schedule[i], **overrides)
        blocks.append(b)
    stack = nn.ModuleList(blocks)
    if init_recipe is not None:
        init_transformer_stack(stack, recipe=init_recipe)
    return stack


# Pre-register built-in blocks into registry for discoverability
try:
    register_block("llama", lambda cfg, **ov: LlamaBlock(cfg, **ov))
    register_block("gpt", lambda cfg, **ov: GPTBlock(cfg, **ov))
    register_block("parallel", lambda cfg, **ov: ParallelTransformerBlock(cfg, **ov))
    register_block("local", lambda cfg, **ov: LocalAttentionBlock(cfg, **ov))
    register_block("prefix", lambda cfg, **ov: PrefixLMBlock(cfg, **ov))
    register_block("banded", lambda cfg, **ov: BandedAttentionBlock(cfg, **ov))
    register_block("dilated", lambda cfg, **ov: DilatedLocalAttentionBlock(cfg, **ov))
    register_block("blocksparse", lambda cfg, **ov: BlockSparseAttentionBlock(cfg, **ov))
    register_block("strided", lambda cfg, **ov: StridedAttentionBlock(cfg, **ov))
    register_block("window", lambda cfg, **ov: WindowPatternAttentionBlock(cfg, **ov))
    register_block("segment", lambda cfg, **ov: SegmentBidirAttentionBlock(cfg, **ov))
    register_block("encoder", lambda cfg, **ov: EncoderBlock(cfg, **ov))
    register_block("decoder", lambda cfg, **ov: DecoderBlock(cfg, **ov))
    register_block("cross", lambda cfg, **ov: CrossAttentionBlock(cfg, **ov))
    register_block("moe", lambda cfg, **ov: MoEBlock(cfg, **ov))
except Exception:
    pass


