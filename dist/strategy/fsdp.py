from __future__ import annotations

from typing import Callable, Optional, Set, Type

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy


def _resolve_block_types() -> Set[Type[torch.nn.Module]]:
    types: Set[Type[torch.nn.Module]] = set()
    try:
        # Core transformer block and common variants
        from blocks.transformer_block import TransformerBlock
        from blocks.encoder_block import EncoderBlock
        from blocks.decoder_block import DecoderBlock
        from blocks.gpt_block import GPTBlock
        from blocks.llama_block import LlamaBlock
        from blocks.parallel_block import ParallelTransformerBlock
        from blocks.moe_block import MoEBlock

        types.update(
            {
                TransformerBlock,
                EncoderBlock,
                DecoderBlock,
                GPTBlock,
                LlamaBlock,
                ParallelTransformerBlock,
                MoEBlock,
            }
        )
    except Exception:
        pass
    return types


def make_transformer_auto_wrap_policy(param_limit: int = 1_000_000) -> Callable:
    block_types = _resolve_block_types()

    def policy(module: torch.nn.Module, recurse: bool, unwrapped_params: int) -> bool:
        # Wrap at transformer block granularity or large submodules
        if type(module) in block_types:
            return True
        if unwrapped_params >= int(param_limit):
            return True
        return False

    return policy


def wrap_fsdp(
    model: torch.nn.Module,
    *,
    auto_wrap: bool = True,
    state_offload: bool = False,
    cpu_offload: bool = False,
    param_limit: int = 1_000_000,
) -> torch.nn.Module:
    policy = make_transformer_auto_wrap_policy(param_limit=param_limit) if auto_wrap else always_wrap_policy
    kwargs = {}
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffload

        kwargs["cpu_offload"] = CPUOffload(offload_params=True)
    if state_offload:
        from torch.distributed.fsdp import ShardingStrategy

        kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD
    return FSDP(model.cuda(), auto_wrap_policy=policy, **kwargs)


