"""FSDP helpers for running HunyuanVideo-Avatar on memory-constrained GPUs.

This adapter deliberately keeps the Avatar transformer on CPU until FSDP owns
placement. The upstream multi-GPU launcher constructs one full CUDA replica per
rank, which cannot fit on 24 GB GPUs.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


@dataclass(frozen=True)
class AvatarFSDPConfig:
    device_id: int
    use_orig_params: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = False


def _add_avatar_root(avatar_root: str | Path) -> None:
    root = str(Path(avatar_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def avatar_transformer_block_types(avatar_root: str | Path) -> set[type[torch.nn.Module]]:
    """Return the actual Hunyuan Avatar layer types to shard as FSDP units."""
    _add_avatar_root(avatar_root)
    from hymm_sp.modules.models_audio import DoubleStreamBlock, SingleStreamBlock
    from hymm_sp.modules.token_refiner import IndividualTokenRefinerBlock

    return {DoubleStreamBlock, IndividualTokenRefinerBlock, SingleStreamBlock}


def avatar_auto_wrap_policy(avatar_root: str | Path):
    block_types = avatar_transformer_block_types(avatar_root)
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=block_types,
    )


def build_avatar_transformer_cpu(args, avatar_root: str | Path) -> torch.nn.Module:
    """Instantiate Avatar's transformer on CPU, never on a CUDA device."""
    _add_avatar_root(avatar_root)
    from hymm_sp.constants import PRECISION_TO_TYPE
    from hymm_sp.modules import load_model

    return load_model(
        args,
        in_channels=args.latent_channels,
        out_channels=args.latent_channels,
        factor_kwargs={"device": "cpu", "dtype": PRECISION_TO_TYPE[args.precision]},
    )


def convert_avatar_linears_to_fp8_cpu(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    args,
    avatar_root: str | Path,
) -> None:
    """Apply Avatar's native FP8 module conversion while parameters are on CPU."""
    _add_avatar_root(avatar_root)
    from hymm_sp.constants import PRECISION_TO_TYPE
    from hymm_sp.modules.fp8_optimization import convert_fp8_linear

    convert_fp8_linear(
        model,
        str(checkpoint_path),
        original_dtype=PRECISION_TO_TYPE[args.precision],
    )
    # Upstream sets ``fp8_scale`` as a plain CPU tensor attribute. Register it
    # as a buffer so its placement follows the FP8 weight through FSDP2 and it
    # is not copied from CPU on every linear call.
    for module in model.modules():
        scale = getattr(module, "fp8_scale", None)
        if isinstance(scale, torch.Tensor) and "fp8_scale" not in module._buffers:
            delattr(module, "fp8_scale")
            module.register_buffer("fp8_scale", scale, persistent=True)


def normalize_residual_fp32_parameters_for_fsdp2(model: torch.nn.Module) -> None:
    """Convert only residual FP32 parameters after native FP8 conversion.

    ``model.to(dtype=...)`` is unsafe here because it would also convert the
    native FP8 parameter storage. FSDP2 still requires each param group to use
    one original dtype, so keep FP8 as-is and make BF16/FP32 compute parameters
    consistently BF16 before the first lazy FSDP initialization.
    """
    assert_cpu_resident(model)
    for parameter in model.parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)


class _AvatarFP8Weight(nn.Module):
    """An FP8 weight as an independently sharded FSDP2 child module."""

    def __init__(self, weight: torch.nn.Parameter, scale: torch.Tensor, compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = weight
        self.register_buffer("scale", scale, persistent=True)
        self.compute_dtype = compute_dtype

    def forward(self, input_tensor: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        # Keep dequantization and matmul inside this FSDP child forward. Its
        # FP8 all-gathered weight may be resharded immediately after return.
        dequant_weight = self.weight.to(self.compute_dtype) * self.scale.to(
            device=self.weight.device, dtype=self.compute_dtype
        )
        return F.linear(input_tensor, dequant_weight, bias)


def split_avatar_fp8_linear_parameters(model: torch.nn.Module, dtype: torch.dtype) -> list[nn.Module]:
    """Separate native FP8 weights from BF16 biases before FSDP2 wrapping.

    Avatar's conversion stores an FP8 ``weight`` and BF16 ``bias`` inside the
    same Linear. FSDP2 requires a uniform original dtype per managed group.
    Each FP8 weight is therefore moved to a child module; the parent Linear
    retains only its BF16 bias and calls the child in its forward path.
    """
    holders: list[nn.Module] = []
    for layer in model.modules():
        if not isinstance(layer, nn.Linear) or layer.weight is None:
            continue
        if layer.weight.dtype != torch.float8_e4m3fn:
            continue
        if hasattr(layer, "fp8_weight_holder"):
            holders.append(layer.fp8_weight_holder)
            continue
        scale = getattr(layer, "fp8_scale", None)
        if not isinstance(scale, torch.Tensor):
            raise RuntimeError("Native Avatar FP8 linear is missing fp8_scale")
        weight = layer.weight
        holder = _AvatarFP8Weight(weight, scale, dtype)
        layer.register_parameter("weight", None)
        if "fp8_scale" in layer._buffers:
            del layer._buffers["fp8_scale"]
        elif hasattr(layer, "fp8_scale"):
            delattr(layer, "fp8_scale")
        layer.add_module("fp8_weight_holder", holder)

        def fp8_forward(input_tensor, linear=layer, weight_holder=holder, compute_dtype=dtype):
            return weight_holder(input_tensor, linear.bias)

        layer.forward = fp8_forward
        holders.append(holder)
    return holders


def load_monolithic_checkpoint_cpu(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    load_key: str = "module",
) -> None:
    """Hydrate a CPU model from Avatar's existing monolithic checkpoint."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if load_key != ".":
        payload = payload[load_key]
    model.load_state_dict(payload, strict=False)
    del payload


def assert_cpu_resident(model: torch.nn.Module) -> None:
    cuda_parameter = next((name for name, value in model.named_parameters() if value.is_cuda), None)
    if cuda_parameter is not None:
        raise RuntimeError(f"Avatar model must be CPU-resident before FSDP wrapping; found CUDA parameter {cuda_parameter}.")


def normalize_floating_point_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Make Avatar's explicit FP32 norms compatible with FSDP flattening.

    Avatar's standard BF16 build retains a small number of FP32 parameters.
    FSDP requires a uniform dtype inside each flattened wrap unit. This must run
    before FSDP and must not be used after FP8 conversion.
    """
    assert_cpu_resident(model)
    model.to(dtype=dtype)


def wrap_avatar_transformer_fsdp(
    model: torch.nn.Module,
    *,
    avatar_root: str | Path,
    config: AvatarFSDPConfig,
) -> FSDP:
    """Shard CPU-resident Avatar transformer blocks across local FSDP ranks.

    Do not change this to ``FSDP(model.cuda())``. That performs the exact full
    CUDA materialization that fails on 24 GB cards before FSDP can shard it.
    """
    assert_cpu_resident(model)
    return FSDP(
        model,
        auto_wrap_policy=avatar_auto_wrap_policy(avatar_root),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.device("cuda", config.device_id),
        use_orig_params=config.use_orig_params,
        sync_module_states=False,
        limit_all_gathers=config.limit_all_gathers,
        forward_prefetch=config.forward_prefetch,
    )


def wrap_avatar_transformer_fsdp2(
    model: torch.nn.Module,
    *,
    avatar_root: str | Path,
    mesh,
) -> torch.nn.Module:
    """Apply FSDP2 bottom-up without FSDP1's homogeneous flatten requirement.

    FSDP2 shards each parameter independently. This permits Avatar's FP8
    linear weights, BF16 parameters, and FP32 normalization parameters to
    coexist within the same transformer block.
    """
    from torch.distributed.fsdp import fully_shard

    assert_cpu_resident(model)
    block_types = avatar_transformer_block_types(avatar_root)
    # Native FP8 linears have an FP8-only child. Wrap them before their BF16
    # parents so each FSDP2 parameter group is dtype-homogeneous.
    fp8_holders = [module for module in model.modules() if isinstance(module, _AvatarFP8Weight)]
    for holder in reversed(fp8_holders):
        fully_shard(holder, mesh=mesh, reshard_after_forward=True)
    blocks = [module for module in model.modules() if isinstance(module, tuple(block_types))]
    for block in reversed(blocks):
        fully_shard(block, mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)
    return model


def fsdp_parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def save_rank_local_shard(
    model: FSDP,
    output_dir: str | Path,
    *,
    rank: int,
) -> Path:
    """Atomically write this rank's CPU-offloaded FSDP state dictionary."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / f"avatar_transformer.rank{rank:02d}.pt"
    temporary_path = final_path.with_suffix(".pt.tmp")
    state_config = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, state_config):
        torch.save(model.state_dict(), temporary_path)
    os.replace(temporary_path, final_path)
    return final_path


def load_rank_local_shard(model: FSDP, shard_dir: str | Path, *, rank: int) -> None:
    """Load the local FSDP state dictionary without reconstructing the monolith."""
    shard_path = Path(shard_dir) / f"avatar_transformer.rank{rank:02d}.pt"
    if not shard_path.is_file():
        raise FileNotFoundError(f"Missing FSDP shard for rank {rank}: {shard_path}")
    state_config = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, state_config):
        state = torch.load(shard_path, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(state, strict=True)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Avatar FSDP shard did not load cleanly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def save_rank_local_fsdp2_shard(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    rank: int,
) -> Path:
    """Atomically persist this rank's FSDP2 DTensor state dictionary."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / f"avatar_transformer.rank{rank:02d}.pt"
    temporary_path = final_path.with_suffix(".pt.tmp")
    torch.save(get_model_state_dict(model), temporary_path)
    os.replace(temporary_path, final_path)
    return final_path


def load_rank_local_fsdp2_shard(model: torch.nn.Module, shard_dir: str | Path, *, rank: int) -> None:
    """Restore this rank's FSDP2 DTensor state dictionary."""
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    shard_path = Path(shard_dir) / f"avatar_transformer.rank{rank:02d}.pt"
    if not shard_path.is_file():
        raise FileNotFoundError(f"Missing FSDP2 shard for rank {rank}: {shard_path}")
    state = torch.load(shard_path, map_location="cpu", weights_only=False)
    incompatible = set_model_state_dict(model, state)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Avatar FSDP2 shard did not load cleanly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def write_shard_manifest(
    output_dir: str | Path,
    *,
    source_checkpoint: str | Path,
    world_size: int,
) -> Path:
    """Describe a completed, deterministic rank-local shard set."""
    destination = Path(output_dir)
    final_path = destination / "avatar_transformer.manifest.json"
    temporary_path = final_path.with_suffix(".json.tmp")
    payload = {
        "format": "hunyuan-avatar-fsdp-sharded-v1",
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "world_size": world_size,
        "shards": [f"avatar_transformer.rank{rank:02d}.pt" for rank in range(world_size)],
    }
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, final_path)
    return final_path
