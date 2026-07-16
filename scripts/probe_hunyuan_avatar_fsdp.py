"""Construct Avatar's real transformer on CPU, then verify two-rank FSDP placement.

Run with Avatar's normal config arguments. Set HUNYUAN_AVATAR_ROOT to the
upstream checkout. This probe intentionally does not hydrate checkpoint weights;
it validates the memory-critical model construction and sharding phase first.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def main() -> None:
    avatar_root = os.environ["HUNYUAN_AVATAR_ROOT"]
    if avatar_root not in sys.path:
        sys.path.insert(0, avatar_root)

    from hymm_sp.config import parse_args
    from runtime.hunyuan_avatar_fsdp import (
        AvatarFSDPConfig,
        build_avatar_transformer_cpu,
        convert_avatar_linears_to_fp8_cpu,
        fsdp_parameter_bytes,
        normalize_floating_point_dtype,
        wrap_avatar_transformer_fsdp,
    )

    args = parse_args()
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    model = build_avatar_transformer_cpu(args, avatar_root)
    if args.use_fp8:
        convert_avatar_linears_to_fp8_cpu(model, args.ckpt, args, avatar_root)
    else:
        normalize_floating_point_dtype(model, torch.bfloat16)
    cpu_bytes = fsdp_parameter_bytes(model)
    wrapped = wrap_avatar_transformer_fsdp(
        model,
        avatar_root=avatar_root,
        config=AvatarFSDPConfig(device_id=rank),
    )
    torch.cuda.synchronize(rank)
    print(
        f"rank={rank} avatar_fsdp_probe cpu_bytes={cpu_bytes} "
        f"gpu_allocated={torch.cuda.memory_allocated(rank)}"
    )
    del wrapped
    torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
