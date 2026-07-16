"""Real Avatar FP8/FSDP2 placement probe, optionally with sequence parallel."""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def main() -> None:
    avatar_root = os.environ["HUNYUAN_AVATAR_ROOT"]
    if avatar_root not in sys.path:
        sys.path.insert(0, avatar_root)

    from hymm_sp.config import parse_args
    from hymm_sp.modules.parallel_states import initialize_sequence_parallel_state
    from runtime.hunyuan_avatar_fsdp import (
        build_avatar_transformer_cpu,
        convert_avatar_linears_to_fp8_cpu,
        fsdp_parameter_bytes,
        wrap_avatar_transformer_fsdp2,
    )

    args = parse_args()
    if not args.use_fp8:
        raise ValueError("This probe is specifically for Avatar's native FP8 checkpoint.")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    initialize_sequence_parallel_state(world_size)

    print(f"rank={rank} constructing Avatar on CPU then hydrating native FP8 modules", flush=True)
    model = build_avatar_transformer_cpu(args, avatar_root)
    convert_avatar_linears_to_fp8_cpu(model, args.ckpt, args, avatar_root)
    cpu_bytes = fsdp_parameter_bytes(model)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    wrapped = wrap_avatar_transformer_fsdp2(model, avatar_root=avatar_root, mesh=mesh)
    torch.cuda.synchronize(rank)
    print(
        f"rank={rank} avatar_fp8_fsdp2_probe cpu_bytes={cpu_bytes} "
        f"gpu_allocated={torch.cuda.memory_allocated(rank)} "
        f"gpu_reserved={torch.cuda.memory_reserved(rank)}",
        flush=True,
    )
    del wrapped
    torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
