"""Materialize Avatar checkpoints as deterministic FSDP2 local shards."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def parse_launcher_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dtype", choices=("bf16", "fp8"), required=True)
    return parser.parse_known_args()


def main() -> None:
    launcher_args, avatar_argv = parse_launcher_args()
    sys.argv = [sys.argv[0], *avatar_argv]
    avatar_root = os.environ["HUNYUAN_AVATAR_ROOT"]
    if avatar_root not in sys.path:
        sys.path.insert(0, avatar_root)

    from hymm_sp.config import parse_args
    from hymm_sp.modules.parallel_states import initialize_sequence_parallel_state
    from runtime.hunyuan_avatar_fsdp import (
        build_avatar_transformer_cpu,
        convert_avatar_linears_to_fp8_cpu,
        load_monolithic_checkpoint_cpu,
        save_rank_local_fsdp2_shard,
        split_avatar_fp8_linear_parameters,
        wrap_avatar_transformer_fsdp2,
        write_shard_manifest,
    )

    args = parse_args()
    if launcher_args.shard_dtype == "fp8" and not args.use_fp8:
        raise ValueError("Pass --use-fp8 with Avatar's native FP8 transformer checkpoint.")
    if launcher_args.shard_dtype == "bf16" and args.use_fp8:
        raise ValueError("Do not pass --use-fp8 when materializing homogeneous BF16 shards.")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.cuda.set_device(rank)
    initialize_sequence_parallel_state(world_size)

    print(f"rank={rank} building native {launcher_args.shard_dtype.upper()} Avatar on CPU", flush=True)
    model = build_avatar_transformer_cpu(args, avatar_root)
    if launcher_args.shard_dtype == "fp8":
        convert_avatar_linears_to_fp8_cpu(model, args.ckpt, args, avatar_root)

    # Keep only one source checkpoint payload resident while both CPU models
    # are prepared. The resulting output is rank-local and reusable.
    if rank == 0:
        print(f"rank=0 hydrating {launcher_args.shard_dtype.upper()} checkpoint", flush=True)
        load_monolithic_checkpoint_cpu(model, args.ckpt)
        print("rank=0 hydration complete", flush=True)
    dist.barrier()
    if rank != 0:
        print(f"rank=1 hydrating {launcher_args.shard_dtype.upper()} checkpoint", flush=True)
        load_monolithic_checkpoint_cpu(model, args.ckpt)
        print("rank=1 hydration complete", flush=True)
    dist.barrier()

    if launcher_args.shard_dtype == "fp8":
        split_avatar_fp8_linear_parameters(model, torch.bfloat16)

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    wrapped = wrap_avatar_transformer_fsdp2(model, avatar_root=avatar_root, mesh=mesh)
    print(f"rank={rank} exporting FSDP2 {launcher_args.shard_dtype.upper()} shard", flush=True)
    shard = save_rank_local_fsdp2_shard(wrapped, launcher_args.output_dir, rank=rank)
    print(f"rank={rank} wrote {shard}", flush=True)
    dist.barrier()
    if rank == 0:
        manifest = write_shard_manifest(
            launcher_args.output_dir,
            source_checkpoint=args.ckpt,
            world_size=world_size,
        )
        print(f"wrote {manifest}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
