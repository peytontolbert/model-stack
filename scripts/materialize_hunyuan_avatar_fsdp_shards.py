"""Convert Avatar's monolithic transformer checkpoint into rank-local FSDP shards.

This is a one-time materialization utility. It intentionally hydrates rank zero
and rank one sequentially on CPU to stay within the host memory budget, then
uses FSDP's sharded state-dict API to write the reusable two-rank checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def parse_materializer_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_known_args()


def main() -> None:
    materializer_args, avatar_argv = parse_materializer_args()
    sys.argv = [sys.argv[0], *avatar_argv]

    avatar_root = os.environ["HUNYUAN_AVATAR_ROOT"]
    if avatar_root not in sys.path:
        sys.path.insert(0, avatar_root)

    from hymm_sp.config import parse_args
    from runtime.hunyuan_avatar_fsdp import (
        AvatarFSDPConfig,
        build_avatar_transformer_cpu,
        load_monolithic_checkpoint_cpu,
        normalize_floating_point_dtype,
        save_rank_local_shard,
        wrap_avatar_transformer_fsdp,
        write_shard_manifest,
    )

    args = parse_args()
    if args.use_fp8:
        raise ValueError("Materialize BF16 FSDP shards first; FP8 requires a dtype-aware wrapping policy.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # The source checkpoint is 29 GB and can be cold on network storage. Give
    # the staggered CPU hydration enough time before creating the NCCL barrier.
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.cuda.set_device(rank)

    print(f"rank={rank} constructing Avatar transformer on CPU", flush=True)
    model = build_avatar_transformer_cpu(args, avatar_root)
    normalize_floating_point_dtype(model, torch.bfloat16)

    # Never hold two checkpoint payloads in memory: rank zero hydrates first,
    # then rank one. Both full CPU models are present only once loading ends.
    if rank == 0:
        print("rank=0 hydrating monolithic checkpoint on CPU", flush=True)
        load_monolithic_checkpoint_cpu(model, args.ckpt)
        print("rank=0 checkpoint hydration complete", flush=True)
    dist.barrier()
    if rank != 0:
        print("rank=1 hydrating monolithic checkpoint on CPU", flush=True)
        load_monolithic_checkpoint_cpu(model, args.ckpt)
        print("rank=1 checkpoint hydration complete", flush=True)
    dist.barrier()

    print(f"rank={rank} entering CPU-first FSDP wrap", flush=True)
    wrapped = wrap_avatar_transformer_fsdp(
        model,
        avatar_root=avatar_root,
        config=AvatarFSDPConfig(device_id=rank),
    )
    print(f"rank={rank} exporting rank-local shard", flush=True)
    shard = save_rank_local_shard(wrapped, materializer_args.output_dir, rank=rank)
    print(f"rank={rank} wrote {shard}")
    dist.barrier()
    if rank == 0:
        manifest = write_shard_manifest(
            materializer_args.output_dir,
            source_checkpoint=args.ckpt,
            world_size=world_size,
        )
        print(f"wrote {manifest}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
