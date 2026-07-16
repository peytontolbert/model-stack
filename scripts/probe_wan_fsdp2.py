"""Probe Wan Animate FSDP2 memory without loading text, VAE, or video inputs."""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def main() -> None:
    wan_root = Path("/home/peyton/src/Wan2.2")
    sys.path.insert(0, str(wan_root))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

    from wan.distributed.fsdp import shard_model_fsdp2
    from wan.modules.animate import WanAnimateModel

    checkpoint_dir = "/data/models/Wan-AI--Wan2.2-Animate-14B"
    model = WanAnimateModel.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    model = shard_model_fsdp2(model, mesh)
    torch.cuda.synchronize(rank)
    print(
        f"rank={rank} allocated={torch.cuda.memory_allocated(rank) / 2**30:.2f}GiB "
        f"reserved={torch.cuda.memory_reserved(rank) / 2**30:.2f}GiB",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
