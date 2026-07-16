"""Two-rank GPU smoke test for transformer_10's Avatar FSDP adapter."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch import nn

from runtime.hunyuan_avatar_fsdp import AvatarFSDPConfig, wrap_avatar_transformer_fsdp


class SingleStreamBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, width * 2, bias=False)
        self.out_proj = nn.Linear(width * 2, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.nn.functional.silu(self.in_proj(value)))


class AvatarShapeModel(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(SingleStreamBlock(width) for _ in range(layers))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--avatar-root", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    # The adapter policy resolves the real Avatar block classes. This synthetic
    # model validates FSDP's CPU-to-sharded-GPU placement independently.
    model = AvatarShapeModel(args.width, args.layers).to("cpu")
    wrapped = wrap_avatar_transformer_fsdp(
        model,
        avatar_root=args.avatar_root,
        config=AvatarFSDPConfig(device_id=rank),
    )
    value = torch.randn(1, 4, args.width, device=f"cuda:{rank}")
    result = wrapped(value)
    torch.cuda.synchronize(rank)
    if rank == 0:
        print(f"FSDP smoke passed: result={tuple(result.shape)} allocated={torch.cuda.memory_allocated(rank)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
