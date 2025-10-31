from __future__ import annotations

from typing import Iterable, List

import torch


class PipelineStage(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.module(x, *args, **kwargs)


def partition_model_into_stages(model: torch.nn.Module, num_stages: int) -> List[PipelineStage]:
    if not hasattr(model, "blocks"):
        raise ValueError("Model has no attribute 'blocks' to partition")
    blocks = list(model.blocks)
    n = len(blocks)
    if num_stages <= 1 or n == 0:
        return [PipelineStage(model)]
    per = max(1, n // int(num_stages))
    stages: List[PipelineStage] = []
    cur = 0
    for i in range(num_stages):
        end = n if i == num_stages - 1 else min(cur + per, n)
        if cur >= end:
            stages.append(PipelineStage(torch.nn.Identity()))
        else:
            seq = torch.nn.Sequential(*blocks[cur:end])
            stages.append(PipelineStage(seq))
        cur = end
    return stages


def run_pipeline(stages: List[PipelineStage], microbatches: Iterable[torch.Tensor]) -> torch.Tensor:
    # Simple sequential pipeline execution (placeholder for 1F1B scheduling)
    outs: List[torch.Tensor] = []
    for mb in microbatches:
        x = mb
        for stage in stages:
            x = stage(x)
        outs.append(x)
    return torch.cat(outs, dim=0)


