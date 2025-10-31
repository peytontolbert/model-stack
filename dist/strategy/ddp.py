from __future__ import annotations

import torch


def wrap_ddp(model: torch.nn.Module, *, find_unused_parameters: bool = False) -> torch.nn.Module:
    return torch.nn.parallel.DistributedDataParallel(
        model.cuda(), device_ids=[torch.cuda.current_device()], find_unused_parameters=bool(find_unused_parameters)
    )


class NoSync:
    def __init__(self, module: torch.nn.parallel.DistributedDataParallel):
        self.module = module

    def __enter__(self):
        return self.module.no_sync()

    def __exit__(self, exc_type, exc, tb):
        # Context handled by DDP; nothing to do here
        return False


