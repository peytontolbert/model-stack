from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist


def get_rank(default: int = 0) -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return int(os.getenv("RANK", str(default)))


def get_world_size(default: int = 1) -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return int(os.getenv("WORLD_SIZE", str(default)))


def get_local_rank(default: int = 0) -> int:
    # Prefer LOCAL_RANK if provided by torchrun
    if "LOCAL_RANK" in os.environ:
        return int(os.getenv("LOCAL_RANK", str(default)))
    # Fallback: derive from global rank
    rank = get_rank(default=default)
    num_devices = max(torch.cuda.device_count(), 1)
    return int(rank % num_devices)


def is_primary() -> bool:
    return get_rank(0) == 0


def barrier() -> None:
    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device() -> int:
    device_index = get_local_rank(0)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
    return device_index


def init_process_group(backend: str = "nccl", init_method: Optional[str] = None, timeout: Optional[float] = None) -> None:
    if dist.is_available() and dist.is_initialized():
        return
    kwargs: dict[str, Any] = {"backend": backend}
    if init_method:
        kwargs["init_method"] = init_method
    if timeout is not None:
        import datetime as _dt
        kwargs["timeout"] = _dt.timedelta(seconds=float(timeout))
    dist.init_process_group(**kwargs)


def destroy_process_group() -> None:
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def broadcast_object(obj: Any, src: int = 0) -> Any:
    try:
        if dist.is_available() and dist.is_initialized():
            obj_list = [obj] if get_rank() == src else [None]
            dist.broadcast_object_list(obj_list, src=src)
            return obj_list[0]
    except Exception:
        pass
    return obj


def broadcast_tensor(x: torch.Tensor, src: int = 0) -> torch.Tensor:
    try:
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(x, src=src)
            return x
    except Exception:
        pass
    return x


