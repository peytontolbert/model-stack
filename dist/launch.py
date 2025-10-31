from __future__ import annotations

import os
from typing import Optional

from . import utils


def infer_from_slurm() -> None:
    # Map SLURM environment to torchrun-like vars if present
    if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:
        os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
    if "SLURM_NTASKS" in os.environ and "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
    if "SLURM_LOCALID" in os.environ and "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")


def initialize_distributed(backend: str = "nccl", *, init_method: Optional[str] = None, seed: Optional[int] = None, timeout_s: Optional[float] = None) -> None:
    # Derive env if running under SLURM
    infer_from_slurm()
    # Set device first to avoid CUDA context on wrong GPU
    utils.setup_device()
    # Initialize process group
    utils.init_process_group(backend=backend, init_method=init_method, timeout=timeout_s)
    # Optional seeding
    if seed is not None:
        utils.seed_everything(int(seed))


def recommended_torchrun_cmd(nnodes: int = 1, nproc_per_node: int = 1, master_addr: str = "127.0.0.1", master_port: int = 29500) -> str:
    return (
        f"torchrun --nnodes {nnodes} --nproc_per_node {nproc_per_node} "
        f"--master_addr {master_addr} --master_port {master_port} your_script.py"
    )


