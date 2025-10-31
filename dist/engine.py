from contextlib import contextmanager
import os
from typing import Optional

import torch

from specs.dist import DistConfig
from . import utils
from .dataloader import build_distributed_dataloader


class DistributedEngine:
    def __init__(self, cfg: DistConfig):
        self.cfg = cfg
        self.rank = utils.get_rank(0)
        self.world = utils.get_world_size(1)

    def init(self, *, seed: Optional[int] = None, init_method: Optional[str] = None, timeout_s: Optional[float] = None) -> None:
        # Initialize distributed and set device
        from .launch import initialize_distributed

        initialize_distributed(
            backend=self.cfg.backend,
            init_method=init_method,
            seed=(seed if seed is not None else None),
            timeout_s=timeout_s,
        )
        # Refresh rank/world after init
        self.rank = utils.get_rank(0)
        self.world = utils.get_world_size(1)

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.cfg.strategy == "DDP":
            return torch.nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[torch.cuda.current_device()], find_unused_parameters=False
            )
        if self.cfg.strategy == "FSDP":
            from .strategy.fsdp import wrap_fsdp, make_transformer_auto_wrap_policy

            auto_wrap = (self.cfg.fsdp_wrap_policy == "transformer_block")
            return wrap_fsdp(
                model,
                auto_wrap=auto_wrap,
                state_offload=getattr(self.cfg, "fsdp_state_offload", False),
                cpu_offload=getattr(self.cfg, "cpu_offload", False),
                param_limit=1_000_000,
            )
        if self.cfg.strategy == "DeepSpeed":
            import deepspeed

            ds_cfg = {
                "train_batch_size": 1,
                "zero_optimization": {"stage": int(getattr(self.cfg, "zero_stage", 2))},
                "bf16": {"enabled": (self.cfg.precision == "bf16")},
                "fp16": {"enabled": (self.cfg.precision == "fp16")},
            }
            model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_cfg)
            return model
        raise ValueError("Unknown strategy")

    def wrap_loader(self, dataset, batch_size: int, num_workers: int = 4):
        return build_distributed_dataloader(
            dataset,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            drop_last=True,
            pin_memory=True,
        )

    def apply_tensor_parallel(self, model: torch.nn.Module, tp_size: int) -> torch.nn.Module:
        if int(tp_size) <= 1:
            return model
        from .parallel.tensor_parallel import apply_tensor_parallel as _apply_tp

        return _apply_tp(model, int(tp_size))

    def build_pipeline(self, model: torch.nn.Module, num_stages: int):
        from .parallel.pipeline import partition_model_into_stages

        return partition_model_into_stages(model, int(num_stages))

    @contextmanager
    def autocast(self):
        if self.cfg.precision == "bf16":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                yield
        elif self.cfg.precision == "fp16":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                yield
        else:
            yield

    def sync(self) -> None:
        utils.barrier()

    def is_primary(self) -> bool:
        return utils.is_primary()

    def finalize(self) -> None:
        utils.destroy_process_group()
