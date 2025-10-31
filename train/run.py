# train/run.py
from __future__ import annotations

from typing import Optional

import torch

from dist.engine import DistributedEngine
from specs.dist import DistConfig
from train.trainer import Trainer, TrainConfig

try:
    from viz.session import VizSession  # optional
except Exception:
    VizSession = None  # type: ignore


def run_training(
    *,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    dist_cfg: DistConfig,
    train_cfg: Optional[TrainConfig] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    viz_cfg: Optional[object] = None,
    max_steps: Optional[int] = None,
    max_epochs: Optional[int] = 1,
    save_dir: Optional[str] = None,
    save_every_steps: int = 0,
    # Optional validation
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    validate_every_steps: int = 0,
) -> None:
    """Initialize distributed, wrap model/loader, and train with Trainer.

    Provide either max_steps or max_epochs (default 1 epoch).
    """
    engine = DistributedEngine(dist_cfg)
    engine.init()

    # Wrap model (DDP/FSDP/DeepSpeed)
    model = engine.wrap_model(model)

    # Build loader
    train_loader = engine.wrap_loader(dataset, batch_size=batch_size, num_workers=int(num_workers))
    val_loader = None
    if val_dataset is not None:
        val_loader = engine.wrap_loader(val_dataset, batch_size=batch_size, num_workers=int(num_workers))

    # Optional viz session
    viz = VizSession(viz_cfg) if (viz_cfg is not None and VizSession is not None) else None

    # Trainer
    tc = (train_cfg or TrainConfig())
    if save_dir is not None:
        tc.save_dir = save_dir
        tc.save_every_steps = int(save_every_steps)
    if validate_every_steps:
        tc.validate_every_steps = int(validate_every_steps)
    trainer = Trainer(model, dist_cfg=dist_cfg, train_cfg=tc, viz_session=viz)

    # Loop
    if max_steps is not None:
        trainer.train_steps(train_loader, int(max_steps), val_loader=val_loader)
    else:
        trainer.train_epochs(train_loader, int(max_epochs or 1), val_loader=val_loader)

