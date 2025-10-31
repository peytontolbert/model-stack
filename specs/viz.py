# specs/viz.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class VizConfig:
    backend: Literal["viz","tensorboard","wandb"] = "viz"
    log_dir: str = "runs/"
    profile_every_n_steps: int = 200
    activation_probes: bool = True
