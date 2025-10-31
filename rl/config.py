from dataclasses import dataclass
from typing import Literal


@dataclass
class RLConfig:
    algo: Literal["ppo", "dpo"] = "ppo"
    steps: int = 1000
    lr: float = 1e-5
    gamma: float = 0.99
    clip_ratio: float = 0.2  # PPO
    kl_target: float = 0.01  # PPO
    beta: float = 0.1        # DPO temperature


