# specs/dist.py
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class DistConfig:
    backend: Literal["nccl","gloo"] = "nccl"
    strategy: Literal["DDP","FSDP","DeepSpeed"] = "FSDP"
    precision: Literal["fp32","fp16","bf16"] = "bf16"
    grad_ckpt: bool = True
    fsdp_wrap_policy: Literal["transformer_block","thin"] = "transformer_block"
    fsdp_state_offload: bool = False
    zero_stage: int = 2               # for DeepSpeed
    cpu_offload: bool = False
    activation_offload: bool = False
    clip_grad_norm: float = 1.0
