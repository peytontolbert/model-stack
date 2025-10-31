# data/batch.py
from dataclasses import dataclass
import torch
@dataclass
class Batch:
    input_ids: torch.Tensor   # (B,T)
    attn_mask: torch.Tensor   # (B,T,T) or causal shorthand
