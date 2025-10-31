import torch.nn as nn

from specs.config import ModelConfig
from .eager import EagerAttention


class TritonAttention(EagerAttention):
    def __init__(self, cfg: ModelConfig, **overrides):
        super().__init__(cfg, backend_override="triton", **overrides)


