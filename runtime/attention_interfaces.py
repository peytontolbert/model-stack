from __future__ import annotations

from typing import Optional, Protocol

import torch


class KVCache(Protocol):
    def append(self, k: torch.Tensor, v: torch.Tensor): ...

    def read(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    def length(self) -> int: ...


class Attention(Protocol):
    def forward(self, q, k, v, mask, cache: Optional[KVCache] = None) -> torch.Tensor: ...
