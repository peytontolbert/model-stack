import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


@dataclass
class CaptureSpec:
    """Configuration for how to capture activations.

    - move_to_cpu: if True, activations are moved to CPU memory
    - dtype: if provided, activations are cast to this dtype before storage
    - detach: if True, detach from autograd graph before storage
    - clone: if True, clone tensor to avoid views referencing transient buffers
    - keep_grad: if True, do not detach (overrides detach)
    """

    move_to_cpu: bool = True
    dtype: Optional[torch.dtype] = None
    detach: bool = True
    clone: bool = True
    keep_grad: bool = False


class ActivationCache:
    """In-memory activation cache with simple namespacing.

    Keys are arbitrary strings, typically module names or "layer.N.field" paths.
    Values are torch.Tensor objects stored according to the provided CaptureSpec.

    Usage:
        cache = ActivationCache()
        cache.store("blocks.0.mlp.out", tensor, spec)
        x = cache.get("blocks.0.mlp.out")
    """

    def __init__(self) -> None:
        self._store: Dict[str, torch.Tensor] = {}

    def __contains__(self, key: str) -> bool:  # pragma: no cover - trivial
        return key in self._store

    def keys(self) -> Iterable[str]:  # pragma: no cover - trivial
        return self._store.keys()

    def items(self) -> Iterable[Tuple[str, torch.Tensor]]:  # pragma: no cover - trivial
        return self._store.items()

    def get(self, key: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        return self._store.get(key, default)

    def pop(self, key: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        return self._store.pop(key, default)

    def clear(self) -> None:  # pragma: no cover - trivial
        self._store.clear()

    def store(self, key: str, value: torch.Tensor, spec: Optional[CaptureSpec] = None) -> None:
        spec = spec or CaptureSpec()
        tensor = value
        if spec.clone:
            tensor = tensor.clone()
        if spec.keep_grad:
            pass
        elif spec.detach:
            tensor = tensor.detach()
        if spec.dtype is not None and tensor.dtype != spec.dtype:
            tensor = tensor.to(dtype=spec.dtype)
        if spec.move_to_cpu and tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        self._store[key] = tensor

    def save_pt(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self._store, path)

    @staticmethod
    def load_pt(path: str) -> "ActivationCache":
        cache = ActivationCache()
        cache._store = torch.load(path, map_location="cpu")
        return cache


