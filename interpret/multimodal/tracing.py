from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from interpret.activation_cache import ActivationCache, CaptureSpec

from .adapter import MultimodalInputs, MultimodalModelAdapter, get_multimodal_adapter


class MultimodalTracer:
    def __init__(self, model: nn.Module, *, spec: Optional[CaptureSpec] = None) -> None:
        self.adapter: MultimodalModelAdapter = get_multimodal_adapter(model)
        self.spec = spec or CaptureSpec()
        self.cache = ActivationCache()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def add_components(self, roles: Iterable[str] = ("vision", "projector", "language")) -> list[str]:
        wanted = set(roles)
        keys: list[str] = []
        for component in self.adapter.components():
            if component.role not in wanted:
                continue
            self._register(component.name, component.module)
            keys.append(component.name)
        return keys

    def add_modules(self, names: Iterable[str]) -> list[str]:
        modules = self.adapter.named_modules()
        keys: list[str] = []
        for name in names:
            module = modules.get(name)
            if module is None:
                continue
            self._register(name, module)
            keys.append(name)
        return keys

    def _register(self, key: str, module: nn.Module) -> None:
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor) else output
            if isinstance(value, torch.Tensor):
                self.cache.store(key, value, self.spec)
            elif hasattr(value, "last_hidden_state") and isinstance(value.last_hidden_state, torch.Tensor):
                self.cache.store(f"{key}.last_hidden_state", value.last_hidden_state, self.spec)
            elif hasattr(value, "logits") and isinstance(value.logits, torch.Tensor):
                self.cache.store(f"{key}.logits", value.logits, self.spec)

        self._handles.append(module.register_forward_hook(_hook))

    @contextmanager
    def trace(self):
        try:
            yield self.cache
        finally:
            self.close()

    def trace_forward(self, inputs: MultimodalInputs | dict[str, Any] | None = None, **kwargs: Any) -> tuple[Any, ActivationCache]:
        self.add_components()
        with self.trace() as cache:
            output = self.adapter.forward(inputs, **kwargs)
        return output, cache

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()


def trace_multimodal_forward(model: nn.Module, inputs: MultimodalInputs | dict[str, Any] | None = None, **kwargs: Any) -> tuple[Any, ActivationCache]:
    tracer = MultimodalTracer(model)
    return tracer.trace_forward(inputs, **kwargs)
