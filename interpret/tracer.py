from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .activation_cache import ActivationCache, CaptureSpec
try:
    from blocks.transformer_block import TransformerBlock as _TransformerBlock
except Exception:
    _TransformerBlock = None  # type: ignore


HookFn = Callable[[str, nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor | None]


def default_output_only_hook(
    key: str,
    module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    return output


class ActivationTracer:
    """Utility to capture activations via forward hooks.

    Example:
        tracer = ActivationTracer(model)
        tracer.add_modules(["embed", "blocks.0.attn", "blocks.0.mlp"])  # by named_modules keys
        with tracer.trace() as cache:
            logits = model(input_ids)
        x = cache.get("blocks.0.attn")
    """

    def __init__(self, model: nn.Module, *, spec: Optional[CaptureSpec] = None) -> None:
        self.model = model
        self.spec = spec or CaptureSpec()
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._cache = ActivationCache()

    def _register_named_module(self, name: str, mod: nn.Module, *, key: Optional[str] = None, hook: Optional[HookFn] = None) -> None:
        storage_key = key or name
        hook_fn = hook or default_output_only_hook

        def _fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            value = hook_fn(storage_key, module, inputs, output)
            if value is not None:
                self._cache.store(storage_key, value, self.spec)

        self._handles.append(mod.register_forward_hook(_fn))

    def add_modules(self, names: Iterable[str], *, hook: Optional[HookFn] = None) -> None:
        name_to_module: Dict[str, nn.Module] = dict(self.model.named_modules())
        for name in names:
            mod = name_to_module.get(name)
            if mod is None:
                # Skip silently; caller can verify post-run what's captured
                continue
            self._register_named_module(name, mod, key=name, hook=hook)

    def add_modules_matching(self, predicate: Callable[[str, nn.Module], bool], *, hook: Optional[HookFn] = None) -> List[str]:
        captured: List[str] = []
        for name, mod in self.model.named_modules():
            try:
                if predicate(name, mod):
                    self._register_named_module(name, mod, key=name, hook=hook)
                    captured.append(name)
            except Exception:
                # Be robust to predicates that might raise
                continue
        return captured

    @contextmanager
    def trace(self) -> Iterable[ActivationCache]:
        try:
            yield self._cache
        finally:
            self.close()

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    # Convenience helpers for common transformer capture points
    def add_block_residual_streams(self, *, prefix: str = "blocks.") -> List[str]:
        """Capture outputs of attention and MLP submodules across all blocks.

        This hooks:
          - f"{prefix}{i}.attn"
          - f"{prefix}{i}.mlp"
        and stores under the same keys. Returns the list of hooked names.
        """
        def is_block_sub(name: str, _m: nn.Module) -> bool:
            return name.startswith(prefix) and (name.endswith(".attn") or name.endswith(".mlp"))

        return self.add_modules_matching(is_block_sub)

    def add_block_outputs(self, *, prefix: str = "blocks.") -> List[str]:
        """Capture outputs of each block module as the residual stream after the block.

        Hooks any module whose name matches f"{prefix}{i}" and appears to be a transformer
        block (heuristic: has both "attn" and "mlp" attributes).
        """
        def is_block(name: str, m: nn.Module) -> bool:
            if not name.startswith(prefix):
                return False
            # Must be a leaf block module like blocks.0, blocks.1
            if "." in name[len(prefix):]:
                return False
            if _TransformerBlock is not None and isinstance(m, _TransformerBlock):
                return True
            return hasattr(m, "attn") and hasattr(m, "mlp")

        return self.add_modules_matching(is_block)


