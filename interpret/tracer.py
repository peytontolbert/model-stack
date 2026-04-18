from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .activation_cache import ActivationCache, CaptureSpec
from .model_adapter import AttentionSnapshot, MLPSnapshot, ModelInputs, coerce_model_inputs, get_model_adapter, patched_attention, patched_embedding_output, patched_mlp


def _load_transformer_block():
    try:
        from runtime.block_modules import TransformerBlock as transformer_block
    except Exception:
        return None
    return transformer_block


_TransformerBlock = _load_transformer_block()  # type: ignore[assignment]


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
        self._trace_context_factories: List[Callable[[], object]] = []
        self._cache = ActivationCache()

    def _register_named_module(self, name: str, mod: nn.Module, *, key: Optional[str] = None, hook: Optional[HookFn] = None) -> None:
        storage_key = key or name
        hook_fn = hook or default_output_only_hook

        def _fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            value = hook_fn(storage_key, module, inputs, output)
            if value is not None:
                self._cache.store(storage_key, value, self.spec)

        self._handles.append(mod.register_forward_hook(_fn))

    def _register_named_module_pre(self, name: str, mod: nn.Module, *, key: Optional[str] = None) -> None:
        storage_key = key or name

        def _fn(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            value = inputs[0]
            if isinstance(value, torch.Tensor):
                self._cache.store(storage_key, value, self.spec)

        self._handles.append(mod.register_forward_pre_hook(_fn))

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
        with ExitStack() as stack:
            for factory in self._trace_context_factories:
                stack.enter_context(factory())
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

    def add_embedding_output(self, *, stack: Optional[str] = None) -> str:
        adapter = get_model_adapter(self.model)
        key = "embed"
        if adapter.kind == "encoder_decoder":
            resolved = stack or "decoder"
            key = f"{resolved}.embed"

        def _factory(*, _adapter=adapter, _stack=stack, _key=key):
            def _capture(output: torch.Tensor) -> None:
                self._cache.store(_key, output, self.spec)

            return patched_embedding_output(_adapter, stack=_stack, capture=_capture, keep_grad=bool(self.spec.keep_grad))

        self._trace_context_factories.append(_factory)
        return key

    def add_residual_streams(self, *, stack: Optional[str] = None) -> List[str]:
        adapter = get_model_adapter(self.model)
        keys: List[str] = []
        for target in adapter.block_targets(stack=stack):
            pre_key = f"{target.name}.resid_pre"
            post_key = f"{target.name}.resid_post"
            self._register_named_module_pre(target.name, target.module, key=pre_key)
            self._register_named_module(target.name, target.module, key=post_key)
            keys.extend([pre_key, post_key])
        return keys

    def add_attention_surfaces(
        self,
        *,
        stack: Optional[str] = None,
        kind: str = "self",
        include: Optional[Iterable[str]] = None,
    ) -> List[str]:
        adapter = get_model_adapter(self.model)
        fields = set(include or ("q", "k", "v", "attn_logits", "attn_probs", "head_out", "attn_out"))
        keys: List[str] = []
        kinds = ["self", "cross"] if kind == "both" else [kind]
        for target_kind in kinds:
            for target in adapter.attention_targets(stack=stack, kind=target_kind):  # type: ignore[arg-type]
                for suffix in fields:
                    keys.append(f"{target.name}.{suffix}")

                def _factory(*, _target=target, _fields=tuple(fields)):
                    def _capture(snapshot: AttentionSnapshot) -> None:
                        mapping = {
                            "q": snapshot.q,
                            "k": snapshot.k,
                            "v": snapshot.v,
                            "attn_logits": snapshot.logits,
                            "attn_probs": snapshot.probs,
                            "head_out": snapshot.head_out,
                            "attn_out": snapshot.output,
                        }
                        for suffix, value in mapping.items():
                            if suffix in _fields and value is not None:
                                self._cache.store(f"{_target.name}.{suffix}", value, self.spec)

                    return patched_attention(
                        _target.module,
                        capture=_capture,
                        capture_logits=("attn_logits" in _fields or "attn_probs" in _fields),
                        capture_probs=("attn_logits" in _fields or "attn_probs" in _fields),
                        keep_grad=bool(self.spec.keep_grad),
                    )

                self._trace_context_factories.append(_factory)
        return keys

    def add_mlp_surfaces(
        self,
        *,
        stack: Optional[str] = None,
        kind: Optional[str] = None,
        include: Optional[Iterable[str]] = None,
    ) -> List[str]:
        adapter = get_model_adapter(self.model)
        fields = set(include or ("mlp_in", "mlp_mid", "mlp_out"))
        keys: List[str] = []
        for target in adapter.mlp_targets(stack=stack, kind=kind):
            for suffix in fields:
                keys.append(f"{target.name}.{suffix}")

            def _factory(*, _target=target, _fields=tuple(fields)):
                def _capture(snapshot: MLPSnapshot) -> None:
                    mapping = {
                        "mlp_in": snapshot.mlp_in,
                        "mlp_mid": snapshot.mlp_mid,
                        "mlp_out": snapshot.mlp_out,
                    }
                    for suffix, value in mapping.items():
                        if suffix in _fields and value is not None:
                            self._cache.store(f"{_target.name}.{suffix}", value, self.spec)

                return patched_mlp(_target.module, capture=_capture, keep_grad=bool(self.spec.keep_grad))

            self._trace_context_factories.append(_factory)
        return keys

    def add_interpret_surfaces(
        self,
        *,
        stack: Optional[str] = None,
        attention_kind: str = "self",
        include_embed: bool = True,
        include_residual: bool = True,
        include_attention: bool = True,
        include_mlp: bool = True,
    ) -> List[str]:
        adapter = get_model_adapter(self.model)
        resolved_stack = stack or ("causal" if adapter.kind == "causal" else "encoder" if adapter.kind == "encoder" else "decoder")
        keys: List[str] = []
        if include_embed:
            keys.append(self.add_embedding_output(stack=stack))
        if include_residual:
            keys.extend(self.add_residual_streams(stack=stack))
        if include_attention:
            keys.extend(self.add_attention_surfaces(stack=stack, kind=attention_kind))
        if include_mlp:
            keys.extend(self.add_mlp_surfaces(stack=stack, kind=(attention_kind if resolved_stack == "decoder" else None)))
        return keys
