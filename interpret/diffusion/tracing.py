from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn

from interpret.activation_cache import ActivationCache, CaptureSpec

from .adapter import DiffusionModelAdapter, get_diffusion_adapter


@dataclass
class DiffusionStepRecord:
    index: int
    timestep: int | float | None
    latent_shape: tuple[int, ...] | None


def _store(cache: ActivationCache, key: str, value: Any, spec: CaptureSpec) -> None:
    if isinstance(value, torch.Tensor):
        cache.store(key, value, spec)


def _maybe_timestep(value: Any) -> int | float | None:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.detach().flatten()[0].item()
    if isinstance(value, (int, float)):
        return value
    return None


def _extract_attention_probs(output: Any) -> torch.Tensor | None:
    if isinstance(output, dict):
        for key in ("attn_probs", "attention_probs", "attentions", "attention"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    if hasattr(output, "attn_probs") and isinstance(output.attn_probs, torch.Tensor):
        return output.attn_probs
    if isinstance(output, tuple):
        for value in output:
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                return value
        for value in output:
            if isinstance(value, torch.Tensor) and value.ndim >= 3:
                return value
    if isinstance(output, torch.Tensor) and output.ndim == 4:
        last = int(output.shape[-1])
        if last > 1 and torch.all(output >= 0):
            sums = output.float().sum(dim=-1)
            if torch.allclose(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3):
                return output
    return None


class DiffusionTracer:
    """Capture text-to-image denoising, latent, and cross-attention surfaces."""

    def __init__(
        self,
        pipeline: Any,
        *,
        spec: Optional[CaptureSpec] = None,
        attention_predicate: Optional[Callable[[str, nn.Module], bool]] = None,
    ) -> None:
        self.adapter: DiffusionModelAdapter = get_diffusion_adapter(pipeline)
        self.spec = spec or CaptureSpec()
        self.attention_predicate = attention_predicate
        self.cache = ActivationCache()
        self.step_records: list[DiffusionStepRecord] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._step_index = -1

    def add_denoiser_latents(self, *, key_prefix: str = "denoiser") -> None:
        denoiser = self.adapter.denoiser_module()
        if denoiser is None:
            return

        def _pre_hook(_module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            self._step_index += 1
            latent = inputs[0] if inputs and isinstance(inputs[0], torch.Tensor) else kwargs.get("sample")
            if latent is None:
                latent = kwargs.get("hidden_states")
            timestep = kwargs.get("timestep")
            if timestep is None and len(inputs) > 1:
                timestep = inputs[1]
            latent_shape = tuple(latent.shape) if isinstance(latent, torch.Tensor) else None
            self.step_records.append(
                DiffusionStepRecord(index=self._step_index, timestep=_maybe_timestep(timestep), latent_shape=latent_shape)
            )
            _store(self.cache, f"{key_prefix}.step_{self._step_index}.latent_in", latent, self.spec)

        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            sample = output.get("sample") if isinstance(output, dict) else getattr(output, "sample", output)
            _store(self.cache, f"{key_prefix}.step_{self._step_index}.latent_out", sample, self.spec)

        self._handles.append(denoiser.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        self._handles.append(denoiser.register_forward_hook(_hook))

    def add_cross_attention(self, *, include_outputs: bool = True, include_probs: bool = True) -> list[str]:
        keys: list[str] = []
        for target in self.adapter.cross_attention_targets(self.attention_predicate):
            out_key = f"{target.name}.step_{{step}}.out"
            probs_key = f"{target.name}.step_{{step}}.attn_probs"
            if include_outputs:
                keys.append(out_key)
            if include_probs:
                keys.append(probs_key)

            def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, _target=target) -> None:
                step = max(self._step_index, 0)
                sample = output[0] if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor) else output
                if include_outputs:
                    _store(self.cache, f"{_target.name}.step_{step}.out", sample, self.spec)
                if include_probs:
                    probs = _extract_attention_probs(output)
                    _store(self.cache, f"{_target.name}.step_{step}.attn_probs", probs, self.spec)

            self._handles.append(target.module.register_forward_hook(_hook))
        return keys

    def add_modules(self, names: Iterable[str]) -> list[str]:
        modules = self.adapter.named_modules()
        added: list[str] = []
        for name in names:
            module = modules.get(name)
            if module is None:
                continue

            def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, _name=name) -> None:
                sample = output[0] if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor) else output
                _store(self.cache, f"{_name}.step_{max(self._step_index, 0)}.out", sample, self.spec)

            self._handles.append(module.register_forward_hook(_hook))
            added.append(name)
        return added

    @contextmanager
    def trace(self):
        try:
            yield self.cache
        finally:
            self.close()

    def trace_generation(self, prompt: str | list[str], **kwargs: Any) -> tuple[Any, ActivationCache, list[DiffusionStepRecord]]:
        self.add_denoiser_latents()
        self.add_cross_attention()
        with self.trace() as cache:
            output = self.adapter.generate(prompt, **kwargs)
        return output, cache, list(self.step_records)

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()


def trace_diffusion_generation(pipeline: Any, prompt: str | list[str], **kwargs: Any) -> tuple[Any, ActivationCache, list[DiffusionStepRecord]]:
    tracer = DiffusionTracer(pipeline)
    return tracer.trace_generation(prompt, **kwargs)
