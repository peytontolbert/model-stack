from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

from interpret.activation_cache import CaptureSpec

from .adapter import DiffusionInputs, get_diffusion_adapter
from .metrics import mse_distance
from .tracing import DiffusionTracer


@dataclass(frozen=True)
class DiffusionPatchResult:
    clean_output: Any
    corrupted_output: Any
    patched_output: Any
    clean_corrupted_distance: float
    clean_patched_distance: float
    recovery_fraction: float
    patched_keys: tuple[str, ...]


def _as_step_set(steps: Iterable[int] | None) -> set[int] | None:
    if steps is None:
        return None
    return {int(step) for step in steps}


def _recovery(clean_corrupted: torch.Tensor, clean_patched: torch.Tensor) -> float:
    denom = float(clean_corrupted.detach().cpu().item())
    if abs(denom) < 1e-12:
        return 0.0
    return float((clean_corrupted - clean_patched).detach().cpu().item() / denom)


def _replace_first_tensor(inputs: tuple[Any, ...], replacement: torch.Tensor) -> tuple[Any, ...]:
    if not inputs:
        return inputs
    values = list(inputs)
    for idx, value in enumerate(values):
        if isinstance(value, torch.Tensor):
            values[idx] = replacement.to(device=value.device, dtype=value.dtype)
            break
    return tuple(values)


def patch_denoiser_latents(
    pipeline: Any,
    *,
    clean_prompt: str | list[str],
    corrupted_prompt: str | list[str],
    patch_steps: Iterable[int] | None = None,
    num_inference_steps: int = 20,
    distance_fn: Callable[[Any, Any], torch.Tensor] = mse_distance,
    generation_kwargs: dict[str, Any] | None = None,
) -> DiffusionPatchResult:
    """Patch clean denoiser input latents into a corrupted generation run."""

    adapter = get_diffusion_adapter(pipeline)
    kwargs = dict(generation_kwargs or {})
    kwargs["num_inference_steps"] = int(num_inference_steps)
    clean_output, clean_cache, _records = DiffusionTracer(pipeline, spec=CaptureSpec(move_to_cpu=False)).trace_generation(clean_prompt, **kwargs)
    corrupted_output = adapter.generate(DiffusionInputs(prompt=corrupted_prompt, extra_kwargs=kwargs))
    denoiser = adapter.denoiser_module()
    if denoiser is None:
        raise TypeError("Pipeline has no denoiser module to patch")

    allowed_steps = _as_step_set(patch_steps)
    step_index = -1
    patched_keys: list[str] = []

    def _pre_hook(_module: nn.Module, inputs: tuple[Any, ...], hook_kwargs: dict[str, Any]):
        nonlocal step_index
        step_index += 1
        if allowed_steps is not None and step_index not in allowed_steps:
            return inputs, hook_kwargs
        key = f"denoiser.step_{step_index}.latent_in"
        clean_latent = clean_cache.get(key)
        if clean_latent is None:
            return inputs, hook_kwargs
        patched_keys.append(key)
        if "sample" in hook_kwargs and isinstance(hook_kwargs["sample"], torch.Tensor):
            ref = hook_kwargs["sample"]
            hook_kwargs = dict(hook_kwargs)
            hook_kwargs["sample"] = clean_latent.to(device=ref.device, dtype=ref.dtype)
            return inputs, hook_kwargs
        return _replace_first_tensor(inputs, clean_latent), hook_kwargs

    handle = denoiser.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        patched_output = adapter.generate(DiffusionInputs(prompt=corrupted_prompt, extra_kwargs=kwargs))
    finally:
        handle.remove()

    clean_corrupted = distance_fn(clean_output, corrupted_output)
    clean_patched = distance_fn(clean_output, patched_output)
    return DiffusionPatchResult(
        clean_output=clean_output,
        corrupted_output=corrupted_output,
        patched_output=patched_output,
        clean_corrupted_distance=float(clean_corrupted.detach().cpu().item()),
        clean_patched_distance=float(clean_patched.detach().cpu().item()),
        recovery_fraction=_recovery(clean_corrupted, clean_patched),
        patched_keys=tuple(patched_keys),
    )


def patch_diffusion_module_outputs(
    pipeline: Any,
    *,
    clean_prompt: str | list[str],
    corrupted_prompt: str | list[str],
    module_names: Iterable[str],
    patch_steps: Iterable[int] | None = None,
    num_inference_steps: int = 20,
    distance_fn: Callable[[Any, Any], torch.Tensor] = mse_distance,
    generation_kwargs: dict[str, Any] | None = None,
) -> DiffusionPatchResult:
    """Patch selected module outputs from a clean generation into a corrupted run."""

    adapter = get_diffusion_adapter(pipeline)
    names = [name for name in module_names if name]
    kwargs = dict(generation_kwargs or {})
    kwargs["num_inference_steps"] = int(num_inference_steps)

    clean_tracer = DiffusionTracer(pipeline, spec=CaptureSpec(move_to_cpu=False))
    clean_tracer.add_denoiser_latents()
    clean_tracer.add_modules(names)
    with clean_tracer.trace() as clean_cache:
        clean_output = adapter.generate(DiffusionInputs(prompt=clean_prompt, extra_kwargs=kwargs))
    corrupted_output = adapter.generate(DiffusionInputs(prompt=corrupted_prompt, extra_kwargs=kwargs))

    modules = adapter.named_modules()
    missing = [name for name in names if name not in modules]
    if missing:
        raise KeyError(f"Unknown diffusion module(s): {missing}")

    allowed_steps = _as_step_set(patch_steps)
    denoiser_step = -1
    module_step: dict[str, int] = {name: -1 for name in names}
    patched_keys: list[str] = []

    denoiser = adapter.denoiser_module()

    def _step_hook(_module: nn.Module, _inputs: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        nonlocal denoiser_step
        denoiser_step += 1

    handles: list[torch.utils.hooks.RemovableHandle] = []
    if denoiser is not None:
        handles.append(denoiser.register_forward_pre_hook(_step_hook, with_kwargs=True))

    def _make_hook(name: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            step = denoiser_step
            module_step[name] = step
            if allowed_steps is not None and step not in allowed_steps:
                return output
            key = f"{name}.step_{step}.out"
            clean_value = clean_cache.get(key)
            if clean_value is None:
                return output
            patched_keys.append(key)
            if isinstance(output, tuple):
                values = list(output)
                if values and isinstance(values[0], torch.Tensor):
                    values[0] = clean_value.to(device=values[0].device, dtype=values[0].dtype)
                    return tuple(values)
            if isinstance(output, torch.Tensor):
                return clean_value.to(device=output.device, dtype=output.dtype)
            if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
                output.sample = clean_value.to(device=output.sample.device, dtype=output.sample.dtype)
                return output
            return output

        return _hook

    with ExitStack() as stack:
        for name in names:
            handles.append(modules[name].register_forward_hook(_make_hook(name)))
        for handle in handles:
            stack.callback(handle.remove)
        patched_output = adapter.generate(DiffusionInputs(prompt=corrupted_prompt, extra_kwargs=kwargs))

    clean_corrupted = distance_fn(clean_output, corrupted_output)
    clean_patched = distance_fn(clean_output, patched_output)
    return DiffusionPatchResult(
        clean_output=clean_output,
        corrupted_output=corrupted_output,
        patched_output=patched_output,
        clean_corrupted_distance=float(clean_corrupted.detach().cpu().item()),
        clean_patched_distance=float(clean_patched.detach().cpu().item()),
        recovery_fraction=_recovery(clean_corrupted, clean_patched),
        patched_keys=tuple(patched_keys),
    )


def diffusion_module_patch_sweep(
    pipeline: Any,
    *,
    clean_prompt: str | list[str],
    corrupted_prompt: str | list[str],
    module_names: Iterable[str],
    patch_steps: Iterable[int],
    num_inference_steps: int = 20,
    distance_fn: Callable[[Any, Any], torch.Tensor] = mse_distance,
    generation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = [name for name in module_names if name]
    steps = [int(step) for step in patch_steps]
    scores = torch.zeros((len(names), len(steps)), dtype=torch.float32)
    distances = torch.zeros_like(scores)
    for i, name in enumerate(names):
        for j, step in enumerate(steps):
            result = patch_diffusion_module_outputs(
                pipeline,
                clean_prompt=clean_prompt,
                corrupted_prompt=corrupted_prompt,
                module_names=[name],
                patch_steps=[step],
                num_inference_steps=num_inference_steps,
                distance_fn=distance_fn,
                generation_kwargs=generation_kwargs,
            )
            scores[i, j] = float(result.recovery_fraction)
            distances[i, j] = float(result.clean_patched_distance)
    return {"names": names, "steps": steps, "recovery": scores, "patched_distance": distances}
