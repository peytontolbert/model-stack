from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DiffusionInputs:
    prompt: str | list[str]
    negative_prompt: str | list[str] | None = None
    num_inference_steps: int | None = None
    generator: torch.Generator | None = None
    latents: torch.Tensor | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(self.extra_kwargs)
        kwargs["prompt"] = self.prompt
        if self.negative_prompt is not None:
            kwargs["negative_prompt"] = self.negative_prompt
        if self.num_inference_steps is not None:
            kwargs["num_inference_steps"] = int(self.num_inference_steps)
        if self.generator is not None:
            kwargs["generator"] = self.generator
        if self.latents is not None:
            kwargs["latents"] = self.latents
        return kwargs


@dataclass(frozen=True)
class DiffusionComponent:
    name: str
    module: nn.Module
    role: str


@dataclass(frozen=True)
class DiffusionAttentionTarget:
    name: str
    module: nn.Module
    kind: str


class DiffusionModelAdapter:
    """Normalize common text-to-image diffusion pipeline surfaces.

    The adapter is intentionally structural instead of tied to Diffusers imports.
    It supports objects exposing common attributes such as ``text_encoder``,
    ``unet``/``transformer``, ``vae``, ``scheduler``, and ``tokenizer``.
    """

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.text_encoder = getattr(pipeline, "text_encoder", None)
        self.text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
        self.tokenizer = getattr(pipeline, "tokenizer", None)
        self.tokenizer_2 = getattr(pipeline, "tokenizer_2", None)
        self.denoiser = (
            getattr(pipeline, "unet", None)
            or getattr(pipeline, "transformer", None)
            or getattr(pipeline, "dit", None)
            or getattr(pipeline, "denoiser", None)
        )
        self.vae = getattr(pipeline, "vae", None)
        self.scheduler = getattr(pipeline, "scheduler", None)
        if self.denoiser is None and not callable(pipeline):
            raise TypeError(f"Unsupported diffusion pipeline type: {type(pipeline).__name__}")

    def components(self) -> list[DiffusionComponent]:
        out: list[DiffusionComponent] = []
        for name, role in (
            ("text_encoder", "text_encoder"),
            ("text_encoder_2", "text_encoder"),
            ("denoiser", "denoiser"),
            ("vae", "vae"),
        ):
            module = getattr(self, name)
            if isinstance(module, nn.Module):
                out.append(DiffusionComponent(name=name, module=module, role=role))
        return out

    def named_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        for component in self.components():
            modules[component.name] = component.module
            for child_name, child in component.module.named_modules():
                if child_name:
                    modules[f"{component.name}.{child_name}"] = child
        return modules

    def cross_attention_targets(self, predicate: Optional[Callable[[str, nn.Module], bool]] = None) -> list[DiffusionAttentionTarget]:
        pred = predicate or is_likely_cross_attention
        out: list[DiffusionAttentionTarget] = []
        for name, module in self.named_modules().items():
            if pred(name, module):
                out.append(DiffusionAttentionTarget(name=name, module=module, kind="cross"))
        return out

    def denoiser_module(self) -> nn.Module | None:
        return self.denoiser if isinstance(self.denoiser, nn.Module) else None

    def encode_prompt(self, prompt: str | list[str], **kwargs: Any) -> Any:
        if hasattr(self.pipeline, "encode_prompt"):
            return self.pipeline.encode_prompt(prompt=prompt, **kwargs)
        if self.tokenizer is None or self.text_encoder is None:
            raise TypeError("Prompt encoding requires pipeline.encode_prompt or tokenizer + text_encoder")
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        device = next(self.text_encoder.parameters()).device
        return self.text_encoder(input_ids.to(device))

    def generate(self, inputs: DiffusionInputs | str | list[str], **kwargs: Any) -> Any:
        if not isinstance(inputs, DiffusionInputs):
            inputs = DiffusionInputs(prompt=inputs, extra_kwargs=kwargs)
        else:
            merged = dict(inputs.extra_kwargs)
            merged.update(kwargs)
            inputs = DiffusionInputs(
                prompt=inputs.prompt,
                negative_prompt=inputs.negative_prompt,
                num_inference_steps=inputs.num_inference_steps,
                generator=inputs.generator,
                latents=inputs.latents,
                extra_kwargs=merged,
            )
        if not callable(self.pipeline):
            raise TypeError("Diffusion pipeline is not callable")
        return self.pipeline(**inputs.generation_kwargs())

    def tokenize_prompt(self, prompt: str | list[str]) -> list[str]:
        if self.tokenizer is None:
            if isinstance(prompt, str):
                return prompt.split()
            return [tok for item in prompt for tok in str(item).split()]
        if hasattr(self.tokenizer, "tokenize"):
            return list(self.tokenizer.tokenize(prompt))
        encoded = self.tokenizer(prompt)
        if isinstance(encoded, dict) and "input_ids" in encoded and hasattr(self.tokenizer, "convert_ids_to_tokens"):
            ids = encoded["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids[0].tolist() if ids.ndim > 1 else ids.tolist()
            elif ids and isinstance(ids[0], list):
                ids = ids[0]
            return list(self.tokenizer.convert_ids_to_tokens(ids))
        if isinstance(prompt, str):
            return prompt.split()
        return [tok for item in prompt for tok in str(item).split()]


def get_diffusion_adapter(pipeline: Any) -> DiffusionModelAdapter:
    return DiffusionModelAdapter(pipeline)


def is_likely_cross_attention(name: str, module: nn.Module) -> bool:
    lowered = name.lower()
    if any(part in lowered for part in ("attn2", "cross_attn", "cross_attention", "cross.attn")):
        return True
    if bool(getattr(module, "is_cross_attention", False)):
        return True
    if getattr(module, "cross_attention_dim", None) is not None:
        return True
    return False


def iter_prompt_token_replacements(prompt: str, replacement: str = "") -> Iterable[tuple[int, str, str]]:
    tokens = prompt.split()
    for idx, token in enumerate(tokens):
        replaced = tokens[:idx] + ([replacement] if replacement else []) + tokens[idx + 1 :]
        yield idx, token, " ".join(replaced)
