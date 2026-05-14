from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MultimodalInputs:
    input_ids: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    extra_kwargs: dict[str, Any] | None = None

    def forward_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.extra_kwargs or {})
        if self.input_ids is not None:
            kwargs["input_ids"] = self.input_ids
        if self.pixel_values is not None:
            kwargs["pixel_values"] = self.pixel_values
        if self.attention_mask is not None:
            kwargs["attention_mask"] = self.attention_mask
        return kwargs


@dataclass(frozen=True)
class MultimodalComponent:
    name: str
    module: nn.Module
    role: str


class MultimodalModelAdapter:
    """Structural adapter for vision-language models.

    Common component names are discovered without importing any specific VLM
    implementation: vision tower/encoder, multimodal projector, and language
    model/decoder.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.vision_encoder = _first_attr(model, ("vision_tower", "vision_encoder", "visual", "vision_model", "image_encoder"))
        self.projector = _first_attr(model, ("multi_modal_projector", "mm_projector", "vision_projector", "projector", "visual_projection"))
        self.language_model = _first_attr(model, ("language_model", "text_model", "model", "decoder", "lm"))
        if self.vision_encoder is None and self.projector is None and self.language_model is None:
            raise TypeError(f"Unsupported multimodal model type: {type(model).__name__}")

    def components(self) -> list[MultimodalComponent]:
        out: list[MultimodalComponent] = []
        for name, role in (
            ("vision_encoder", "vision"),
            ("projector", "projector"),
            ("language_model", "language"),
        ):
            module = getattr(self, name)
            if isinstance(module, nn.Module):
                out.append(MultimodalComponent(name=name, module=module, role=role))
        return out

    def named_modules(self) -> dict[str, nn.Module]:
        out: dict[str, nn.Module] = {"model": self.model}
        for component in self.components():
            out[component.name] = component.module
            for child_name, child in component.module.named_modules():
                if child_name:
                    out[f"{component.name}.{child_name}"] = child
        return out

    def forward(self, inputs: MultimodalInputs | dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if isinstance(inputs, MultimodalInputs):
            merged = inputs.forward_kwargs()
            merged.update(kwargs)
            return self.model(**merged)
        if isinstance(inputs, dict):
            merged = dict(inputs)
            merged.update(kwargs)
            return self.model(**merged)
        return self.model(**kwargs)


def _first_attr(obj: Any, names: tuple[str, ...]) -> Optional[Any]:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, nn.Module):
            return value
    return None


def get_multimodal_adapter(model: nn.Module) -> MultimodalModelAdapter:
    return MultimodalModelAdapter(model)
