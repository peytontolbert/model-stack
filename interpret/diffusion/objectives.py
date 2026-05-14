from __future__ import annotations

from typing import Any, Callable

import torch

from .metrics import extract_tensor_output


def tensor_mean_score(output: Any) -> torch.Tensor:
    return extract_tensor_output(output).float().mean()


def tensor_region_score(mask: torch.Tensor, *, channel: int | None = None) -> Callable[[Any], torch.Tensor]:
    mask_f = mask.float()

    def _score(output: Any) -> torch.Tensor:
        image = extract_tensor_output(output).float()
        local_mask = mask_f.to(device=image.device, dtype=image.dtype)
        while local_mask.ndim < image.ndim:
            local_mask = local_mask.unsqueeze(0)
        if channel is not None:
            image = image[:, int(channel) : int(channel) + 1] if image.ndim == 4 else image[..., int(channel) : int(channel) + 1]
        denom = local_mask.sum().clamp_min(1e-12)
        return (image * local_mask).sum() / denom

    return _score


def classifier_logit_score(classifier: Callable[[torch.Tensor], torch.Tensor], class_index: int) -> Callable[[Any], torch.Tensor]:
    def _score(output: Any) -> torch.Tensor:
        image = extract_tensor_output(output).float()
        logits = classifier(image)
        return logits.reshape(logits.shape[0], -1)[0, int(class_index)]

    return _score


def clip_similarity_score(
    image_encoder: Callable[[torch.Tensor], torch.Tensor],
    text_encoder: Callable[[str | list[str]], torch.Tensor],
    prompt: str | list[str],
) -> Callable[[Any], torch.Tensor]:
    text_features = text_encoder(prompt)
    if not isinstance(text_features, torch.Tensor):
        text_features = torch.as_tensor(text_features)
    text_features = torch.nn.functional.normalize(text_features.float().flatten(), dim=0)

    def _score(output: Any) -> torch.Tensor:
        image = extract_tensor_output(output).float()
        image_features = image_encoder(image)
        if not isinstance(image_features, torch.Tensor):
            image_features = torch.as_tensor(image_features)
        image_features = torch.nn.functional.normalize(image_features.float().flatten(), dim=0)
        return torch.dot(image_features, text_features.to(device=image_features.device, dtype=image_features.dtype))

    return _score


def attention_entropy(attention_probs: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    probs = attention_probs.float().clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=dim)
