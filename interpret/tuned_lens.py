from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class AffineTunedLens(nn.Module):
    """Layer-indexed affine translators before unembedding."""

    def __init__(self, n_layers: int, d_model: int, *, bias: bool = True) -> None:
        super().__init__()
        self.translators = nn.ModuleList([nn.Linear(d_model, d_model, bias=bias) for _ in range(int(n_layers))])
        for translator in self.translators:
            nn.init.eye_(translator.weight)
            if translator.bias is not None:
                nn.init.zeros_(translator.bias)

    def forward(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        return self.translators[int(layer)](hidden)


def tuned_lens_logits(
    hidden_by_layer: dict[int, torch.Tensor],
    lm_head: nn.Module | torch.Tensor,
    *,
    lens: AffineTunedLens | None = None,
) -> dict[int, torch.Tensor]:
    weight = lm_head if isinstance(lm_head, torch.Tensor) else getattr(lm_head, "weight")
    bias = None if isinstance(lm_head, torch.Tensor) else getattr(lm_head, "bias", None)
    out: dict[int, torch.Tensor] = {}
    for layer, hidden in hidden_by_layer.items():
        translated = lens(hidden, layer) if lens is not None else hidden
        logits = translated @ weight.to(device=translated.device, dtype=translated.dtype).T
        if isinstance(bias, torch.Tensor):
            logits = logits + bias.to(device=translated.device, dtype=translated.dtype)
        out[int(layer)] = logits
    return out


def tuned_lens_topk(
    hidden_by_layer: dict[int, torch.Tensor],
    lm_head: nn.Module | torch.Tensor,
    *,
    lens: AffineTunedLens | None = None,
    position: int = -1,
    topk: int = 10,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    logits = tuned_lens_logits(hidden_by_layer, lm_head, lens=lens)
    out = {}
    for layer, value in logits.items():
        pos_logits = value[:, position, :] if value.ndim == 3 else value
        vals, idx = torch.topk(pos_logits, k=min(int(topk), pos_logits.shape[-1]), dim=-1)
        out[layer] = (idx.detach().cpu(), vals.detach().cpu())
    return out


def collect_layer_hidden_from_cache(cache, *, prefix: str = "blocks.", suffix: str = ".resid_post") -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for key, value in cache.items():
        if not key.startswith(prefix) or not key.endswith(suffix) or not isinstance(value, torch.Tensor):
            continue
        middle = key[len(prefix) : -len(suffix)]
        if middle.isdigit():
            out[int(middle)] = value
    return out


def tuned_lens_training_pairs(cache, final_hidden: torch.Tensor, *, layers: Iterable[int] | None = None) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    hidden_by_layer = collect_layer_hidden_from_cache(cache)
    selected = sorted(hidden_by_layer) if layers is None else [int(layer) for layer in layers]
    return [(layer, hidden_by_layer[layer], final_hidden) for layer in selected if layer in hidden_by_layer]
