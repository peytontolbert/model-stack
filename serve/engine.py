from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn.functional as F

from tensor.sampling import (
    apply_temperature,
    apply_topk_mask,
    apply_topp_mask,
    apply_presence_frequency_penalty,
    apply_no_repeat_ngram_mask,
)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    eos_id: Optional[int] = None
    no_repeat_ngram: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def _apply_sampling_policies(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    cfg: GenerationConfig,
) -> torch.Tensor:
    x = logits
    if cfg.temperature is not None and cfg.temperature != 1.0:
        x = apply_temperature(x, cfg.temperature)
    mask = None
    if cfg.top_k is not None:
        m = apply_topk_mask(x, cfg.top_k)
        mask = m if mask is None else (mask | m)
    if cfg.top_p is not None:
        m = apply_topp_mask(x, cfg.top_p)
        mask = m if mask is None else (mask | m)
    if cfg.no_repeat_ngram and cfg.no_repeat_ngram > 0:
        m = apply_no_repeat_ngram_mask(x, input_ids, cfg.no_repeat_ngram)
        mask = m if mask is None else (mask | m)
    if mask is not None:
        min_val = torch.finfo(x.dtype).min if x.dtype.is_floating_point else -1e9
        x = x.masked_fill(mask, min_val)
    if cfg.presence_penalty != 0.0 or cfg.frequency_penalty != 0.0:
        # Build counts (simplified): frequency per token over the sequence
        B, V = x.shape[0], x.shape[-1]
        counts = torch.zeros(B, V, dtype=x.dtype, device=x.device)
        for b in range(B):
            ids, c = input_ids[b].unique(return_counts=True)
            counts[b].index_add_(0, ids, c.to(x.dtype))
        x = apply_presence_frequency_penalty(x, counts, cfg.presence_penalty, cfg.frequency_penalty)
    return x


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    cache=None,
    config: Optional[GenerationConfig] = None,
    sampler: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    cfg = config or GenerationConfig()
    seq = input_ids
    for _ in range(int(cfg.max_new_tokens)):
        # Only pass the last token embedding to enable incremental decoding
        logits = model(seq[:, -1:].contiguous(), None, cache)
        next_logits = logits[:, -1, :]
        sampled_logits = _apply_sampling_policies(next_logits, seq, cfg)
        if sampler is not None:
            next_id = sampler(sampled_logits)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(-1)
        else:
            probs = F.softmax(sampled_logits.float(), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_id], dim=1)
        if cfg.eos_id is not None:
            if (next_id == int(cfg.eos_id)).all():
                break
    return seq


