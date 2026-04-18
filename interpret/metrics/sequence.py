from __future__ import annotations

from typing import Optional

import torch

from tensor.losses import masked_perplexity, sequence_nll


@torch.inference_mode()
def sequence_perplexity(logits: torch.Tensor, targets: torch.Tensor, *, mask: Optional[torch.Tensor] = None, dim: int = -1) -> torch.Tensor:
    """Wrapper for masked_perplexity (returns scalar)."""
    if dim != -1:
        raise ValueError("sequence_perplexity only supports class logits on the last dimension")
    return masked_perplexity(logits, targets, mask=mask)


@torch.inference_mode()
def sequence_negative_log_likelihood(logits: torch.Tensor, targets: torch.Tensor, *, mask: Optional[torch.Tensor] = None, dim: int = -1) -> torch.Tensor:
    if dim != -1:
        raise ValueError("sequence_negative_log_likelihood only supports class logits on the last dimension")
    return sequence_nll(logits, targets, attention_mask=mask)


