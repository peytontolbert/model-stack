from __future__ import annotations

from typing import Optional

import torch

from tensor.losses import masked_perplexity, sequence_nll


@torch.inference_mode()
def sequence_perplexity(logits: torch.Tensor, targets: torch.Tensor, *, mask: Optional[torch.Tensor] = None, dim: int = -1) -> torch.Tensor:
    """Wrapper for masked_perplexity (returns scalar)."""
    return masked_perplexity(logits, targets, mask=mask, dim=dim)


@torch.inference_mode()
def sequence_negative_log_likelihood(logits: torch.Tensor, targets: torch.Tensor, *, mask: Optional[torch.Tensor] = None, dim: int = -1) -> torch.Tensor:
    return sequence_nll(logits, targets, mask=mask, dim=dim)


