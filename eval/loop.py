from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable

import torch

from tensor.losses import sequence_nll, masked_perplexity
from .metrics import token_accuracy


@dataclass
class EvalResult:
    nll: float
    ppl: float
    acc: Optional[float] = None
    num_tokens: int = 0


@torch.no_grad()
def evaluate_lm_next_token(
    model: torch.nn.Module,
    data: Iterable,
    *,
    device: Optional[torch.device | str] = None,
    max_batches: Optional[int] = None,
    report_accuracy: bool = True,
) -> EvalResult:
    """Evaluate next-token LM metrics (NLL, perplexity, optional accuracy).

    Expects each batch to have fields input_ids[B,T] and attn_mask (or None).
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0.0

    for i, batch in enumerate(data):
        if max_batches is not None and i >= max_batches:
            break
        x = batch.input_ids.to(dev)
        mask = batch.attn_mask
        logits = model(x, mask, None)
        # shift for next-token prediction
        logits_shift = logits[:, :-1, :]
        targets = x[:, 1:]
        mask_shift = None if mask is None else mask[:, 1:]

        nll = sequence_nll(logits_shift, targets, mask_shift, reduction="sum")
        total_nll += float(nll.item())
        num = int(targets.numel()) if mask_shift is None else int(mask_shift.sum().item())
        total_tokens += num

        if report_accuracy:
            acc_num = token_accuracy(logits_shift, targets, mask_shift)
            total_correct += float(acc_num.item()) * (num if mask_shift is not None else targets.numel())

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    acc = None
    if report_accuracy and total_tokens > 0:
        acc = total_correct / total_tokens
    return EvalResult(nll=mean_nll, ppl=ppl, acc=acc, num_tokens=total_tokens)


