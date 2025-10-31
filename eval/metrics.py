import torch


def perplexity(loss: float) -> float:
    return float(torch.exp(torch.tensor(loss)))


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute next-token accuracy given logits (B,T,V) and targets (B,T).

    If mask is provided, it should be (B,T) with 1.0 for valid tokens.
    Returns a scalar tensor accuracy.
    """
    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float()
    if mask is not None:
        denom = mask.to(correct.dtype).sum().clamp_min(1.0)
        return (correct * mask.to(correct.dtype)).sum() / denom
    return correct.mean()

