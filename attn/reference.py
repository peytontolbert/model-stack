import torch
from tensor.numerics import safe_softmax


def compute_attention_scores(q: torch.Tensor, k: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    # q: (B,H,T,D), k: (B,H,S,D) -> scores: (B,H,T,S)
    scores = torch.matmul(q, k.transpose(-2, -1))
    if scale is not None:
        scores = scores * float(scale)
    return scores


def compute_attention_probs(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    scores = compute_attention_scores(q, k, scale=scale)
    if bias is not None:
        scores = scores + bias
    probs = safe_softmax(scores, mask=mask, dim=-1)
    return probs


def apply_attention_probs(v: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    # v: (B,H,S,D), probs: (B,H,T,S) -> out: (B,H,T,D)
    return torch.matmul(probs, v)


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    probs = compute_attention_probs(q, k, mask=mask, bias=bias, scale=scale)
    return apply_attention_probs(v, probs)


