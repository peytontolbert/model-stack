import torch
from losses import masked_cross_entropy, sequence_nll


def test_masked_cross_entropy_padding_ignored():
    B, T, V = 2, 4, 10
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    attn = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
    loss = masked_cross_entropy(logits, targets, attn, reduction="mean")
    assert loss.ndim == 0


def test_sequence_nll_matches_cross_entropy():
    B, T, V = 1, 3, 5
    logits = torch.zeros(B, T, V)
    targets = torch.zeros(B, T, dtype=torch.long)
    attn = torch.ones(B, T, dtype=torch.long)
    ce = masked_cross_entropy(logits, targets, attn)
    nll = sequence_nll(logits, targets, attn)
    assert torch.allclose(ce, nll)


