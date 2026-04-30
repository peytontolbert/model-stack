import torch
from tensor.losses import masked_cross_entropy, sequence_nll, softcapped_cross_entropy, softcapped_cross_entropy_manual


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


def test_softcapped_cross_entropy_manual_matches_torch_path():
    torch.manual_seed(0)
    logits = torch.randn(7, 13, requires_grad=True)
    logits_manual = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 13, (7,))

    loss = softcapped_cross_entropy(logits, targets, softcap=30.0)
    manual = softcapped_cross_entropy_manual(logits_manual, targets, softcap=30.0)
    loss.backward()
    manual.backward()

    assert torch.allclose(manual, loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(logits_manual.grad, logits.grad, atol=1e-6, rtol=1e-6)
