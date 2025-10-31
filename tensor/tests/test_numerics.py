import torch
from numerics import safe_softmax, masked_log_softmax, masked_logsumexp


def test_safe_softmax_masked_probability_zero():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[False, True, False]])
    p = safe_softmax(x, mask=mask, dim=-1)
    assert torch.allclose(p.sum(dim=-1), torch.ones(1))
    assert p[0, 1].item() == 0.0


def test_masked_log_softmax_and_logsumexp():
    x = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.tensor([[False, True, False]])
    lsm = masked_log_softmax(x, mask=mask, dim=-1)
    # Only two tokens active -> each gets log(0.5)
    assert torch.allclose(lsm[0, [0, 2]], torch.log(torch.tensor([0.5, 0.5])))
    assert lsm[0, 1].item() < -1e6 or torch.isneginf(lsm[0, 1].float())
    lse = masked_logsumexp(x, mask=mask, dim=-1)
    # logsumexp over active zeros -> log(2)
    assert torch.allclose(lse, torch.log(torch.tensor([2.0])))


