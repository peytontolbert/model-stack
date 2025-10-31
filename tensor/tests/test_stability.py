import torch
from tensor import pairwise_sum, stable_norm, softplus_inv, logcumsumexp


def test_stability_ops():
    x = torch.randn(7, 5)
    s = pairwise_sum(x, axis=-1)
    n = stable_norm(x, ord=2, axis=-1)
    y = softplus_inv(torch.softplus(x))
    l = logcumsumexp(x, dim=-1)
    assert s.shape == x.shape[:-1]
    assert n.shape[-1] == 1
    assert y.shape == x.shape and l.shape == x.shape

