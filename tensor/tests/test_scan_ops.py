import torch
from tensor import inclusive_scan, exclusive_scan, stable_cumprod


def test_logsumexp_scans():
    x = torch.randn(3, 5)
    inc = inclusive_scan(x, op="logsumexp", axis=-1)
    exc = exclusive_scan(x, op="logsumexp", axis=-1)
    assert inc.shape == x.shape and exc.shape == x.shape


def test_stable_cumprod():
    x = torch.rand(4, 6).clamp_min(1e-3)
    y = stable_cumprod(x, axis=-1, logspace=True)
    assert y.shape == x.shape

