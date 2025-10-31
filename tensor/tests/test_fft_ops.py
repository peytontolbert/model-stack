import torch
from tensor import fft_conv1d, dct, idct, hilbert_transform, power_spectrum_generic


def test_fft_ops_basic():
    x = torch.randn(2, 32)
    k = torch.randn(9)
    y = fft_conv1d(x, k)
    assert y.shape[-1] == x.shape[-1] + k.shape[-1] - 1
    X = dct(x, type=2, dim=-1)
    xi = idct(X, type=2, dim=-1)
    assert xi.shape == x.shape
    h = hilbert_transform(x, axis=-1)
    p = power_spectrum_generic(x, axis=-1)
    assert h.shape == x.shape and p.shape[0] == x.shape[0]

