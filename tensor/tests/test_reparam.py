import torch
from tensor import spectral_norm, weight_norm, orthogonalize


def test_reparam_weight_shapes():
    W = torch.randn(16, 8)
    Wsn, u = spectral_norm(W, iters=2)
    Wwn, g = weight_norm(W)
    Wo = orthogonalize(W, scheme="qr")
    assert Wsn.shape == W.shape and Wwn.shape == W.shape and Wo.shape[0] == W.shape[0]

