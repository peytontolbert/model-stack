import torch
from norms import RMSNorm
from mlp import MLP


def test_rmsnorm_normalizes_variance():
    x = torch.randn(4, 8, 16)
    norm = RMSNorm(16)
    y = norm(x)
    var = (y.pow(2).mean(dim=-1)).mean().item()
    assert 0.5 < var < 2.0  # around 1.0; loose bound


def test_mlp_shapes():
    x = torch.randn(2, 10, 32)
    mlp = MLP(32, 64, activation="silu")
    y = mlp(x)
    assert y.shape == x.shape


