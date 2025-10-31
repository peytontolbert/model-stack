import torch
from norms import ScaleNorm
from regularization import drop_path, StochasticDepth


def test_scalenorm_scales_by_norm():
    x = torch.randn(3, 7, 13)
    sn = ScaleNorm(13)
    y = sn(x)
    # norms should be roughly equal to learned g initially (1.0)
    norms = y.norm(dim=-1).mean().item()
    assert 0.5 < norms < 2.0


def test_drop_path_train_vs_eval_identity():
    x = torch.randn(4, 5, 6)
    # eval -> identity
    sd = StochasticDepth(0.5)
    sd.eval()
    y = sd(x)
    assert torch.allclose(x, y)
    # drop_prob=0 -> identity even in train
    y2 = drop_path(x, drop_prob=0.0, training=True)
    assert torch.allclose(x, y2)

