import torch
from dtypes import FP8AmaxTracker, fp8_scale_from_amax
from random import seed_everything


def test_fp8_tracker_updates():
    tr = FP8AmaxTracker()
    a1 = tr.update(torch.tensor([1.0, 2.0]))
    a2 = tr.update(torch.tensor([3.0]))
    assert a2 >= a1
    scale = fp8_scale_from_amax(a2)
    assert scale > 0


def test_seed_everything_makes_deterministic():
    seed_everything(123)
    a = torch.randn(3)
    seed_everything(123)
    b = torch.randn(3)
    assert torch.allclose(a, b)


