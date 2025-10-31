import torch
from positional import build_rope_cache_2d


def test_rope_cache_2d_shapes():
    H, W, Dh = 4, 6,  eight = 8
    Dh = 8
    cos, sin = build_rope_cache_2d(H, W, Dh)
    assert cos.shape == (H, W, Dh)
    assert sin.shape == (H, W, Dh)


