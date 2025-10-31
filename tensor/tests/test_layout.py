import torch
from tensor import bhdt_to_bthd, pack_heads, unpack_heads, rechunk, tile


def test_tile_symbol_aware():
    x = torch.randn(2, 3, 4, 5)
    y = tile(x, ("B", 2, "T", 1))
    assert y.shape == (2, 6, 4, 5)


def test_pack_unpack_heads():
    q = torch.randn(2, 8, 4, 16)
    qp, kp, vp = pack_heads(q, q, q, groups=2)
    qu, ku, vu = unpack_heads(qp, kp, vp)
    assert qu.shape == q.shape

