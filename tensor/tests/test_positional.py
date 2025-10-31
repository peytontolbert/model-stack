import torch
from positional import build_rope_cache, apply_rotary, build_alibi_bias


def test_rope_cache_shapes():
    cos, sin = build_rope_cache(seq_len=8, head_dim=16, device="cpu")
    assert cos.shape == (8, 16)
    assert sin.shape == (8, 16)


def test_apply_rotary_roundtrip_shapes():
    B, H, T, Dh = 2, 4, 8, 16
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    cos, sin = build_rope_cache(T, Dh, device=q.device)
    q2, k2 = apply_rotary(q, k, cos, sin)
    assert q2.shape == q.shape and k2.shape == k.shape


def test_alibi_bias_shape_and_monotonicity():
    H, T = 4, 8
    bias = build_alibi_bias(H, T, device="cpu")
    assert bias.shape == (1, H, T, T)
    # check that bias becomes more negative with distance for head 0
    b = bias[0, 0]
    assert torch.all(b.diag() == 0)
    assert (b[7, 0] < b[7, 1] < b[7, 2]).item() is True


