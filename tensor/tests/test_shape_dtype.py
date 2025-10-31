import torch
from shape import split_heads, merge_heads, ensure_even_last_dim
from dtypes import cast_for_softmax, cast_for_norm, to_dtype_like, restore_dtype


def test_split_merge_heads_roundtrip():
    x = torch.randn(2, 5, 12)
    h = 3
    xh = split_heads(x, h)
    xr = merge_heads(xh)
    assert xr.shape == x.shape
    assert torch.allclose(x, xr)


def test_ensure_even_last_dim():
    ok = torch.randn(1, 1, 8)
    ensure_even_last_dim(ok)


def test_dtype_helpers():
    x = torch.randn(2, 3, dtype=torch.float16)
    xf = cast_for_softmax(x)
    xn = cast_for_norm(x)
    assert xf.dtype == torch.float32 and xn.dtype == torch.float32
    y = torch.zeros(2, 3, dtype=torch.bfloat16)
    z = to_dtype_like(xf, y)
    assert z.dtype == torch.bfloat16
    back = restore_dtype(xf, x)
    assert back.dtype == x.dtype


