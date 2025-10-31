import pytest
import torch
from shape import split_qkv, merge_qkv, center_pad, assert_mask_shape, assert_boolean_mask, assert_broadcastable


def test_split_merge_qkv_roundtrip():
    x = torch.randn(2, 5, 12)
    q, k, v = split_qkv(x, num_heads=3)
    out = merge_qkv(q, k, v)
    assert out.shape == x.shape


def test_center_pad_length():
    x = torch.randn(2, 5, 3)
    y = center_pad(x, total=9, dim=1)
    assert y.shape[1] == 9


def test_asserts():
    with pytest.raises(ValueError):
        assert_mask_shape(torch.zeros(2, 3, 4, 5, dtype=torch.bool), 2, 3, 4, 6)
    with pytest.raises(TypeError):
        assert_boolean_mask(torch.zeros(2, 3, 4, 5))
    with pytest.raises(ValueError):
        assert_broadcastable(torch.zeros(2, 3), (4, 3))


