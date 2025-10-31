import torch
from masking import combine_masks, build_banded_mask, build_strided_mask, build_block_sparse_mask, build_segment_bidir_mask


def test_combine_masks_broadcast():
    m1 = torch.tensor([[False, True], [False, False]])
    m2 = torch.tensor([False, True]).unsqueeze(0).expand(2, 2)
    out = combine_masks([m1, m2])
    assert out.shape == m1.shape
    assert out.dtype == torch.bool


def test_banded_and_strided_masks():
    T = 6
    band = build_banded_mask(T, 1)
    assert band.shape == (T, T)
    stride = build_strided_mask(T, 2)
    assert stride.shape == (T, T)


def test_block_sparse_mask():
    T = 8
    block = 4
    pattern = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    mask = build_block_sparse_mask(T, block, pattern)
    assert mask.shape == (T, T)


def test_segment_bidir_mask():
    seg = torch.tensor([[0, 0, 1, 1]])
    m = build_segment_bidir_mask(seg)
    assert m.shape == (1, 4, 4)
    assert m[0, 0, 1] == False and m[0, 0, 2] == True


