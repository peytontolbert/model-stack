import torch
from tensor import blocksparse_mask, sparsify_topk, magnitude_prune


def test_sparse_helpers():
    m = blocksparse_mask((64, 64), block=(16, 16), density=0.5)
    assert m.shape == (64, 64)
    x = torch.randn(8, 10)
    vals, idx = sparsify_topk(x, k=3, axis=-1)
    assert vals.shape[-1] == 3 and idx.shape[-1] == 3
    W = torch.randn(32, 32)
    Wp = magnitude_prune(W, density=0.5)
    assert Wp.shape == W.shape

