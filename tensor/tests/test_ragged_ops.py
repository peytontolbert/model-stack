import torch
from tensor import ragged_gather, ragged_scatter


def test_ragged_gather_scatter():
    x = torch.arange(10).view(10, 1).float()
    offsets = torch.tensor([0, 4, 7])
    lengths = torch.tensor([4, 3])
    g = ragged_gather(x, offsets, lengths)
    assert g.shape == (2, 4, 1)
    s = ragged_scatter(g.view(-1, 1)[:7], offsets, lengths, T=4)
    assert s.shape == (2, 4, 1)

