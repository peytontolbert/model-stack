import torch
from ragged import pack_sequences, unpack_sequences, segment_sum, segment_mean, segment_max


def test_pack_unpack_roundtrip():
    x = torch.randn(4, 6, 3)
    lengths = torch.tensor([6, 4, 2, 5])
    packed, idx, rev = pack_sequences(x, lengths)
    out = unpack_sequences(packed, rev, pad_to=6)
    assert out.shape == x.shape


def test_segment_reductions():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    seg = torch.tensor([0, 1, 1])
    s = segment_sum(x, seg)
    m = segment_mean(x, seg)
    mx = segment_max(x, seg)
    assert torch.allclose(s[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(s[1], torch.tensor([8.0, 10.0]))
    assert torch.allclose(m[1], torch.tensor([4.0, 5.0]))
    assert torch.allclose(mx[1], torch.tensor([5.0, 6.0]))


