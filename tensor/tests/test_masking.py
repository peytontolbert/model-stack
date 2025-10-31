import torch
from masking import build_causal_mask, build_padding_mask, broadcast_mask, build_sliding_window_causal_mask


def test_broadcast_mask_combines_causal_and_padding():
    B, H, T, S = 2, 3, 5, 5
    causal = build_causal_mask(T)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    padding = build_padding_mask(attention_mask)
    m = broadcast_mask(batch_size=B, num_heads=H, tgt_len=T, src_len=S, causal_mask=causal, padding_mask=padding, padding_mask_is_1_for_token=False)
    assert m.shape == (B, H, T, S)
    # future masked
    assert m[:, :, 0, 1].all()
    # pad masked
    assert m[0, 0, :, 3].all()


def test_sliding_window_causal_mask_window_size():
    T = 6
    w = 2
    m = build_sliding_window_causal_mask(T, w)
    # token i=3 can see j in [2,3]
    allowed = [2, 3]
    for j in range(T):
        if j in allowed:
            assert not m[3, j]
        else:
            assert m[3, j]


