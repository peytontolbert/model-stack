import torch
from masking import build_prefix_lm_mask
from positional import build_relative_position_indices, relative_position_bias_from_table


def test_prefix_lm_mask_rules():
    T = 6
    p = 2
    m = build_prefix_lm_mask(T, p)
    # prefix rows cannot see continuation cols
    assert m[0, 3]
    # continuation rows can't see future
    assert m[5, 0] == False and m[5, 4] == False and m[5, 5] == False
    assert m[5, 3] == False and m[5, 2] == False
    assert m[3, 4]  # future masked


def test_rpb_indices_and_bias_shape():
    T, S, H, D = 5, 7, 4, 9
    idx = build_relative_position_indices(T, S, max_distance=5)
    assert idx.shape == (T, S)
    table = torch.randn(H, 2 * 5 - 1)
    bias = relative_position_bias_from_table(idx, table)
    assert bias.shape == (1, H, T, S)


