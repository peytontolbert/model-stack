import torch
from tensor import bilinear_discretize, zoh_discretize, ssm_step, ssm_stability_margin


def test_discretize_and_margin():
    A = torch.tensor([[0.0, 1.0], [-1.0, -0.1]])
    B = torch.tensor([[0.0], [1.0]])
    Ad1, Bd1 = bilinear_discretize(A, B, 0.1)
    Ad2, Bd2 = zoh_discretize(A, B, 0.1)
    assert Ad1.shape == A.shape and Bd1.shape[0] == A.shape[0]
    m = ssm_stability_margin(A)
    assert m.dtype == A.dtype

