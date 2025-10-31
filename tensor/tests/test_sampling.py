import torch
from sampling import apply_temperature, apply_repetition_penalty, apply_min_p_mask, apply_typical_mask, mixture_of_logits


def test_apply_temperature_scales_logits():
    x = torch.tensor([[1.0, 2.0]])
    y = apply_temperature(x, 2.0)
    assert torch.allclose(y, x / 2.0)


def test_min_p_and_typical_masks_shapes():
    x = torch.randn(2, 5)
    m1 = apply_min_p_mask(x, 0.1)
    m2 = apply_typical_mask(x, 1.0)
    assert m1.shape == x.shape and m2.shape == x.shape


def test_mixture_of_logits_returns_log_probs():
    a = torch.zeros(2, 4)
    b = torch.zeros(2, 4)
    out = mixture_of_logits([a, b], [0.7, 0.3], temperature=1.0)
    probs = out.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))


