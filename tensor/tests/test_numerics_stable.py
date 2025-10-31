import torch
from numerics.compensate import kahan_sum
from numerics.stable import safe_softmax_with_logsumexp
from numerics import entropy_from_logits, entropy_from_probs, temperature_linear, temperature_cosine


def test_kahan_sum_matches_sum():
    x = torch.randn(1000, 8, dtype=torch.float16)
    s1 = x.sum(dim=0).float()
    s2 = kahan_sum(x, dim=0).float()
    assert torch.allclose(s1, s2, atol=1e-2, rtol=1e-2)


def test_safe_softmax_with_logsumexp():
    logits = torch.randn(3, 5)
    probs, logZ = safe_softmax_with_logsumexp(logits)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3))
    expected_logZ = torch.logsumexp(logits, dim=-1)
    assert torch.allclose(logZ, expected_logZ, atol=1e-6)


def test_entropy_helpers_and_temperature_schedules():
    logits = torch.randn(2, 4)
    ent1 = entropy_from_logits(logits)
    probs = torch.softmax(logits, dim=-1)
    ent2 = entropy_from_probs(probs)
    assert torch.all(ent1 >= 0) and torch.allclose(ent1, ent2, atol=1e-5)
    t_lin = temperature_linear(1.0, 0.5, step=5, total_steps=10)
    t_cos = temperature_cosine(1.0, 0.5, step=5, total_steps=10)
    assert 0.5 <= t_lin <= 1.0 and 0.5 <= t_cos <= 1.0


