import torch
from export_safe import gelu_export, rmsnorm_export, gather2d_export, gather1d_export
from shard import tp_linear_partition, kv_partition, estimate_activation_bytes


def test_export_safe_funcs():
    x = torch.randn(2, 3)
    y = gelu_export(x)
    assert y.shape == x.shape
    w = torch.ones(3)
    z = rmsnorm_export(x, w)
    assert z.shape == x.shape
    logits = torch.randn(2, 4, 5)
    idx = torch.randint(0, 5, (2, 4))
    g = gather2d_export(logits, idx)
    assert g.shape == (2, 4)
    v = torch.randn(5)
    gi = torch.tensor([0, 2, 4])
    g1 = gather1d_export(v, gi, dim=0)
    assert g1.shape == (3,)


def test_shard_helpers():
    p = tp_linear_partition(16, 32, tp_degree=4, scheme="row")
    assert len(p["in_shards"]) == 4
    hr, tr = kv_partition(12, 1024, tp_degree=3)
    assert len(hr) == 3 and len(tr) == 3
    bytes_est = estimate_activation_bytes(2, 8, 16, dtype="fp16")
    assert bytes_est == 2 * 8 * 16 * 2


