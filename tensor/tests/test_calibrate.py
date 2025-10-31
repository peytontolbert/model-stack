import torch
from tensor import percentile_scale, mse_scale, range_track, ema_range_track


def test_scales_and_trackers():
    x = torch.randn(10, 10)
    s1 = percentile_scale(x, p=0.9)
    s2 = mse_scale(x)
    st = range_track(x)
    st2 = ema_range_track(x, state=st)
    assert s1.dtype == x.dtype and s2.dtype == x.dtype
    assert "lo" in st2 and "hi" in st2

