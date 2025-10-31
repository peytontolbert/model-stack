import torch


def kahan_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # simple loop over dim; for long dims only (not vectorized)
    x = x.moveaxis(dim, 0)
    s = torch.zeros_like(x[0])
    c = torch.zeros_like(s)
    for i in range(x.shape[0]):
        y = x[i] - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


