import torch
import torch.nn as nn
from .numerics import chunked_norm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF-compatible: accumulate in fp32
        input_dtype = x.dtype
        xf = x.float()
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = xf * torch.rsqrt(variance + self.eps)
        y = x_normalized.to(dtype=input_dtype)
        return y * self.weight.to(dtype=input_dtype, device=x.device)


class ScaleNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, scale: float = 1.0):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(float(scale)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 norm over last dim
        norm = x.norm(dim=-1, keepdim=True)
        return x * (self.g / (norm + self.eps))


def _to_tuple_dims(dim: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if isinstance(dim, tuple):
        dims = dim
    else:
        dims = (dim,)
    out: list[int] = []
    for d in dims:
        dd = d if d >= 0 else ndim + d
        out.append(dd)
    return tuple(out)


def _reshape_param_for_dims(param: torch.Tensor | None, x: torch.Tensor, dims: int | tuple[int, ...]) -> torch.Tensor | None:
    if param is None:
        return None
    dims_t = _to_tuple_dims(dims, x.ndim)
    target_shape = [1] * x.ndim
    for d in dims_t:
        target_shape[d] = x.size(d)
    p = param.to(dtype=x.dtype, device=x.device)
    # Common cases:
    # 1) param has shape equal to the normalized axes
    if p.ndim == len(dims_t) and list(p.shape) == [x.size(d) for d in dims_t]:
        return p.view(*target_shape)
    # 2) param is 1D with product size of normalized axes
    prod = 1
    for d in dims_t:
        prod *= x.size(d)
    if p.ndim == 1 and p.numel() == prod:
        return p.view(*target_shape)
    # 3) param is already broadcastable with x
    if p.ndim == x.ndim:
        return p
    # 4) single-dim fallback
    if len(dims_t) == 1 and p.ndim == 1 and p.shape[0] == x.size(dims_t[0]):
        return p.view(*target_shape)
    return p


def layer_norm_fp32(x: torch.Tensor, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, eps: float = 1e-5) -> torch.Tensor:
    mu = x.float().mean(dim=-1, keepdim=True)
    var = x.float().var(dim=-1, unbiased=False, keepdim=True)
    y = (x.float() - mu) / torch.sqrt(var + eps)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(dtype=x.dtype)


def masked_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    dim: int = -1,
) -> torch.Tensor:
    """RMSNorm that excludes masked elements from statistics (mask True = masked).

    - Normalizes along ``dim`` (default last) using fp32 accumulation.
    - If ``mask`` is provided, masked elements are excluded from the mean-of-squares
      and outputs at masked positions are zeroed.
    """
    dims = _to_tuple_dims(dim, x.ndim)
    xf = x.float()
    if mask is not None:
        m = mask.to(torch.bool)
        while m.ndim < x.ndim:
            m = m.unsqueeze(-1)
        m = m.expand_as(x)
        inv = (~m).to(xf.dtype)
        sumsq = (xf * xf * inv).sum(dim=dims, keepdim=True)
        count = inv.sum(dim=dims, keepdim=True).clamp_min(1.0)
        mean_sq = sumsq / count
    else:
        mean_sq = (xf * xf).mean(dim=dims, keepdim=True)
    y = x * torch.rsqrt(mean_sq + eps)
    w = _reshape_param_for_dims(weight, x, dims)
    y = y * w
    if mask is not None:
        y = y.masked_fill(m, 0)
    return y


def rmsnorm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6, dim: int = -1) -> torch.Tensor:
    """Functional RMSNorm with axis support and fp32 accumulation."""
    dims = _to_tuple_dims(dim, x.ndim)
    mean_sq = (x.float() * x.float()).mean(dim=dims, keepdim=True)
    y = x * torch.rsqrt(mean_sq + eps)
    if weight is not None:
        w = _reshape_param_for_dims(weight, x, dims)
        y = y * w
    return y.to(dtype=x.dtype)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    dim: int | tuple[int, ...] = -1,
) -> torch.Tensor:
    """Functional LayerNorm with axis support and fp32 accumulation."""
    dims = _to_tuple_dims(dim, x.ndim)
    xf = x.float()
    mu = xf.mean(dim=dims, keepdim=True)
    var = xf.var(dim=dims, unbiased=False, keepdim=True)
    y = (xf - mu) / torch.sqrt(var + eps)
    if weight is not None:
        y = y * _reshape_param_for_dims(weight, x, dims)
    if bias is not None:
        y = y + _reshape_param_for_dims(bias, x, dims)
    return y.to(dtype=x.dtype)


def mean_only_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    dim: int | tuple[int, ...] = -1,
) -> torch.Tensor:
    """Centering-only LayerNorm (subtract mean; no variance scaling)."""
    dims = _to_tuple_dims(dim, x.ndim)
    xf = x.float()
    mu = xf.mean(dim=dims, keepdim=True)
    y = xf - mu
    if weight is not None:
        y = y * _reshape_param_for_dims(weight, x, dims)
    if bias is not None:
        y = y + _reshape_param_for_dims(bias, x, dims)
    return y.to(dtype=x.dtype)


def lp_norm(
    x: torch.Tensor,
    p: float = 2.0,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    dim: int | tuple[int, ...] = -1,
) -> torch.Tensor:
    """Lp/Power normalization: x / (E[|x|^p] + eps)^(1/p)."""
    dims = _to_tuple_dims(dim, x.ndim)
    xf = x.float().abs().pow(float(p))
    mean_p = xf.mean(dim=dims, keepdim=True)
    denom = torch.pow(mean_p + eps, 1.0 / float(p))
    y = x / denom
    if weight is not None:
        y = y * _reshape_param_for_dims(weight, x, dims)
    return y.to(dtype=x.dtype)


def power_norm(
    x: torch.Tensor,
    p: float = 2.0,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    dim: int | tuple[int, ...] = -1,
) -> torch.Tensor:
    """Alias of lp_norm for convenience."""
    return lp_norm(x, p=p, weight=weight, eps=eps, dim=dim)


def chunked_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    dim: int = -1,
    chunk: int | None = None,
) -> torch.Tensor:
    """RMSNorm using chunked reductions to smooth peak memory along ``dim``.

    Only a single axis is supported for chunking.
    """
    d = dim if dim >= 0 else x.ndim + dim
    l2 = chunked_norm(x, ord=2, dim=d, chunk=chunk).float()  # shape with dim squeezed
    # Restore reduced axis
    l2 = l2.unsqueeze(d)
    n = float(x.size(d))
    mean_sq = (l2 * l2) / n
    y = x * torch.rsqrt(mean_sq + eps)
    if weight is not None:
        w = _reshape_param_for_dims(weight, x, d)
        y = y * w
    return y.to(dtype=x.dtype)


def spectral_norm(weight: torch.Tensor, iters: int = 1, u: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Power iteration spectral normalization (functional).

    Returns (normalized_weight, u_new). ``u`` is a persistent estimate of the left singular vector.
    """
    w = weight.view(weight.shape[0], -1).float()
    device = weight.device
    if u is None:
        u = torch.randn(w.shape[0], device=device)
    u = u.float() / (u.norm() + 1e-12)
    v = None
    for _ in range(max(1, int(iters))):
        v = (w.t() @ u).clamp_min(1e-45)
        v = v / (v.norm() + 1e-12)
        u = (w @ v).clamp_min(1e-45)
        u = u / (u.norm() + 1e-12)
    sigma = (u @ (w @ v)).clamp_min(1e-12)
    w_hat = (weight.float() / sigma).to(dtype=weight.dtype)
    return w_hat, u.to(dtype=weight.dtype)


def weight_norm(weight: torch.Tensor, g: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional weight normalization.

    Normalizes ``weight`` over all dims except the first (out/features) and rescales by ``g``.
    Returns (reparam_weight, g_used).
    """
    w = weight.float()
    out = w.shape[0]
    v = w.view(out, -1)
    v_norm = v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if g is None:
        g_used = v_norm.squeeze(1).to(dtype=weight.dtype)
    else:
        g_used = g.to(dtype=weight.dtype)
    w_hat = (v / v_norm) * g_used.view(out, 1).to(dtype=w.dtype)
    w_hat = w_hat.view_as(weight).to(dtype=weight.dtype)
    return w_hat, g_used


def orthogonalize(weight: torch.Tensor, scheme: str = "cayley", steps: int = 1) -> torch.Tensor:
    """Return an orthogonalized version of ``weight``.

    - scheme="qr": uses QR factorization (Q)
    - scheme="cayley": uses Cayley transform of the skew-symmetric part (square-only)
    """
    if scheme == "qr":
        q, _ = torch.linalg.qr(weight.float(), mode="reduced")
        return q.to(dtype=weight.dtype)
    if scheme == "householder":
        # Construct Householder orthogonalization via QR fallback for now; steps unused
        q, _ = torch.linalg.qr(weight.float(), mode="reduced")
        return q.to(dtype=weight.dtype)
    # Cayley fallback
    W = weight.float()
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        q, _ = torch.linalg.qr(W, mode="reduced")
        return q.to(dtype=weight.dtype)
    A = 0.5 * (W - W.t())
    I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
    # (I - A)^{-1}(I + A)
    try:
        X = torch.linalg.solve(I - A, I + A)
    except Exception:
        X = torch.linalg.pinv(I - A) @ (I + A)
    return X.to(dtype=weight.dtype)


def rolling_norm(x: torch.Tensor, axis: int = -2, window: int | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Compute rolling RMS norm over a sliding window along ``axis``.

    If ``window`` is None, uses full prefix windows. Outputs same shape as x.
    """
    d = axis if axis >= 0 else x.ndim + axis
    xf = x.float()
    if window is None:
        csum = (xf * xf).cumsum(dim=d)
        idx = torch.arange(x.size(d), device=x.device).view([1] * d + [-1] + [1] * (x.ndim - d - 1)) + 1
        mean_sq = csum / idx
        rms = xf / torch.sqrt(mean_sq + eps)
        return rms.to(dtype=x.dtype)
    # fixed-size window via unfold if possible
    if d != x.ndim - 2 and d != x.ndim - 1:
        # generic: fall back to naive loop
        out = torch.empty_like(xf)
        it = [slice(None)] * x.ndim
        for i in range(x.size(d)):
            lo = max(0, i - window + 1)
            hi = i + 1
            it[d] = slice(lo, hi)
            seg = xf[tuple(it)]
            ms = (seg * seg).mean(dim=d, keepdim=True)
            out.select(d, i).copy_(xf.select(d, i) / torch.sqrt(ms.squeeze(d) + eps))
        return out.to(dtype=x.dtype)
    # try unfold on last dims by transposing
    perm = list(range(x.ndim))
    perm[d], perm[-2] = perm[-2], perm[d]
    y = xf.permute(*perm)
    B = y.shape[:-2]
    T = y.shape[-2]
    C = y.shape[-1]
    pad = (window - 1, 0)
    y2 = (y * y).transpose(-2, -1)
    y2 = torch.nn.functional.pad(y2, (pad[0], pad[1]))
    y2 = y2.unfold(-1, window, 1)  # (..., C, T, window)
    ms = y2.mean(dim=-1).transpose(-2, -1)
    rms = y / torch.sqrt(ms + eps)
    inv = [0] * x.ndim
    for i, p in enumerate(perm):
        inv[p] = i
    out = rms.permute(*inv)
    return out.to(dtype=x.dtype)


def online_rms(x: torch.Tensor, state: dict | None = None, eps: float = 1e-6) -> tuple[torch.Tensor, dict]:
    """Online RMS normalization using Welford updates over the last dim.

    ``state`` should be a dict with keys {mean, var, count} shaped broadcastable to x with last dim kept.
    Returns (y, new_state).
    """
    xf = x.float()
    if state is None:
        mean = torch.zeros_like(xf[..., :1])
        var = torch.zeros_like(xf[..., :1])
        count = torch.zeros_like(xf[..., :1])
    else:
        mean = state["mean"].float()
        var = state["var"].float()
        count = state["count"].float()
    from .numerics import welford_update
    mean, var, count = welford_update(mean, var, count, xf.mean(dim=-1, keepdim=True))
    rms = xf / torch.sqrt(var / count.clamp_min(1.0) + eps)
    new_state = {"mean": mean.to(dtype=x.dtype), "var": var.to(dtype=x.dtype), "count": count.to(dtype=x.dtype)}
    return rms.to(dtype=x.dtype), new_state

