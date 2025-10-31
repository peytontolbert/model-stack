import torch
import torch.nn.functional as F


def _min_value_for_dtype(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return torch.finfo(dtype).min
    # default fallback
    return -1e9


def safe_softmax(x: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    x_float = x.float()
    if mask is not None:
        x_float = x_float.masked_fill(mask, _min_value_for_dtype(x_float.dtype))
    out = F.softmax(x_float, dim=dim)
    return out.to(dtype=x.dtype)


def masked_log_softmax(x: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    x_float = x.float()
    if mask is not None:
        x_float = x_float.masked_fill(mask, _min_value_for_dtype(x_float.dtype))
    out = F.log_softmax(x_float, dim=dim)
    return out.to(dtype=x.dtype)


def masked_logsumexp(x: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    x_float = x.float()
    if mask is not None:
        x_float = x_float.masked_fill(mask, _min_value_for_dtype(x_float.dtype))
    out = torch.logsumexp(x_float, dim=dim)
    return out.to(dtype=x.dtype)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    # numerically stable log(1 - exp(x)) for x <= 0
    x_float = x.float()
    out = torch.where(x_float < -0.6931, torch.log1p(-torch.exp(x_float)), torch.log(-torch.expm1(x_float)))
    return out.to(dtype=x.dtype)


def expm1_clip(x: torch.Tensor, max_val: float = 1e6) -> torch.Tensor:
    y = torch.expm1(x.float())
    y = torch.clamp(y, max=max_val)
    return y.to(dtype=x.dtype)


def softplus_safe(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    return F.softplus(x, beta=beta, threshold=threshold)


def safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x.float()).to(dtype=x.dtype)


def safe_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x.float()).to(dtype=x.dtype)


def log_sigmoid_safe(x: torch.Tensor) -> torch.Tensor:
    return F.logsigmoid(x.float()).to(dtype=x.dtype)


def log1pexp_safe(x: torch.Tensor) -> torch.Tensor:
    # numerically stable log(1 + exp(x))
    return F.softplus(x)


def logaddexp_many(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute log(sum(exp(x), dim=dim)) with stability."""
    return torch.logsumexp(x.float(), dim=dim).to(dtype=x.dtype)


def welford_update(mean: torch.Tensor, var: torch.Tensor, count: torch.Tensor, x: torch.Tensor):
    # mean/var over last dim by default; reshape to broadcast
    count_new = count + 1
    delta = x - mean
    mean_new = mean + delta / count_new
    delta2 = x - mean_new
    var_new = var + delta * delta2
    return mean_new, var_new, count_new


def finite_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)


def nan_to_num_(x: torch.Tensor, nan: float = 0.0, posinf: float = 1e9, neginf: float = -1e9) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def global_l2_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    total = torch.tensor(0.0, device=tensors[0].device)
    for t in tensors:
        total = total + (t.float().pow(2).sum())
    return total.sqrt()


def scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-8)


def clamp_logits(logits: torch.Tensor, min: float = -30.0, max: float = 30.0) -> torch.Tensor:
    return logits.clamp(min=min, max=max)


def mask_topk(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    # True means masked (not in top-k); inclusive threshold to avoid ties dropping under k
    values, indices = torch.topk(logits, k=k, dim=dim)
    kth = values.select(dim, k - 1).unsqueeze(dim)
    mask = logits < kth
    # ensure at least k unmasked per vector by unmasking exact equals until count==k
    equals = logits == kth
    # leave as is; equality included keeps at least k
    return mask & (~equals)


def mask_topp(logits: torch.Tensor, p: float, dim: int = -1) -> torch.Tensor:
    # Nucleus sampling mask; True -> masked
    probs = F.softmax(logits.float(), dim=dim)
    sorted_probs, sorted_idx = torch.sort(probs, dim=dim, descending=True)
    cum = torch.cumsum(sorted_probs, dim=dim)
    cutoff = cum > p
    # shift to keep at least one token
    cutoff[..., 0] = False
    base = torch.zeros_like(cutoff, dtype=torch.bool)
    mask = base.scatter(dim, sorted_idx, cutoff)
    return mask


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    w = (~mask).to(x.dtype)
    num = (x * w).sum(dim=dim, keepdim=keepdim)
    den = w.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return num / den


# ---- MoE routing helpers ----
def router_topk_with_capacity(scores: torch.Tensor, k: int = 1, capacity_factor: float = 1.25, drop_policy: str = "dropless") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (assignments, combine_weights) for MoE routing with optional capacity.

    - scores: (B,T,E) router logits
    - assignments: LongTensor (B,T,k) expert indices
    - combine_weights: FloatTensor (B,T,k) gating weights (zeroed if dropped, renormalized per token)
    """
    probs = torch.softmax(scores, dim=-1)
    topk_vals, topk_idx = probs.topk(k, dim=-1)

    if not drop_policy.startswith("drop"):
        return topk_idx, topk_vals

    B, T, E = scores.shape
    mean_tokens_per_expert = (B * T * k) / E
    cap = int((mean_tokens_per_expert * capacity_factor + 0.999999))

    dispatch = torch.zeros((B, T, E), dtype=torch.bool, device=scores.device)
    dispatch.scatter_(-1, topk_idx, True)

    N = B * T
    probs_flat = probs.reshape(N, E)
    dispatch_flat = dispatch.reshape(N, E)
    neg_inf = torch.finfo(probs.dtype).min
    masked = torch.where(dispatch_flat, probs_flat, torch.full_like(probs_flat, neg_inf))
    k_keep = min(cap, N)
    vals, idx = masked.topk(k=k_keep, dim=0)
    valid = vals > neg_inf

    keep_flat = torch.zeros_like(dispatch_flat)
    if k_keep > 0:
        cols = torch.arange(E, device=scores.device).unsqueeze(0).expand(k_keep, E)
        keep_flat[idx[valid], cols[valid]] = True
    keep = keep_flat.reshape(B, T, E)

    keep_selected = keep.gather(-1, topk_idx)
    topk_vals = topk_vals * keep_selected.to(topk_vals.dtype)
    denom = topk_vals.sum(dim=-1, keepdim=True)
    topk_vals = torch.where(denom > 0, topk_vals / denom, topk_vals)
    return topk_idx, topk_vals


def masked_var(x: torch.Tensor, mask: torch.Tensor | None, dim: int = -1, keepdim: bool = False, unbiased: bool = False) -> torch.Tensor:
    m = masked_mean(x, mask, dim=dim, keepdim=True)
    if mask is None:
        return (x - m).pow(2).mean(dim=dim, keepdim=keepdim)
    w = (~mask).to(x.dtype)
    num = ((x - m) ** 2 * w).sum(dim=dim, keepdim=keepdim)
    den = w.sum(dim=dim, keepdim=keepdim)
    if unbiased:
        den = (den - 1.0).clamp_min(1.0)
    return num / den.clamp_min(1.0)


def masked_std(x: torch.Tensor, mask: torch.Tensor | None, dim: int = -1, keepdim: bool = False, unbiased: bool = False) -> torch.Tensor:
    return masked_var(x, mask, dim=dim, keepdim=keepdim, unbiased=unbiased).sqrt()


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    return safe_softmax(logits, mask=mask, dim=dim)


# ---- Stable block/chunked utilities (inlined to avoid package shadowing) ----
def _iter_chunks(x: torch.Tensor, dim: int, chunk: int):
    size = x.size(dim)
    for start in range(0, size, chunk):
        end = min(start + chunk, size)
        sl = [slice(None)] * x.ndim
        sl[dim] = slice(start, end)
        yield start, end, tuple(sl)


def _choose_chunk_size(x: torch.Tensor, dim: int, chunk: int | None) -> int:
    import os
    if chunk is not None and chunk > 0:
        return chunk
    env = int(os.environ.get("TENSOR_CHUNK_SIZE", "0"))
    if env > 0:
        return env
    size = x.size(dim)
    bytes_est = x.numel() * x.element_size()
    if size > 8192 or bytes_est > 64 * 1024 * 1024:
        return 4096
    return size


def chunked_softmax(logits: torch.Tensor, dim: int = -1, chunk: int | None = None) -> torch.Tensor:
    x = logits.float()
    chunk = _choose_chunk_size(x, dim, chunk)
    x_max = x.max(dim=dim, keepdim=True).values
    denom = torch.zeros_like(x_max)
    for _, _, sl in _iter_chunks(x, dim, chunk):
        denom = denom + torch.exp(x[sl] - x_max).sum(dim=dim, keepdim=True)
    out = torch.empty_like(x)
    for _, _, sl in _iter_chunks(x, dim, chunk):
        out[sl] = torch.exp(x[sl] - x_max) / denom
    return out.to(dtype=logits.dtype)


def blockwise_logsumexp(x: torch.Tensor, dim: int = -1, block: int | None = None) -> torch.Tensor:
    x = x.float()
    block = _choose_chunk_size(x, dim, block)
    x_max = x.max(dim=dim, keepdim=True).values
    s = torch.zeros_like(x_max)
    for _, _, sl in _iter_chunks(x, dim, block):
        s = s + torch.exp(x[sl] - x_max).sum(dim=dim, keepdim=True)
    return (x_max + torch.log(s)).squeeze(dim)


def masked_softmax_chunked(logits: torch.Tensor, mask: torch.Tensor | None, dim: int = -1, chunk: int | None = None) -> torch.Tensor:
    x = logits.float()
    chunk = _choose_chunk_size(x, dim, chunk)
    if mask is not None:
        x = x.masked_fill(mask, torch.finfo(x.dtype).min)
    return chunked_softmax(x, dim=dim, chunk=chunk).to(dtype=logits.dtype)


def chunked_norm(x: torch.Tensor, ord: int | float = 2, dim: int = -1, chunk: int | None = None) -> torch.Tensor:
    """Stable norm along a dimension computed in chunks to limit peak memory."""
    xf = x.float()
    c = _choose_chunk_size(xf, dim, chunk)
    if ord == 2 or ord == 2.0:
        acc = torch.zeros_like(xf.select(dim, 0), dtype=torch.float32).unsqueeze(dim)
        for _, _, sl in _iter_chunks(xf, dim, c):
            acc = acc + (xf[sl] * xf[sl]).sum(dim=dim, keepdim=True)
        out = acc.sqrt().squeeze(dim)
    else:
        p = float(ord)
        acc = torch.zeros_like(xf.select(dim, 0), dtype=torch.float32).unsqueeze(dim)
        for _, _, sl in _iter_chunks(xf, dim, c):
            acc = acc + xf[sl].abs().pow(p).sum(dim=dim, keepdim=True)
        out = acc.pow(1.0 / p).squeeze(dim)
    return out.to(dtype=x.dtype)


def safe_softmax_with_logsumexp(logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1):
    x = logits.float()
    if mask is not None:
        x = x.masked_fill(mask, torch.finfo(x.dtype).min)
    x_max = x.max(dim=dim, keepdim=True).values
    logZ = x_max + torch.log(torch.exp(x - x_max).sum(dim=dim, keepdim=True))
    probs = torch.exp(x - logZ)
    return probs.to(dtype=logits.dtype), logZ.squeeze(dim).to(dtype=logits.dtype)


def kahan_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Kahan compensated summation along dim for improved numerical stability."""
    xf = x.float()
    s = torch.zeros_like(xf.select(dim, 0))
    c = torch.zeros_like(s)
    it = [slice(None)] * xf.ndim
    for i in range(xf.size(dim)):
        it[dim] = slice(i, i + 1)
        y = xf[tuple(it)].squeeze(dim) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s.to(dtype=x.dtype)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    n = x.float().norm(dim=dim, keepdim=True).clamp_min(eps)
    return (x.float() / n).to(dtype=x.dtype)


def assert_fp32_for_loss(logits: torch.Tensor):
    # No-op unless run in debug contexts; here we raise if bf16/fp16
    if logits.dtype in (torch.float16, torch.bfloat16):
        raise AssertionError("Expected fp32 logits for loss/metrics; use cast_logits_for_loss() or maybe_autocast")


def entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=dim)
    ent = -(probs * torch.log(probs.clamp_min(1e-45))).sum(dim=dim)
    return ent.to(dtype=logits.dtype)


def entropy_from_probs(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ent = -(probs.float() * torch.log(probs.float().clamp_min(1e-45))).sum(dim=dim)
    return ent.to(dtype=probs.dtype)


# Cumulative/stable ops
def logcumsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    try:
        return torch.logcumsumexp(x.float(), dim=dim).to(dtype=x.dtype)
    except Exception:
        # Manual fallback
        x_float = x.float()
        out = torch.empty_like(x_float)
        it = [slice(None)] * x.ndim
        acc = None
        for i in range(x.size(dim)):
            it[dim] = slice(0, i + 1)
            cur = x_float[tuple(it)]
            out.select(dim, i).copy_(torch.logsumexp(cur, dim=dim))
        return out.to(dtype=x.dtype)


def masked_cumsum(x: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
    if mask is None:
        return x.cumsum(dim=dim)
    m = (~mask).to(x.dtype)
    return (x * m).cumsum(dim=dim)


def masked_cummax(x: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
    if mask is None:
        return torch.cummax(x, dim=dim).values
    neg_inf = torch.finfo(x.dtype).min if x.dtype.is_floating_point else -1e9
    y = x.masked_fill(mask, neg_inf)
    return torch.cummax(y, dim=dim).values


def expm1mexp_safe(x: torch.Tensor) -> torch.Tensor:
    """Compute expm1(x) - exp(-x) stably."""
    xf = x.float()
    return (torch.expm1(xf) - torch.exp(-xf)).to(dtype=x.dtype)


def logdiffexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute log(exp(a) - exp(b)) safely; requires a > b."""
    a_, b_ = a.float(), b.float()
    m = torch.maximum(a_, b_)
    return (m + torch.log(torch.exp(a_ - m) - torch.exp(b_ - m).clamp_min(0.0).clamp_min(1e-45))).to(dtype=a.dtype)


# Banded/triangular ops
def banded_mm(A: torch.Tensor, B: torch.Tensor, bandwidth: int) -> torch.Tensor:
    i = torch.arange(A.size(-2), device=A.device).view(-1, 1)
    j = torch.arange(A.size(-1), device=A.device).view(1, -1)
    mask = (j - i).abs() <= int(bandwidth)
    Ab = A * mask.to(A.dtype)
    return Ab @ B


def triangular_mask_mm(A: torch.Tensor, B: torch.Tensor, upper: bool = True) -> torch.Tensor:
    tri = torch.triu if upper else torch.tril
    Am = tri(A)
    return Am @ B


# Stable linalg
def pinv_safe(x: torch.Tensor, rcond: float | None = None) -> torch.Tensor:
    try:
        return torch.linalg.pinv(x.float(), rcond=rcond).to(dtype=x.dtype)
    except Exception:
        return torch.pinverse(x.float()).to(dtype=x.dtype)


def solve_cholesky_safe(A: torch.Tensor, b: torch.Tensor, upper: bool = False) -> torch.Tensor:
    # A should be SPD; do robust cholesky with fallback
    L, info = torch.linalg.cholesky_ex(A.float(), upper=upper)
    x = torch.cholesky_solve(b.float(), L, upper=upper)
    return x.to(dtype=b.dtype)


def assert_prob_simplex(p: torch.Tensor, dim: int = -1, atol: float = 1e-5) -> bool:
    s = p.sum(dim=dim)
    if not torch.allclose(s, torch.ones_like(s), atol=atol):
        raise AssertionError("Probabilities must sum to 1 along dim")
    if (p < -atol).any():
        raise AssertionError("Probabilities must be non-negative")
    return True


def temperature_linear(start: float, end: float, step: int, total_steps: int) -> float:
    w = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return float(start + (end - start) * w)


def temperature_cosine(start: float, end: float, step: int, total_steps: int) -> float:
    import math
    w = (1 - math.cos(math.pi * min(max(step / max(total_steps, 1), 0.0), 1.0))) / 2
    return float(start + (end - start) * w)


def hyperbolic_project(x: torch.Tensor, K: float) -> torch.Tensor:
    """Project x onto Poincaré ball of curvature K (>0).

    Radius r = 1/sqrt(K). If ||x|| >= r, scale down to lie just inside the ball.
    """
    r = 1.0 / (float(K) ** 0.5)
    xn = x.float()
    n = xn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(n), (r - 1e-6) / n)
    y = xn * scale
    return y.to(dtype=x.dtype)


def percentile_scale(x: torch.Tensor, p: float = 0.999, dim: int | None = None) -> torch.Tensor:
    """Return p-percentile of |x| as a scale; if dim is None, global."""
    xa = x.abs().float()
    q = float(max(0.0, min(1.0, p)))
    if dim is None:
        k = max(int(q * (xa.numel() - 1)), 0)
        val = xa.view(-1).kthvalue(k + 1).values
        return val.to(dtype=x.dtype)
    k = (xa.size(dim) - 1) * q
    k = k.clamp(min=0.0)
    idx = k.long().clamp(max=max(xa.size(dim) - 1, 0))
    sorted_vals, _ = xa.sort(dim=dim)
    take = idx
    while take.ndim < sorted_vals.ndim:
        take = take.unsqueeze(-1)
    out = sorted_vals.gather(dim, take).squeeze(dim)
    return out.to(dtype=x.dtype)


def mse_scale(x: torch.Tensor, bins: int = 2048) -> torch.Tensor:
    """Heuristic MSE scale for quantization (absmax fallback)."""
    return x.abs().amax(dim=tuple(range(x.ndim)), keepdim=False).clamp_min(1e-8)


def range_track(x: torch.Tensor, state: dict | None = None, ema: float = 0.999) -> dict:
    """Track running min/max with EMA. Returns updated state {lo, hi}."""
    xa = x.detach().float()
    lo = xa.amin()
    hi = xa.amax()
    if state is None:
        return {"lo": lo.to(dtype=x.dtype), "hi": hi.to(dtype=x.dtype)}
    new_lo = ema * state["lo"].float() + (1.0 - ema) * lo
    new_hi = ema * state["hi"].float() + (1.0 - ema) * hi
    return {"lo": new_lo.to(dtype=x.dtype), "hi": new_hi.to(dtype=x.dtype)}


def roofline(flops: float, bytes_: float, hw: dict) -> dict:
    """Simple roofline classification.

    hw: {peak_flops: float (FLOP/s), peak_bw: float (bytes/s)}
    Returns: {intensity, ridge_point, bound}
    """
    peak_flops = float(hw.get("peak_flops", 1.0))
    peak_bw = float(hw.get("peak_bw", 1.0))
    intensity = float(flops) / max(float(bytes_), 1e-12)
    ridge = peak_flops / max(peak_bw, 1e-12)
    bound = "compute" if intensity >= ridge else "memory"
    return {"intensity": intensity, "ridge_point": ridge, "bound": bound}


def inclusive_scan(x: torch.Tensor, op: str = "add", axis: int = -1) -> torch.Tensor:
    if op == "add":
        return x.cumsum(dim=axis)
    if op == "max":
        return torch.cummax(x, dim=axis).values
    if op == "logsumexp":
        return logcumsumexp(x, dim=axis)
    raise ValueError(f"unsupported op: {op}")


def exclusive_scan(x: torch.Tensor, op: str = "add", axis: int = -1) -> torch.Tensor:
    if op == "add":
        cs = x.cumsum(dim=axis)
        zeros = torch.zeros_like(x.select(axis, 0))
        return torch.cat([zeros.unsqueeze(axis), cs.narrow(axis, 0, x.size(axis) - 1)], dim=axis)
    if op == "max":
        vals = []
        it = [slice(None)] * x.ndim
        running = None
        for i in range(x.size(axis)):
            it[axis] = slice(0, i)
            seg = x[tuple(it)]
            if i == 0:
                vals.append(x.select(axis, 0).new_full(x.select(axis, 0).shape, torch.finfo(x.dtype).min if x.dtype.is_floating_point else -1e9))
            else:
                vals.append(seg.max(dim=axis).values)
        return torch.stack(vals, dim=axis)
    if op == "logsumexp":
        ne = torch.full_like(x.select(axis, 0), -float("inf"))
        inc = logcumsumexp(x, dim=axis)
        return torch.cat([ne.unsqueeze(axis), inc.narrow(axis, 0, x.size(axis) - 1)], dim=axis)
    raise ValueError(f"unsupported op: {op}")


def fft_conv1d(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """1D convolution via FFT along the last dimension (valid-length output)."""
    n = x.size(-1)
    m = k.size(-1)
    L = int(1 << (n + m - 1).bit_length())
    X = torch.fft.rfft(x.float(), n=L)
    K = torch.fft.rfft(k.float(), n=L)
    Y = X * K
    y = torch.fft.irfft(Y, n=L)
    return y[..., : n + m - 1].to(dtype=x.dtype)


def dct(x: torch.Tensor, type: int = 2, norm: str | None = None, dim: int = -1) -> torch.Tensor:
    """DCT (type-II) fallback implementation."""
    if type != 2:
        raise NotImplementedError("Only DCT type-II supported")
    N = x.size(dim)
    n = torch.arange(N, device=x.device, dtype=x.float().dtype)
    k = torch.arange(N, device=x.device, dtype=x.float().dtype).view(-1, 1)
    M = torch.cos(torch.pi * (n + 0.5) * k / N)
    y = torch.tensordot(M, x.float(), dims=([1], [dim]))
    if norm == "ortho":
        y[0] = y[0] * (1.0 / (N**0.5))
        y[1:] = y[1:] * (2.0 / (N**0.5))
    y = y.movedim(0, dim)
    return y.to(dtype=x.dtype)


def clip_unit_sphere(x: torch.Tensor) -> torch.Tensor:
    n = x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    s = torch.clamp(n, max=1.0)
    x_unit = x.float() / torch.where(n > 1.0, n, torch.ones_like(n))
    return x_unit.to(dtype=x.dtype)


def proj_simplex(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project onto probability simplex along dim using sorting method."""
    x = p.float()
    x_sorted, _ = torch.sort(x, dim=dim, descending=True)
    rho = (x_sorted.cumsum(dim=dim) - 1) / (torch.arange(1, x.size(dim) + 1, device=x.device).view([1] * (x.ndim - 1) + [-1]))
    cond = x_sorted > rho
    k = cond.sum(dim=dim, keepdim=True)
    tau = (x_sorted * cond).sum(dim=dim, keepdim=True) - 1
    tau = tau / k.clamp_min(1)
    return (x - tau).clamp_min(0).to(dtype=p.dtype)


def proj_psd(A: torch.Tensor) -> torch.Tensor:
    """Project symmetric matrix to PSD cone via eigenvalue clamping."""
    Af = A.float()
    S = 0.5 * (Af + Af.transpose(-2, -1))
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = eigvals.clamp_min(0)
    Spsd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
    return Spsd.to(dtype=A.dtype)


def assert_finite(x: torch.Tensor, name: str | None = None) -> None:
    if not torch.isfinite(x).all():
        n = name or "tensor"
        raise AssertionError(f"Non-finite values in {n}: shape={tuple(x.shape)} dtype={x.dtype}")


def nan_guard(x: torch.Tensor, on_error: str = "raise") -> torch.Tensor | None:
    ok = torch.isfinite(x).all()
    if ok:
        return None
    if on_error == "raise":
        raise RuntimeError("NaN/Inf detected")
    return (~torch.isfinite(x))


def accum_steps(global_bs: int, device_bs: int) -> int:
    import math
    return int(math.ceil(max(1, int(global_bs)) / max(1, int(device_bs))))


def microbatch_plan(T: int, memory_bytes: int, bytes_per_token: int = 1024) -> list[tuple[int, int]]:
    """Plan microbatches along time dimension given a crude memory model.

    Returns list of (start, end) indices covering [0, T).
    """
    if memory_bytes <= 0:
        return [(0, int(T))]
    import math
    tokens_per_batch = max(1, int(memory_bytes // max(1, int(bytes_per_token))))
    plan = []
    s = 0
    while s < T:
        e = min(T, s + tokens_per_batch)
        plan.append((s, e))
        s = e
    return plan


def bilinear_discretize(A: torch.Tensor, B: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Tustin (bilinear) discretization for continuous-time (A,B)."""
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    M = I - (0.5 * float(dt)) * A
    N = I + (0.5 * float(dt)) * A
    try:
        M_inv = torch.linalg.inv(M.float())
    except Exception:
        M_inv = torch.linalg.pinv(M.float())
    Ad = (M_inv @ N.float()).to(dtype=A.dtype)
    Bd = (M_inv @ (float(dt) * B.float())).to(dtype=B.dtype)
    return Ad, Bd


def ssm_step(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, dt: float) -> torch.Tensor:
    """Single step of discretized SSM: y = C x' + D u, x' = Ad x + Bd u."""
    Ad, Bd = bilinear_discretize(A, B, dt)
    u = x
    x_next = Ad @ x + Bd @ u
    y = C @ x_next + D @ u
    return y


def power_spectrum(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, dt: float, n: int) -> torch.Tensor:
    """Estimate power spectrum by simulating impulse response of length n and FFT."""
    Ad, Bd = bilinear_discretize(A, B, dt)
    x = torch.zeros(A.shape[-1], device=A.device, dtype=A.dtype)
    u = torch.zeros_like(x)
    # unit impulse on first input dim if available
    h = []
    for t in range(int(n)):
        if t == 0:
            u = torch.zeros_like(x)
            u[: B.shape[-1] if B.ndim == 2 else 1] = 1.0
        x = Ad @ x + (Bd @ u)
        y = C @ x
        h.append(y)
        u.zero_()
    h = torch.stack(h, dim=0).float()
    H = torch.fft.rfft(h, dim=0)
    P = (H.conj() * H).real
    return P.to(dtype=A.dtype)


from .shape import segment_max, segment_sum


def segment_softmax(x: torch.Tensor, segments: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Softmax per segment id along the 0th dimension; operates over last dim
    x_max = segment_max(x.max(dim=dim, keepdim=True).values.squeeze(dim), segments).unsqueeze(-1)
    exps = torch.exp((x - x_max[segments]).float())
    denom = segment_sum(exps.sum(dim=dim, keepdim=True).squeeze(dim), segments).unsqueeze(-1)
    out = (exps / denom[segments]).to(dtype=x.dtype)
    return out


def segment_norm(x: torch.Tensor, segments: torch.Tensor, p: int = 2) -> torch.Tensor:
    num = segment_sum(x.abs().pow(p).sum(dim=tuple(range(1, x.ndim))), segments)
    return num.pow(1.0 / p)


def segment_logsumexp(x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """Stable logsumexp per segment over the 0th dimension."""
    m = segment_max(x, segments)
    y = torch.exp((x - m[segments]).float())
    s = segment_sum(y, segments)
    return (m + torch.log(s.clamp_min(1e-45))).to(dtype=x.dtype)


def softmax_zloss(logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1):
    """Return (probs, z) where probs = softmax(logits), z = logsumexp(logits).

    Caller can apply z-loss coefficient externally.
    """
    x = logits.float()
    if mask is not None:
        x = x.masked_fill(mask, torch.finfo(x.dtype).min)
    x_max = x.max(dim=dim, keepdim=True).values
    logZ = x_max + torch.log(torch.exp(x - x_max).sum(dim=dim, keepdim=True))
    probs = torch.exp(x - logZ)
    return probs.to(dtype=logits.dtype), logZ.squeeze(dim).to(dtype=logits.dtype)



# ---- Additions: stable cumprod, idct, hilbert, generic spectrum, SSM utils, stability, scheduling, memory, profiling ----
def stable_cumprod(x: torch.Tensor, axis: int = -1, logspace: bool = True) -> torch.Tensor:
    if logspace:
        return torch.exp(inclusive_scan(torch.log(x.clamp_min(1e-45)), op="add", axis=axis)).to(dtype=x.dtype)
    return x.cumprod(dim=axis)


def idct(X: torch.Tensor, type: int = 2, norm: str | None = None, dim: int = -1) -> torch.Tensor:
    if type != 2:
        raise NotImplementedError("Only IDCT for type-II supported")
    N = X.size(dim)
    k = torch.arange(N, device=X.device, dtype=X.float().dtype)
    n = torch.arange(N, device=X.device, dtype=X.float().dtype).view(-1, 1)
    M = torch.cos(torch.pi * (n + 0.5) * k / N)
    Y = X.float().movedim(dim, -1)
    y = torch.tensordot(Y, M, dims=([-1], [0]))
    if norm == "ortho":
        y[..., 0] = y[..., 0] * (1.0 / (N**0.5))
        y[..., 1:] = y[..., 1:] * (2.0 / (N**0.5))
    return y.movedim(-1, dim).to(dtype=X.dtype)


def hilbert_transform(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    d = axis if axis >= 0 else x.ndim + axis
    X = torch.fft.rfft(x.float(), dim=d)
    N = x.size(d)
    h = torch.zeros(X.shape[-1], device=x.device, dtype=X.dtype)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    Y = X * h
    y = torch.fft.irfft(Y, n=N, dim=d)
    return y.to(dtype=x.dtype)


def power_spectrum_generic(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    X = torch.fft.rfft(x.float(), dim=axis)
    P = (X.conj() * X).real
    return P.to(dtype=x.dtype)


def zoh_discretize(A: torch.Tensor, B: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    Af = A.float()
    Bf = B.float()
    n = Af.shape[-1]
    M = torch.zeros(n + Bf.shape[-1], n + Bf.shape[-1], device=A.device, dtype=Af.dtype)
    M[:n, :n] = Af
    M[:n, n:] = Bf
    Md = torch.linalg.matrix_exp(M * float(dt))
    Ad = Md[:n, :n].to(dtype=A.dtype)
    Bd = Md[:n, n:].to(dtype=B.dtype)
    return Ad, Bd


def ssm_stability_margin(A: torch.Tensor) -> torch.Tensor:
    eig = torch.linalg.eigvals(A.float())
    max_real = eig.real.max()
    return (-max_real).to(dtype=A.dtype)


def pairwise_sum(x: torch.Tensor, axis: int | None = None) -> torch.Tensor:
    xf = x.float()
    if axis is None:
        xf = xf.view(-1)
        while xf.numel() > 1:
            if xf.numel() % 2 == 1:
                xf = torch.cat([xf, torch.zeros(1, device=xf.device)])
            xf = xf.view(-1, 2).sum(dim=-1)
        return xf.view(()).to(dtype=x.dtype)
    t = xf
    while t.size(axis) > 1:
        n = t.size(axis)
        if n % 2 == 1:
            pad_shape = list(t.shape)
            pad_shape[axis] = 1
            t = torch.cat([t, t.new_zeros(pad_shape)], dim=axis)
        t = t.view(*t.shape[:axis], t.size(axis) // 2, 2, *t.shape[axis + 1:]).sum(dim=axis + 1)
    return t.squeeze(axis).to(dtype=x.dtype)


def stable_norm(x: torch.Tensor, ord: int | float = 2, axis: int = -1, eps: float = 1e-12) -> torch.Tensor:
    xf = x.float().abs().pow(float(ord)).sum(dim=axis, keepdim=True)
    return (xf + eps).pow(1.0 / float(ord)).to(dtype=x.dtype)


def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    yf = y.float()
    return (yf + torch.log(-torch.expm1(-yf).clamp_min(1e-45))).to(dtype=y.dtype)


def bucket_sizes(N: int, max_bytes: int, item_bytes: int = 1) -> list[int]:
    per = max(1, int(max_bytes // max(1, int(item_bytes))))
    out = []
    s = 0
    while s < N:
        e = min(N, s + per)
        out.append(e - s)
        s = e
    return out


def tensor_shards(shape: tuple[int, ...], bytes_budget: int, prefer: tuple[str, ...] = ("T", "B"), dtype: torch.dtype = torch.float32) -> dict:
    from .shard import tensor_bytes
    size = tensor_bytes(shape, dtype)
    if size <= bytes_budget:
        return {"axis": len(shape) - 2 if len(shape) >= 2 else 0, "parts": 1}
    axis = len(shape) - 2 if len(shape) >= 2 else 0
    k = int((size + bytes_budget - 1) // max(bytes_budget, 1))
    return {"axis": axis, "parts": max(1, k)}


def estimate_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    from .shard import tensor_bytes
    return tensor_bytes(shape, dtype)


def activation_bytes(shapes: list[tuple[int, ...]], dtypes: list[torch.dtype]) -> int:
    from .shard import tensor_bytes
    total = 0
    for s, dt in zip(shapes, dtypes):
        total += tensor_bytes(s, dt)
    return total


def flops_linear(in_dim: int, out_dim: int, bias: bool = True) -> int:
    return int(in_dim) * int(out_dim) * 2 + (int(out_dim) if bias else 0)


def flops_conv1d(L: int, Cin: int, Cout: int, K: int, stride: int = 1, bias: bool = True) -> int:
    Lout = (L - K) // stride + 1
    return Lout * Cout * Cin * K * 2 + (Lout * Cout if bias else 0)


def flops_conv2d(H: int, W: int, Cin: int, Cout: int, Kh: int, Kw: int, stride: tuple[int, int] = (1, 1), bias: bool = True) -> int:
    Ho = (H - Kh) // stride[0] + 1
    Wo = (W - Kw) // stride[1] + 1
    return Ho * Wo * Cout * Cin * Kh * Kw * 2 + (Ho * Wo * Cout if bias else 0)


def params_count_linear(in_dim: int, out_dim: int, bias: bool = True) -> int:
    return in_dim * out_dim + (out_dim if bias else 0)


def time_op(fn, *args, warmup: int = 2, repeat: int = 10, **kwargs) -> dict:
    import time
    try:
        import torch.cuda as cuda
        sync = cuda.synchronize
    except Exception:
        def sync():
            return None
    for _ in range(max(0, int(warmup))):
        _ = fn(*args, **kwargs)
    times = []
    for _ in range(max(1, int(repeat))):
        sync()
        t0 = time.time()
        _ = fn(*args, **kwargs)
        sync()
        times.append(time.time() - t0)
    import statistics as st
    return {"mean": st.mean(times), "std": st.pstdev(times), "p50": st.median(times), "p95": sorted(times)[int(0.95 * (len(times) - 1))]}


def ema_range_track(x: torch.Tensor, state: dict | None = None, beta: float = 0.999) -> dict:
    return range_track(x, state=state, ema=float(beta))
