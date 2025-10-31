from typing import Iterable, Callable, Optional, Sequence, Tuple, Union
import torch


def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    params = list(parameters)
    device = params[0].device if params else torch.device("cpu")
    total = torch.tensor(0.0, device=device)
    for p in params:
        if p.grad is not None:
            total = total + p.grad.detach().float().pow(2).sum()
    return total.sqrt()


# Public alias for parameter-wise grad norm
def grad_norm_parameters(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    return global_grad_norm(parameters)


def clip_grad_norm_(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> float:
    params = list(parameters)
    norm = global_grad_norm(params)
    scale = (max_norm / (norm + eps)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale)
    return float(norm.item())


def assert_no_nan_grad(module: torch.nn.Module):
    for name, p in module.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise AssertionError(f"Non-finite gradient detected in {name}")


def loss_scaler_step_safe(loss: torch.Tensor, scale: float) -> torch.Tensor:
    """Return scaled loss for backprop; stateless helper."""
    return loss * float(scale)


# -------------------------
# Additions: Norms & Clipping
# -------------------------

def _unitwise_reduce_dims(t: torch.Tensor) -> Tuple[int, ...]:
    if t.ndim <= 1:
        return ()
    return tuple(range(1, t.ndim))


def unitwise_l2_norm(param: torch.nn.Parameter, grad: torch.Tensor) -> torch.Tensor:
    """Per-unit L2 norm of grad following common layer grouping rules.

    Returns a tensor broadcastable to grad with shape keeping the unit axis.
    """
    if grad is None:
        return torch.tensor(0.0, device=param.device)
    reduce_dims = _unitwise_reduce_dims(param)
    if reduce_dims == ():
        return grad.detach().float().abs().sqrt().pow(2.0)  # degenerate; avoid division by zero later
    return grad.detach().float().pow(2).sum(dim=reduce_dims, keepdim=True).sqrt()


def unitwise_clip_(param: torch.nn.Parameter, grad: Optional[torch.Tensor], max_norm: float, eps: float = 1e-12) -> None:
    """In-place per-unit clipping of grad to max_norm (AGC-compatible form)."""
    if grad is None:
        return
    norms = unitwise_l2_norm(param, grad)
    scale = (float(max_norm) / (norms + eps)).clamp(max=1.0)
    grad.mul_(scale)


def clip_grad_value_(parameters: Iterable[torch.nn.Parameter], clip_value: float) -> None:
    params = list(parameters)
    if clip_value <= 0:
        return
    for p in params:
        if p.grad is not None:
            p.grad.data.clamp_(-float(clip_value), float(clip_value))


def global_grad_norm_fp32(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    params = list(parameters)
    device = params[0].device if params else torch.device("cpu")
    total = torch.tensor(0.0, device=device, dtype=torch.float32)
    for p in params:
        if p.grad is not None:
            total = total + p.grad.detach().to(dtype=torch.float32).pow(2).sum()
    return total.sqrt()


def clip_grad_norm_masked_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    mask_fn: Callable[[torch.nn.Parameter], bool],
    eps: float = 1e-6,
) -> float:
    """Clip only parameters for which mask_fn(param) is False; exclude when True."""
    params = list(parameters)
    selected: list[torch.nn.Parameter] = [p for p in params if not mask_fn(p)]
    norm = global_grad_norm(selected)
    scale = (max_norm / (norm + eps)).clamp(max=1.0)
    for p in selected:
        if p.grad is not None:
            p.grad.mul_(scale)
    return float(norm.item())


# -------------------------
# Additions: Decoupled Weight Decay & Masks
# -------------------------

def decoupled_weight_decay_(param: torch.nn.Parameter, lr: float, wd: float) -> None:
    if wd == 0.0 or lr == 0.0:
        return
    param.data.mul_(1.0 - float(lr) * float(wd))


def decay_mask_from_names(
    names: Sequence[str],
    patterns: Sequence[str] = ("bias", "norm", "ln", "bn"),
) -> list[bool]:
    """Return boolean mask to skip decay (True means skip) based on name substrings."""
    lowered = [n.lower() for n in names]
    pats = [p.lower() for p in patterns]
    mask: list[bool] = []
    for n in lowered:
        skip = any(p in n for p in pats)
        mask.append(skip)
    return mask


def apply_weight_decay_masked_(param: torch.nn.Parameter, lr: float, wd: float, mask: torch.Tensor) -> None:
    if wd == 0.0 or lr == 0.0:
        return
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if mask.shape != param.data.shape:
        raise ValueError("mask must have the same shape as param")
    decay = 1.0 - float(lr) * float(wd)
    param.data[mask] *= decay


# -------------------------
# Additions: Gradient Conditioning
# -------------------------

def zero_nan_inf_grad_(parameters: Iterable[torch.nn.Parameter], fill: float = 0.0) -> None:
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.data
        finite = torch.isfinite(g)
        if finite.all():
            continue
        g[~finite] = float(fill)


def gradient_centralization_(param_or_grad: Union[torch.nn.Parameter, torch.Tensor], dims: Union[int, Tuple[int, ...]]) -> None:
    t = param_or_grad.data if isinstance(param_or_grad, torch.nn.Parameter) else param_or_grad
    mean = t.mean(dim=dims, keepdim=True)
    t.sub_(mean)


def project_grad_orthogonal_(grad: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12) -> None:
    if ref is None:
        return
    num = (grad * ref).sum()
    denom = ref.float().pow(2).sum().add_(float(eps))
    scale = num / denom
    grad.add_(ref, alpha=-scale)


def add_grad_noise_(grad: torch.Tensor, std: float, rng: Optional[torch.Generator] = None) -> None:
    if std == 0.0:
        return
    noise = torch.randn_like(grad, generator=rng) * float(std)
    grad.add_(noise)


# -------------------------
# Additions: EMA / SWA
# -------------------------

def ema_update_(shadow: torch.Tensor, src: torch.Tensor, decay: float) -> None:
    shadow.mul_(float(decay)).add_(src, alpha=1.0 - float(decay))


def ema_compute_decay(horizon_steps: float, dt: float = 1.0) -> float:
    if horizon_steps <= 0:
        return 0.0
    return float(torch.exp(torch.tensor(-dt / horizon_steps)).item())


def swa_merge_(target: torch.Tensor, model: torch.Tensor, n_models: int) -> None:
    if n_models <= 0:
        return
    alpha = 1.0 / float(n_models)
    target.mul_(1.0 - alpha).add_(model, alpha=alpha)


# -------------------------
# Additions: SAM primitives
# -------------------------

def sam_perturbation_(
    params: Iterable[torch.nn.Parameter],
    grads: Optional[Iterable[Optional[torch.Tensor]]],
    rho: float,
    eps: float = 1e-12,
) -> None:
    plist = list(params)
    glist = list(grads) if grads is not None else [p.grad for p in plist]
    device = plist[0].device if plist else torch.device("cpu")
    total_sq = torch.tensor(0.0, device=device, dtype=torch.float32)
    for g in glist:
        if g is None:
            continue
        total_sq = total_sq + g.detach().to(torch.float32).pow(2).sum()
    norm = total_sq.sqrt()
    scale = float(rho) / (float(norm.item()) + float(eps)) if norm.numel() > 0 else 0.0
    if scale == 0.0:
        return
    for p, g in zip(plist, glist):
        if g is None:
            continue
        p.data.add_(g, alpha=scale)


def sam_restore_(params: Iterable[torch.nn.Parameter], backup: Sequence[torch.Tensor]) -> None:
    for p, b in zip(params, backup):
        p.data.copy_(b)


# -------------------------
# Additions: Schedules (pure scalars)
# -------------------------

def schedule_linear_with_warmup(step: int, warmup: int, total: int) -> float:
    if total <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return float(step) / float(max(1, warmup))
    decay_steps = max(1, total - max(0, warmup))
    progress = (min(step, total) - max(0, warmup)) / float(decay_steps)
    return float(max(0.0, 1.0 - progress))


def schedule_cosine_with_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.0) -> float:
    if total <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return float(step) / float(max(1, warmup))
    t = min(step, total) - max(0, warmup)
    T = max(1, total - max(0, warmup))
    cos_decay = 0.5 * (1.0 + float(torch.cos(torch.tensor(torch.pi * t / T)).item()))
    return float(min_ratio + (1.0 - min_ratio) * cos_decay)


def schedule_poly(step: int, total: int, power: float = 1.0, min_ratio: float = 0.0) -> float:
    if total <= 0:
        return 1.0
    step_clamped = min(max(step, 0), total)
    ratio = 1.0 - (step_clamped / float(total))
    val = ratio ** float(power)
    return float(max(min_ratio, val))


def schedule_piecewise(step: int, boundaries: Sequence[int], values: Sequence[float]) -> float:
    if len(values) != len(boundaries) + 1:
        raise ValueError("values must be len(boundaries)+1")
    for b, v in zip(boundaries, values):
        if step < b:
            return float(v)
    return float(values[-1])


# -------------------------
# Additions: Distributed-friendly reducers
# -------------------------

def reduce_grad_norm(norm_sq_local: torch.Tensor, allreduce_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    total_sq = allreduce_fn(norm_sq_local)
    return total_sq.sqrt()


def bucketed_grad_norm(parameters: Iterable[torch.nn.Parameter], bucket_bytes: int = 64 << 20) -> torch.Tensor:
    params = [p for p in parameters if p.grad is not None]
    device = params[0].device if params else torch.device("cpu")
    total_sq = torch.tensor(0.0, device=device, dtype=torch.float32)
    bytes_acc = 0
    bucket_sq = torch.tensor(0.0, device=device, dtype=torch.float32)
    for p in params:
        g = p.grad.detach()
        bucket_sq = bucket_sq + g.to(torch.float32).pow(2).sum()
        bytes_acc += g.numel() * g.element_size()
        if bytes_acc >= bucket_bytes:
            total_sq = total_sq + bucket_sq
            bucket_sq = torch.tensor(0.0, device=device, dtype=torch.float32)
            bytes_acc = 0
    total_sq = total_sq + bucket_sq
    return total_sq.sqrt()


# -------------------------
# Additions: Reporting & Assertions
# -------------------------

def grad_norm_report(
    parameters: Iterable[Union[torch.nn.Parameter, Tuple[str, torch.nn.Parameter], None]]
) -> dict:
    report: dict[str, dict] = {}
    idx = 0
    for item in parameters:
        if item is None:
            continue
        if isinstance(item, tuple):
            name, p = item
        else:
            name, p = f"param_{idx}", item
        idx += 1
        if not isinstance(p, torch.nn.Parameter) or p.grad is None:
            report[name] = {"norm": 0.0, "max": 0.0, "finite": True}
            continue
        g = p.grad.detach()
        norm = g.float().pow(2).sum().sqrt()
        gmax = g.detach().abs().max()
        finite = bool(torch.isfinite(g).all().item())
        report[name] = {"norm": float(norm.item()), "max": float(gmax.item()), "finite": finite}
    return report


def assert_global_norm_below(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    norm = global_grad_norm(parameters)
    if float(norm.item()) > float(max_norm):
        raise AssertionError(f"Global grad norm {float(norm.item()):.6f} exceeds {float(max_norm):.6f}")


# -------------------------
# Stateless update kernels (fp32 accum, decoupled wd)
# -------------------------

def _to_fp32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32)


def adamw_update_(
    p: torch.nn.Parameter,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    decoupled: bool = True,
    bias_correction: bool = True,
    maximize: bool = False,
) -> None:
    grad = -g if maximize else g
    m_fp32 = _to_fp32(m)
    v_fp32 = _to_fp32(v)
    grad_fp32 = _to_fp32(grad)

    m_fp32.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
    v_fp32.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

    if bias_correction:
        # Caller must pass step-appropriate betas if doing step-wise correction externally.
        m_hat = m_fp32 / (1.0 - beta1)
        v_hat = v_fp32 / (1.0 - beta2)
    else:
        m_hat = m_fp32
        v_hat = v_fp32

    denom = v_hat.sqrt().add_(float(eps))
    update = m_hat / denom

    if decoupled and weight_decay != 0.0:
        p.data.mul_(1.0 - float(lr) * float(weight_decay))
    elif not decoupled and weight_decay != 0.0:
        update = update.add(p.data, alpha=float(weight_decay))

    p.data.add_(update, alpha=-float(lr))

    # Write back moments with original dtype
    m.copy_(m_fp32.to(m.dtype))
    v.copy_(v_fp32.to(v.dtype))


def lamb_update_(
    p: torch.nn.Parameter,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    trust_clip: Tuple[float, float] = (0.0, 10.0),
) -> float:
    # Adam-style moments
    m_fp32 = _to_fp32(m)
    v_fp32 = _to_fp32(v)
    grad_fp32 = _to_fp32(g)
    m_fp32.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
    v_fp32.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
    m_hat = m_fp32 / (1.0 - beta1)
    v_hat = v_fp32 / (1.0 - beta2)
    adam_step = m_hat / (v_hat.sqrt().add_(float(eps)))

    # Decoupled wd
    if weight_decay != 0.0:
        p.data.mul_(1.0 - float(lr) * float(weight_decay))

    # Trust ratio
    w_norm = p.data.detach().to(torch.float32).norm(p=2)
    u_norm = adam_step.detach().norm(p=2)
    trust_ratio = (w_norm / (u_norm + 1e-12)).item() if u_norm > 0 else 1.0
    trust_ratio = float(max(trust_clip[0], min(trust_ratio, trust_clip[1])))

    p.data.add_(adam_step, alpha=-float(lr) * trust_ratio)

    m.copy_(m_fp32.to(m.dtype))
    v.copy_(v_fp32.to(v.dtype))
    return trust_ratio


def lion_update_(
    p: torch.nn.Parameter,
    g: torch.Tensor,
    m: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
) -> None:
    # LION: m_t = beta1*m + (1-beta1)*g; update = sign(m_t)
    m_fp32 = _to_fp32(m)
    grad_fp32 = _to_fp32(g)
    m_fp32.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
    update = m_fp32.sign()
    # decoupled wd
    if weight_decay != 0.0:
        p.data.mul_(1.0 - float(lr) * float(weight_decay))
    # second momentum scaling beta2
    p.data.add_(update, alpha=-float(lr) * float(beta2))
    m.copy_(m_fp32.to(m.dtype))


def adafactor_update_(
    p: torch.nn.Parameter,
    g: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    *,
    lr: float,
    beta2: float,
    eps1: float,
    eps2: float,
    weight_decay: float,
    factored: bool = True,
    clip_threshold: float = 1.0,
) -> None:
    grad = _to_fp32(g)
    if factored and grad.ndim >= 2:
        # Row/col averages of squared grad
        # r: shape (..., N) across last dim; c: shape (M, ...) across first dim
        r_data = _to_fp32(r)
        c_data = _to_fp32(c)
        r_update = grad.pow(2).mean(dim=-1)
        c_update = grad.pow(2).mean(dim=-2)
        r_data.mul_(beta2).add_(r_update, alpha=1.0 - beta2)
        c_data.mul_(beta2).add_(c_update, alpha=1.0 - beta2)
        # v_hat approx via outer(r, c) normalized
        v_hat = torch.einsum("...i,i...->...i...", r_data, c_data)
        v_hat = v_hat / (c_data.mean() + float(eps1))
        denom = v_hat.sqrt().add_(float(eps2))
        update = grad / denom
        r.copy_(r_data.to(r.dtype))
        c.copy_(c_data.to(c.dtype))
    else:
        # Unfactored path using c as accumulator for second moment
        v = _to_fp32(c)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        update = grad / (v.sqrt().add_(float(eps2)))
        c.copy_(v.to(c.dtype))

    rms = grad.pow(2).mean().sqrt()
    clip = float(max(1.0, (rms / max(clip_threshold, 1e-12)).item()))
    update = update / clip

    if weight_decay != 0.0:
        p.data.mul_(1.0 - float(lr) * float(weight_decay))
    p.data.add_(update.to(p.dtype), alpha=-float(lr))


# -------------------------
# Unified global-norm policy
# -------------------------

def clip_by_policy_(
    params: Iterable[torch.nn.Parameter],
    *,
    max_norm: Optional[float] = None,
    value: Optional[float] = None,
    unitwise_norm: Optional[Union[bool, float]] = None,
    mask: Optional[Callable[[torch.nn.Parameter], bool]] = None,
    percentile: Optional[float] = None,
    eps: float = 1e-6,
) -> Optional[float]:
    """Apply one of several mutually-exclusive clipping policies.

    Returns the global norm if max_norm policy was used, else None.
    """
    strategies = [x is not None and x != 0 for x in (max_norm, value, unitwise_norm, percentile)]
    if sum(bool(s) for s in strategies) > 1:
        raise ValueError("Only one of max_norm, value, unitwise_norm, percentile may be set")

    selected = list(params)
    if mask is not None:
        selected = [p for p in selected if not mask(p)]

    if max_norm is not None:
        return clip_grad_norm_masked_(selected, float(max_norm), lambda _: False, eps=eps)

    if value is not None:
        clip_grad_value_(selected, float(value))
        return None

    if unitwise_norm is not None:
        max_u = float(unitwise_norm if isinstance(unitwise_norm, (int, float)) else 1.0)
        for p in selected:
            if p.grad is not None:
                unitwise_clip_(p, p.grad, max_u, eps=eps)
        return None

    if percentile is not None:
        perc = float(percentile)
        if not (0.0 < perc < 100.0):
            raise ValueError("percentile must be in (0,100)")
        vals = []
        for p in selected:
            if p.grad is not None:
                vals.append(p.grad.detach().abs().reshape(-1))
        if not vals:
            return None
        all_abs = torch.cat(vals).to(torch.float32)
        k = int(max(1, round(len(all_abs) * perc / 100.0)))
        thresh = all_abs.kthvalue(k).values
        for p in selected:
            if p.grad is not None:
                p.grad.data.clamp_(-thresh.to(p.grad.dtype), thresh.to(p.grad.dtype))
        return None

    return None


# -------------------------
# Overflow-aware loss scaling (stateless)
# -------------------------

def loss_scale_update_(
    scale: Union[float, torch.Tensor],
    found_inf: Union[bool, torch.Tensor],
    *,
    growth: float = 2.0,
    backoff: float = 0.5,
    growth_interval: int = 2000,
) -> float:
    s = float(scale if not isinstance(scale, torch.Tensor) else scale.item())
    inf = bool(found_inf if not isinstance(found_inf, torch.Tensor) else bool(found_inf.item()))
    if inf:
        return max(1.0, s * float(backoff))
    # continuous growth matching growth every growth_interval steps on average
    per_step = float(growth) ** (1.0 / max(1, int(growth_interval)))
    return s * per_step


def unscale_grads_(params: Iterable[torch.nn.Parameter], scale: Union[float, torch.Tensor]) -> None:
    s = float(scale if not isinstance(scale, torch.Tensor) else scale.item())
    inv = 1.0 / max(s, 1e-12)
    for p in params:
        if p.grad is not None:
            p.grad.data.mul_(inv)


def detect_overflow(params: Iterable[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


# -------------------------
# Per-param decay router (names/modules heuristics)
# -------------------------

def decay_mask_from_params(
    params: Iterable[Union[torch.nn.Parameter, Tuple[str, torch.nn.Parameter]]],
    *,
    bias_names: Tuple[str, ...] = ("bias",),
    norm_modules: Tuple[str, ...] = ("layernorm", "rmsnorm", "batchnorm", "instancenorm"),
    embed_modules: Tuple[str, ...] = ("embedding",),
) -> list[bool]:
    mask: list[bool] = []
    for idx, item in enumerate(params):
        if isinstance(item, tuple):
            name, p = item
            lname = name.lower()
        else:
            name, p = f"param_{idx}", item
            lname = name.lower()
        skip = False
        if any(b in lname for b in bias_names):
            skip = True
        if any(nm in lname for nm in norm_modules):
            skip = True
        if any(em in lname for em in embed_modules):
            skip = True
        if p.ndim == 1:  # common for norm scales and bias vectors
            skip = True
        mask.append(skip)
    return mask


def apply_weight_decay_routed_(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    wd: float,
    mask: Sequence[bool],
) -> None:
    for p, skip in zip(params, mask):
        if skip or wd == 0.0 or lr == 0.0:
            continue
        p.data.mul_(1.0 - float(lr) * float(wd))


# -------------------------
# SAM correctness & curvature-aware variants
# -------------------------

def sam_compute_rho_(
    params: Iterable[torch.nn.Parameter],
    grads: Iterable[Optional[torch.Tensor]],
    *,
    rho: float,
    eps: float = 1e-12,
) -> None:
    plist = list(params)
    glist = list(grads)
    device = plist[0].device if plist else torch.device("cpu")
    total_sq = torch.tensor(0.0, device=device, dtype=torch.float32)
    for g in glist:
        if g is not None:
            total_sq = total_sq + g.detach().to(torch.float32).pow(2).sum()
    denom = total_sq.sqrt().add_(float(eps))
    scale = float(rho) / float(denom.item()) if denom.item() > 0 else 0.0
    if scale == 0.0:
        return
    for p, g in zip(plist, glist):
        if g is None:
            continue
        p.data.add_(g, alpha=scale)


def asam_scale_(
    params: Iterable[torch.nn.Parameter],
    grads: Iterable[Optional[torch.Tensor]],
    *,
    rho: float,
    eps: float = 1e-12,
) -> None:
    # Scale gradients by |w|+eps elementwise, then normalize globally
    plist = list(params)
    glist = list(grads)
    device = plist[0].device if plist else torch.device("cpu")
    total_sq = torch.tensor(0.0, device=device, dtype=torch.float32)
    scaled: list[Optional[torch.Tensor]] = []
    for p, g in zip(plist, glist):
        if g is None:
            scaled.append(None)
            continue
        s = g.detach() * (p.data.detach().abs() + float(eps))
        scaled.append(s)
        total_sq = total_sq + s.to(torch.float32).pow(2).sum()
    denom = total_sq.sqrt().add_(float(eps))
    scale = float(rho) / float(denom.item()) if denom.item() > 0 else 0.0
    if scale == 0.0:
        return
    for p, s in zip(plist, scaled):
        if s is None:
            continue
        p.data.add_(s, alpha=scale)


def sam_should_skip(found_inf: Union[bool, torch.Tensor]) -> bool:
    return bool(found_inf if not isinstance(found_inf, torch.Tensor) else bool(found_inf.item()))


# -------------------------
# EMA/SWA numerics
# -------------------------

def ema_update_bc_(ema: torch.Tensor, p: torch.Tensor, *, decay: float, step: int) -> float:
    ema.mul_(float(decay)).add_(p, alpha=1.0 - float(decay))
    # bias-corrected readout factor to obtain unbiased estimate: ema / (1 - decay**step)
    return float(1.0 - (float(decay) ** max(1, int(step))))


def swa_collect_(buffers: Sequence[torch.Tensor], params: Sequence[torch.Tensor]) -> None:
    for buf, p in zip(buffers, params):
        buf.add_(p)


def swa_finalize_(params: Sequence[torch.Tensor], cnt: int) -> None:
    if cnt <= 0:
        return
    inv = 1.0 / float(cnt)
    for p in params:
        p.mul_(inv)


# -------------------------
# New schedules (pure scalar math)
# -------------------------

def schedule_cosine_restart(t: int, t0: int, period: int, warmup: int) -> float:
    if t < warmup and warmup > 0:
        return float(t) / float(max(1, warmup))
    if period <= 0:
        return 0.0
    tau = max(0, t - max(0, warmup) - t0)
    mod = tau % period
    cos_decay = 0.5 * (1.0 + float(torch.cos(torch.tensor(torch.pi * mod / period)).item()))
    return float(cos_decay)


def schedule_linear_floor(t: int, warmup: int, floor: float) -> float:
    if warmup > 0 and t < warmup:
        return float(t) / float(max(1, warmup))
    return float(max(floor, 1.0 - max(0, t - max(0, warmup))))

