import torch


def residual_add(x: torch.Tensor, residual: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    if alpha != 1.0:
        return x + residual * alpha
    return x + residual


def gated_residual_add(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x + gate * y


def residual_bias_dropout_add(x: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor | None, p: float = 0.0, training: bool = False) -> torch.Tensor:
    if bias is not None:
        x = x + bias
    if p > 0.0 and training:
        mask = (torch.rand_like(x) > p).to(x.dtype) / (1.0 - p)
        x = x * mask
    return x + residual


def prenorm(x: torch.Tensor, norm_fn, fn):
    return x + fn(norm_fn(x))


def postnorm(x: torch.Tensor, norm_fn, fn):
    y = fn(x)
    return norm_fn(y)


def residual_alpha_schedule(step: int, total_steps: int, start: float = 1.0, end: float = 1.0, warmup_ratio: float = 0.0) -> float:
    warm = int(total_steps * warmup_ratio)
    if step <= warm:
        return float(start * (step / max(warm, 1)))
    t = (step - warm) / max(total_steps - warm, 1)
    return float(start + (end - start) * t)


