from __future__ import annotations

import torch

from runtime.native import has_native_op, native_module


def _to_tuple_dims(dim: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if isinstance(dim, tuple):
        dims = dim
    else:
        dims = (dim,)
    out: list[int] = []
    for d in dims:
        out.append(d if d >= 0 else ndim + d)
    return tuple(out)


def _reshape_param_for_dims(
    param: torch.Tensor | None, x: torch.Tensor, dims: int | tuple[int, ...]
) -> torch.Tensor | None:
    if param is None:
        return None
    dims_t = _to_tuple_dims(dims, x.ndim)
    target_shape = [1] * x.ndim
    for d in dims_t:
        target_shape[d] = x.size(d)
    p = param.to(dtype=x.dtype, device=x.device)
    if p.ndim == len(dims_t) and list(p.shape) == [x.size(d) for d in dims_t]:
        return p.view(*target_shape)
    prod = 1
    for d in dims_t:
        prod *= x.size(d)
    if p.ndim == 1 and p.numel() == prod:
        return p.view(*target_shape)
    if p.ndim == x.ndim:
        return p
    if len(dims_t) == 1 and p.ndim == 1 and p.shape[0] == x.size(dims_t[0]):
        return p.view(*target_shape)
    return p


def _rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    dim: int | tuple[int, ...] = -1,
) -> torch.Tensor:
    dims = _to_tuple_dims(dim, x.ndim)
    mean_sq = (x.float() * x.float()).mean(dim=dims, keepdim=True)
    y = x * torch.rsqrt(mean_sq + eps)
    if weight is not None:
        y = y * _reshape_param_for_dims(weight, x, dims)
    return y.to(dtype=x.dtype)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    dim: int | tuple[int, ...] = -1,
) -> torch.Tensor:
    if dim == -1 and has_native_op("rms_norm"):
        module = native_module()
        if module is not None and hasattr(module, "rms_norm_forward"):
            return module.rms_norm_forward(x, weight, eps)
    return _rms_norm_reference(x, weight=weight, eps=eps, dim=dim)
