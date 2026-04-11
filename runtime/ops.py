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


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if has_native_op("rope"):
        module = native_module()
        if module is not None and hasattr(module, "apply_rotary_forward"):
            q_out, k_out = module.apply_rotary_forward(q, k, cos, sin)
            return q_out, k_out

    cos_b = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sin_b = sin.view(1, 1, sin.shape[0], sin.shape[1])

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos_b) + (rotate_half(q) * sin_b), (k * cos_b) + (rotate_half(k) * sin_b)


def kv_cache_append(
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if has_native_op("kv_cache_append"):
        module = native_module()
        if module is not None and hasattr(module, "kv_cache_append_forward"):
            next_k, next_v = module.kv_cache_append_forward(k_cache, v_cache, k_new, v_new)
            return next_k, next_v

    k_chunk = k_new.contiguous()
    v_chunk = v_new.contiguous()
    if k_cache is None or v_cache is None:
        return k_chunk, v_chunk
    return torch.cat([k_cache.contiguous(), k_chunk], dim=1), torch.cat([v_cache.contiguous(), v_chunk], dim=1)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    op_name = "attention_decode" if q.shape[2] == 1 else "attention_prefill"
    if has_native_op(op_name):
        module = native_module()
        if module is not None and hasattr(module, "attention_forward"):
            return module.attention_forward(q, k, v, attn_mask, is_causal, scale)

    scores = torch.matmul(q, k.transpose(2, 3))
    if scale is None:
        scores = scores * (q.shape[-1] ** -0.5)
    else:
        scores = scores * float(scale)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype)
    if is_causal:
        causal = torch.triu(
            torch.ones(q.shape[2], k.shape[2], device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal.view(1, 1, q.shape[2], k.shape[2]), float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v)
