from __future__ import annotations

import torch
import torch.nn.functional as F

from runtime.native import has_native_op, native_module, resolve_linear_backend


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


def temperature(logits: torch.Tensor, tau: float) -> torch.Tensor:
    if has_native_op("sampling"):
        module = native_module()
        if module is not None and hasattr(module, "temperature_forward"):
            return module.temperature_forward(logits, float(tau))
    return logits / max(float(tau), 1e-8)


def topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    if has_native_op("sampling"):
        module = native_module()
        if module is not None and hasattr(module, "topk_mask_forward"):
            return module.topk_mask_forward(logits, int(k))
    values = torch.topk(logits, k=int(k), dim=-1).values
    kth = values[..., -1:].contiguous()
    mask = logits < kth
    equals = logits == kth
    return mask & (~equals)


def topp_mask(logits: torch.Tensor, p: float) -> torch.Tensor:
    if has_native_op("sampling"):
        module = native_module()
        if module is not None and hasattr(module, "topp_mask_forward"):
            return module.topp_mask_forward(logits, float(p))
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cum > float(p)
    cutoff[..., 0] = False
    mask = torch.zeros_like(cutoff, dtype=torch.bool)
    return mask.scatter(-1, sorted_idx, cutoff)


def presence_frequency_penalty(
    logits: torch.Tensor,
    counts: torch.Tensor,
    alpha_presence: float,
    alpha_frequency: float,
) -> torch.Tensor:
    if has_native_op("sampling"):
        module = native_module()
        if module is not None and hasattr(module, "presence_frequency_penalty_forward"):
            return module.presence_frequency_penalty_forward(
                logits, counts, float(alpha_presence), float(alpha_frequency)
            )
    penalty = alpha_presence * (counts > 0).to(logits.dtype) + alpha_frequency * counts.to(logits.dtype)
    return logits - penalty


def sample_next_token(logits: torch.Tensor, do_sample: bool) -> torch.Tensor:
    if has_native_op("sampling"):
        module = native_module()
        if module is not None and hasattr(module, "sample_next_token_forward"):
            return module.sample_next_token_forward(logits, bool(do_sample))
    if do_sample:
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


def repetition_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if has_native_op("sampling"):
        module = native_module()
        if module is not None and hasattr(module, "repetition_penalty_forward"):
            return module.repetition_penalty_forward(logits, token_ids, float(penalty))
    out = logits.clone()
    if float(penalty) == 1.0 or token_ids.numel() == 0:
        return out
    for b in range(token_ids.shape[0]):
        idx = torch.unique(token_ids[b].to(torch.long))
        if idx.numel() == 0:
            continue
        values = out[b, idx]
        out[b, idx] = torch.where(values > 0, values / float(penalty), values * float(penalty))
    return out


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    if has_native_op("linear"):
        module = native_module()
        if module is not None and hasattr(module, "linear_forward"):
            return module.linear_forward(x, weight, bias, resolve_linear_backend(backend))
    return F.linear(x, weight, bias)


def qkv_projection(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor | None,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor | None,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor | None,
    *,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if has_native_op("qkv_projection"):
        module = native_module()
        if module is not None and hasattr(module, "qkv_projection_forward"):
            q, k, v = module.qkv_projection_forward(
                x,
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                resolve_linear_backend(backend),
            )
            return q, k, v
    return (
        linear(x, q_weight, q_bias, backend=backend),
        linear(x, k_weight, k_bias, backend=backend),
        linear(x, v_weight, v_bias, backend=backend),
    )


def mlp(
    x: torch.Tensor,
    w_in_weight: torch.Tensor,
    w_in_bias: torch.Tensor | None,
    w_out_weight: torch.Tensor,
    w_out_bias: torch.Tensor | None,
    *,
    activation: str,
    gated: bool,
    backend: str | None = None,
) -> torch.Tensor:
    if has_native_op("mlp"):
        module = native_module()
        if module is not None and hasattr(module, "mlp_forward"):
            return module.mlp_forward(
                x,
                w_in_weight,
                w_in_bias,
                w_out_weight,
                w_out_bias,
                str(activation),
                bool(gated),
                resolve_linear_backend(backend),
            )
    hidden = linear(x, w_in_weight, w_in_bias, backend=backend)
    act = str(activation).lower()
    if gated:
        a, b = hidden.chunk(2, dim=-1)
        if act in ("swiglu", "gated-silu"):
            hidden = F.silu(a) * b
        elif act == "geglu":
            hidden = F.gelu(a) * b
        elif act == "reglu":
            hidden = F.relu(a) * b
        else:
            hidden = F.silu(a) * b
    else:
        if act == "gelu":
            hidden = F.gelu(hidden)
        elif act in ("silu", "swish"):
            hidden = F.silu(hidden)
        else:
            hidden = F.gelu(hidden)
    return linear(hidden, w_out_weight, w_out_bias, backend=backend)


def embedding(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int | None = None,
) -> torch.Tensor:
    if has_native_op("embedding"):
        module = native_module()
        if module is not None and hasattr(module, "embedding_forward"):
            return module.embedding_forward(weight, indices, -1 if padding_idx is None else int(padding_idx))
    return F.embedding(indices, weight, padding_idx=padding_idx)
