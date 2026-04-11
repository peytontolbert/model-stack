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


def _layer_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    dim: int | tuple[int, ...] = -1,
) -> torch.Tensor:
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


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    dim: int | tuple[int, ...] = -1,
) -> torch.Tensor:
    if dim == -1 and has_native_op("layer_norm"):
        module = native_module()
        if module is not None and hasattr(module, "layer_norm_forward"):
            return module.layer_norm_forward(x, weight, bias, eps)
    return _layer_norm_reference(x, weight=weight, bias=bias, eps=eps, dim=dim)


def add_rms_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    residual_scale: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if has_native_op("add_rms_norm"):
        module = native_module()
        if module is not None and hasattr(module, "add_rms_norm_forward"):
            combined, normalized = module.add_rms_norm_forward(
                x,
                update,
                weight,
                float(residual_scale),
                float(eps),
            )
            return combined, normalized
    combined = x + (update * float(residual_scale))
    return combined, _rms_norm_reference(combined, weight=weight, eps=eps, dim=-1)


def add_layer_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    residual_scale: float = 1.0,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if has_native_op("add_layer_norm"):
        module = native_module()
        if module is not None and hasattr(module, "add_layer_norm_forward"):
            combined, normalized = module.add_layer_norm_forward(
                x,
                update,
                weight,
                bias,
                float(residual_scale),
                float(eps),
            )
            return combined, normalized
    combined = x + (update * float(residual_scale))
    return combined, _layer_norm_reference(combined, weight=weight, bias=bias, eps=eps, dim=-1)


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


def kv_cache_write(
    cache: torch.Tensor,
    chunk: torch.Tensor,
    start: int,
) -> torch.Tensor:
    if has_native_op("kv_cache_write"):
        module = native_module()
        if module is not None and hasattr(module, "kv_cache_write_forward"):
            return module.kv_cache_write_forward(cache, chunk, int(start))
    cache[:, int(start): int(start) + chunk.shape[1], :].copy_(chunk)
    return cache


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

    k_all = k
    v_all = v
    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1] != 0 or k.shape[1] != v.shape[1]:
            raise ValueError("attention fallback requires q heads to be a multiple of kv heads")
        repeat = q.shape[1] // k.shape[1]
        k_all = k.repeat_interleave(repeat, dim=1)
        v_all = v.repeat_interleave(repeat, dim=1)

    scores = torch.matmul(q, k_all.transpose(2, 3))
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
            torch.ones(q.shape[2], k_all.shape[2], device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal.view(1, 1, q.shape[2], k_all.shape[2]), float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v_all)


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
            return module.linear_forward(x, weight, bias, str(backend or "auto"))
    return F.linear(x, weight, bias)


def pack_linear_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if has_native_op("pack_linear_weight"):
        module = native_module()
        if module is not None and hasattr(module, "pack_linear_weight_forward"):
            packed_weight, packed_bias = module.pack_linear_weight_forward(weight, bias)
            return packed_weight, packed_bias
    return weight.contiguous(), None if bias is None else bias.contiguous()


def split_heads(
    x: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if has_native_op("split_heads"):
        module = native_module()
        if module is not None and hasattr(module, "split_heads_forward"):
            return module.split_heads_forward(x, int(num_heads))
    if x.ndim != 3:
        raise ValueError("split_heads expects x with shape (B, T, D)")
    bsz, seq, width = x.shape
    if width % int(num_heads) != 0:
        raise ValueError(f"Model dim {width} not divisible by heads {num_heads}")
    head_dim = width // int(num_heads)
    return x.view(bsz, seq, int(num_heads), head_dim).permute(0, 2, 1, 3).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    if has_native_op("merge_heads"):
        module = native_module()
        if module is not None and hasattr(module, "merge_heads_forward"):
            return module.merge_heads_forward(x)
    if x.ndim != 4:
        raise ValueError("merge_heads expects x with shape (B, H, T, Dh)")
    bsz, heads, seq, head_dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq, heads * head_dim)


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
                str(backend or "auto"),
            )
            return q, k, v
    return (
        linear(x, q_weight, q_bias, backend=backend),
        linear(x, k_weight, k_bias, backend=backend),
        linear(x, v_weight, v_bias, backend=backend),
    )


def pack_qkv_weights(
    q_weight: torch.Tensor,
    q_bias: torch.Tensor | None,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor | None,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, int, int, int]:
    if has_native_op("pack_qkv_weights"):
        module = native_module()
        if module is not None and hasattr(module, "pack_qkv_weights_forward"):
            packed_weight, packed_bias, q_size, k_size, v_size = module.pack_qkv_weights_forward(
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
            )
            return packed_weight, packed_bias, int(q_size), int(k_size), int(v_size)
    packed_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).contiguous()
    packed_bias = None
    if q_bias is not None or k_bias is not None or v_bias is not None:
        bias_parts: list[torch.Tensor] = []
        target_device = packed_weight.device
        target_dtype = packed_weight.dtype
        for bias, width in ((q_bias, q_weight.shape[0]), (k_bias, k_weight.shape[0]), (v_bias, v_weight.shape[0])):
            if bias is None:
                bias_parts.append(torch.zeros(width, device=target_device, dtype=target_dtype))
            else:
                bias_parts.append(bias.to(device=target_device, dtype=target_dtype).contiguous())
        packed_bias = torch.cat(bias_parts, dim=0).contiguous()
    return packed_weight, packed_bias, int(q_weight.shape[0]), int(k_weight.shape[0]), int(v_weight.shape[0])


def qkv_packed_heads_projection(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_bias: torch.Tensor | None,
    *,
    q_size: int,
    k_size: int,
    v_size: int,
    q_heads: int,
    kv_heads: int,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if has_native_op("qkv_packed_heads_projection"):
        module = native_module()
        if module is not None and hasattr(module, "qkv_packed_heads_projection_forward"):
            q, k, v = module.qkv_packed_heads_projection_forward(
                x,
                packed_weight,
                packed_bias,
                int(q_size),
                int(k_size),
                int(v_size),
                int(q_heads),
                int(kv_heads),
                str(backend or "auto"),
            )
            return q, k, v
    fused = linear(x, packed_weight, packed_bias, backend=backend)
    return (
        split_heads(fused[..., :q_size], q_heads),
        split_heads(fused[..., q_size: q_size + k_size], kv_heads),
        split_heads(fused[..., q_size + k_size: q_size + k_size + v_size], kv_heads),
    )


def qkv_heads_projection(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor | None,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor | None,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor | None,
    *,
    q_heads: int,
    kv_heads: int,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if has_native_op("qkv_heads_projection"):
        module = native_module()
        if module is not None and hasattr(module, "qkv_heads_projection_forward"):
            q, k, v = module.qkv_heads_projection_forward(
                x,
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                int(q_heads),
                int(kv_heads),
                str(backend or "auto"),
            )
            return q, k, v
    q, k, v = qkv_projection(
        x,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        backend=backend,
    )
    return (
        split_heads(q, q_heads),
        split_heads(k, kv_heads),
        split_heads(v, kv_heads),
    )


def head_output_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    if has_native_op("head_output_projection"):
        module = native_module()
        if module is not None and hasattr(module, "head_output_projection_forward"):
            return module.head_output_projection_forward(x, weight, bias, str(backend or "auto"))
    return linear(merge_heads(x), weight, bias, backend=backend)


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
                str(backend or "auto"),
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
