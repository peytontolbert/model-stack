from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from runtime.native import has_native_op, native_module, resolve_linear_backend


def _should_use_eager_autograd_fallback(*tensors: torch.Tensor | None) -> bool:
    if not torch.is_grad_enabled():
        return False
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            return True
    return False


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
    if len(dims) == 1 and dims[0] == x.ndim - 1:
        w = None if weight is None else weight.to(dtype=x.dtype, device=x.device)
        b = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
        return F.layer_norm(x, (x.shape[-1],), w, b, eps)
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
    if _should_use_eager_autograd_fallback(x, weight):
        return _rms_norm_reference(x, weight=weight, eps=eps, dim=dim)
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
    if _should_use_eager_autograd_fallback(x, weight, bias):
        return _layer_norm_reference(x, weight=weight, bias=bias, eps=eps, dim=dim)
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
    if _should_use_eager_autograd_fallback(x, update, weight):
        combined = x + (update * float(residual_scale))
        return combined, _rms_norm_reference(combined, weight=weight, eps=eps, dim=-1)
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


def residual_add(
    x: torch.Tensor,
    update: torch.Tensor,
    *,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    if _should_use_eager_autograd_fallback(x, update):
        return x + (update * float(residual_scale))
    if has_native_op("residual_add"):
        module = native_module()
        if module is not None and hasattr(module, "residual_add_forward"):
            return module.residual_add_forward(
                x,
                update,
                float(residual_scale),
            )
    return x + (update * float(residual_scale))


def add_layer_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    residual_scale: float = 1.0,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _should_use_eager_autograd_fallback(x, update, weight, bias):
        combined = x + (update * float(residual_scale))
        return combined, _layer_norm_reference(combined, weight=weight, bias=bias, eps=eps, dim=-1)
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
    if _should_use_eager_autograd_fallback(q, k, cos, sin):
        cos_b = cos.view(1, 1, cos.shape[0], cos.shape[1])
        sin_b = sin.view(1, 1, sin.shape[0], sin.shape[1])

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        return (q * cos_b) + (rotate_half(q) * sin_b), (k * cos_b) + (rotate_half(k) * sin_b)
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


def kv_cache_gather(
    cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if has_native_op("kv_cache_gather"):
        module = native_module()
        if module is not None and hasattr(module, "kv_cache_gather_forward"):
            return module.kv_cache_gather_forward(cache, positions)
    positions_long = positions.to(device=cache.device, dtype=torch.long)
    if cache.dim() == 3:
        return cache.index_select(1, positions_long)
    if cache.dim() != 4:
        raise ValueError("kv_cache_gather fallback requires cache to be rank-3 or rank-4")
    if positions_long.dim() == 1:
        return cache.index_select(2, positions_long)
    if positions_long.dim() != 2 or positions_long.shape[0] != cache.shape[0]:
        raise ValueError("kv_cache_gather fallback requires rank-2 positions to match cache batch size")
    return torch.stack(
        [cache[b].index_select(1, positions_long[b]) for b in range(cache.shape[0])],
        dim=0,
    )


def paged_kv_gather(
    pages: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if has_native_op("paged_kv_gather"):
        module = native_module()
        if module is not None and hasattr(module, "paged_kv_gather_forward"):
            return module.paged_kv_gather_forward(pages, block_table, positions)
    block_table_long = block_table.to(device=pages.device, dtype=torch.long)
    positions_long = positions.to(device=pages.device, dtype=torch.long)
    if pages.dim() != 4:
        raise ValueError("paged_kv_gather fallback requires pages to be rank-4 (P,H,page_size,D)")
    if block_table_long.dim() != 2:
        raise ValueError("paged_kv_gather fallback requires block_table to be rank-2 (B,max_blocks)")
    if positions_long.dim() not in (1, 2):
        raise ValueError("paged_kv_gather fallback requires positions to be rank-1 or rank-2")
    if positions_long.dim() == 2 and positions_long.shape[0] != block_table_long.shape[0]:
        raise ValueError("paged_kv_gather fallback requires rank-2 positions to match block_table batch size")
    gather_seq = int(positions_long.shape[0] if positions_long.dim() == 1 else positions_long.shape[1])
    out = torch.empty(
        block_table_long.shape[0],
        pages.shape[1],
        gather_seq,
        pages.shape[3],
        dtype=pages.dtype,
        device=pages.device,
    )
    page_size = int(pages.shape[2])
    for b in range(block_table_long.shape[0]):
        for t in range(gather_seq):
            pos = int(positions_long[t] if positions_long.dim() == 1 else positions_long[b, t])
            block_idx = pos // page_size
            page_offset = pos % page_size
            page_id = int(block_table_long[b, block_idx])
            out[b, :, t, :] = pages[page_id, :, page_offset, :]
    return out


def paged_kv_assign_blocks(
    block_table: torch.Tensor,
    block_ids: torch.Tensor,
    starts: torch.Tensor,
    total: int,
    page_size: int,
    next_page_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if has_native_op("paged_kv_assign_blocks"):
        module = native_module()
        if module is not None and hasattr(module, "paged_kv_assign_blocks_forward"):
            next_table, selected_table, next_page = module.paged_kv_assign_blocks_forward(
                block_table,
                block_ids,
                starts,
                int(total),
                int(page_size),
                int(next_page_id),
            )
            return next_table, selected_table, int(next_page)

    block_table_long = block_table.to(device=block_table.device, dtype=torch.long).contiguous()
    block_ids_long = block_ids.to(device=block_table.device, dtype=torch.long).contiguous().view(-1)
    starts_long = starts.to(device=block_table.device, dtype=torch.long).contiguous().view(-1)
    if block_table_long.dim() != 2:
        raise ValueError("paged_kv_assign_blocks fallback requires block_table to be rank-2 (B,max_blocks)")
    if block_ids_long.numel() != starts_long.numel():
        raise ValueError("paged_kv_assign_blocks fallback requires block_ids and starts to have the same length")
    if int(total) < 0:
        raise ValueError("paged_kv_assign_blocks fallback requires total to be non-negative")
    if int(page_size) <= 0:
        raise ValueError("paged_kv_assign_blocks fallback requires page_size to be positive")
    if int(next_page_id) < 0:
        raise ValueError("paged_kv_assign_blocks fallback requires next_page_id to be non-negative")
    if block_ids_long.numel() > 0:
        if int(block_ids_long.min().item()) < 0 or int(block_ids_long.max().item()) >= block_table_long.shape[0]:
            raise ValueError("paged_kv_assign_blocks fallback requires block_ids to be within block_table batch range")
        if int(torch.unique(block_ids_long).numel()) != int(block_ids_long.numel()):
            raise ValueError("paged_kv_assign_blocks fallback requires block_ids to be unique")
    if starts_long.numel() > 0 and int(starts_long.min().item()) < 0:
        raise ValueError("paged_kv_assign_blocks fallback requires starts to be non-negative")
    if block_ids_long.numel() == 0 or int(total) == 0:
        empty = torch.empty(block_ids_long.numel(), 0, dtype=torch.long, device=block_table_long.device)
        return block_table_long, empty, int(next_page_id)

    end_positions = starts_long + (int(total) - 1)
    start_blocks = torch.div(starts_long, int(page_size), rounding_mode="floor")
    end_blocks = torch.div(end_positions, int(page_size), rounding_mode="floor")
    needed_blocks = int(end_blocks.max().item()) + 1

    next_blocks = int(block_table_long.shape[1])
    if next_blocks < needed_blocks:
        next_blocks = max(1, next_blocks if next_blocks > 0 else 1)
        while next_blocks < needed_blocks:
            next_blocks *= 2
        next_table = torch.full(
            (block_table_long.shape[0], next_blocks),
            -1,
            dtype=torch.long,
            device=block_table_long.device,
        )
        if block_table_long.numel() > 0:
            next_table[:, : block_table_long.shape[1]].copy_(block_table_long)
    else:
        next_table = block_table_long.clone()

    selected = next_table.index_select(0, block_ids_long).clone()
    selected_prefix = selected[:, :needed_blocks].clone()
    active_slots = torch.arange(needed_blocks, device=block_table_long.device, dtype=torch.long).view(1, needed_blocks)
    active_mask = (active_slots >= start_blocks.view(-1, 1)) & (active_slots <= end_blocks.view(-1, 1))
    missing_mask = active_mask & (selected_prefix < 0)
    missing_count = int(missing_mask.sum().item())
    if missing_count > 0:
        new_ids = torch.arange(
            int(next_page_id),
            int(next_page_id) + missing_count,
            device=block_table_long.device,
            dtype=torch.long,
        )
        selected_prefix[missing_mask] = new_ids

    selected[:, :needed_blocks] = selected_prefix
    next_table.index_copy_(0, block_ids_long, selected)
    return next_table, selected_prefix.clamp_min(0).contiguous(), int(next_page_id) + missing_count


def paged_kv_reserve_pages(
    pages: torch.Tensor,
    used_pages: int,
    needed_pages: int,
) -> torch.Tensor:
    if has_native_op("paged_kv_reserve_pages"):
        module = native_module()
        if module is not None and hasattr(module, "paged_kv_reserve_pages_forward"):
            return module.paged_kv_reserve_pages_forward(pages, int(used_pages), int(needed_pages))

    if pages.dim() != 4:
        raise ValueError("paged_kv_reserve_pages fallback requires pages to be rank-4 (P,H,page_size,D)")
    if int(used_pages) < 0:
        raise ValueError("paged_kv_reserve_pages fallback requires used_pages to be non-negative")
    if int(needed_pages) < 0:
        raise ValueError("paged_kv_reserve_pages fallback requires needed_pages to be non-negative")
    if int(used_pages) > int(pages.shape[0]):
        raise ValueError("paged_kv_reserve_pages fallback requires used_pages to fit current capacity")
    if int(needed_pages) <= int(pages.shape[0]):
        return pages

    new_cap = max(1, int(pages.shape[0]) if int(pages.shape[0]) > 0 else 1)
    while new_cap < int(needed_pages):
        new_cap *= 2
    next_pages = torch.empty(
        new_cap,
        pages.shape[1],
        pages.shape[2],
        pages.shape[3],
        dtype=pages.dtype,
        device=pages.device,
    )
    if int(used_pages) > 0:
        next_pages[: int(used_pages)].copy_(pages[: int(used_pages)])
    return next_pages


def paged_kv_read_last(
    pages: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    keep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if has_native_op("paged_kv_read_last"):
        module = native_module()
        if module is not None and hasattr(module, "paged_kv_read_last_forward"):
            return module.paged_kv_read_last_forward(pages, block_table, lengths, int(keep))

    block_table_long = block_table.to(device=pages.device, dtype=torch.long).contiguous()
    lengths_long = lengths.to(device=pages.device, dtype=torch.long).contiguous().view(-1)
    if pages.dim() != 4:
        raise ValueError("paged_kv_read_last fallback requires pages to be rank-4 (P,H,page_size,D)")
    if block_table_long.dim() != 2:
        raise ValueError("paged_kv_read_last fallback requires block_table to be rank-2 (B,max_blocks)")
    if lengths_long.dim() != 1:
        raise ValueError("paged_kv_read_last fallback requires lengths to be rank-1")
    if block_table_long.shape[0] != lengths_long.shape[0]:
        raise ValueError("paged_kv_read_last fallback requires block_table and lengths batch sizes to match")
    if int(keep) < 0:
        raise ValueError("paged_kv_read_last fallback requires keep to be non-negative")
    kept_lengths = torch.clamp(lengths_long, min=0, max=int(keep))
    max_keep = int(kept_lengths.max().item()) if kept_lengths.numel() > 0 else 0
    out = torch.zeros(
        block_table_long.shape[0],
        pages.shape[1],
        max_keep,
        pages.shape[3],
        dtype=pages.dtype,
        device=pages.device,
    )
    if max_keep == 0 or block_table_long.shape[0] == 0:
        return out, kept_lengths
    page_size = int(pages.shape[2])
    for b in range(block_table_long.shape[0]):
        row_keep = int(kept_lengths[b].item())
        live_len = int(lengths_long[b].item())
        start = max(live_len - row_keep, 0)
        for t in range(row_keep):
            pos = start + t
            block_idx = pos // page_size
            page_offset = pos % page_size
            page_id = int(block_table_long[b, block_idx])
            out[b, :, t, :] = pages[page_id, :, page_offset, :]
    return out, kept_lengths


def paged_kv_read_range(
    pages: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    if has_native_op("paged_kv_read_range"):
        module = native_module()
        if module is not None and hasattr(module, "paged_kv_read_range_forward"):
            return module.paged_kv_read_range_forward(pages, block_table, lengths, int(start), int(end))

    block_table_long = block_table.to(device=pages.device, dtype=torch.long).contiguous().clamp_min(0)
    lengths_long = lengths.to(device=pages.device, dtype=torch.long).contiguous().view(-1)
    if pages.dim() != 4:
        raise ValueError("paged_kv_read_range fallback requires pages to be rank-4 (P,H,page_size,D)")
    if block_table_long.dim() != 2:
        raise ValueError("paged_kv_read_range fallback requires block_table to be rank-2 (B,max_blocks)")
    if lengths_long.dim() != 1:
        raise ValueError("paged_kv_read_range fallback requires lengths to be rank-1")
    if block_table_long.shape[0] != lengths_long.shape[0]:
        raise ValueError("paged_kv_read_range fallback requires block_table and lengths batch sizes to match")
    if int(start) < 0:
        raise ValueError("paged_kv_read_range fallback requires start to be non-negative")
    if int(end) < int(start):
        raise ValueError("paged_kv_read_range fallback requires end >= start")
    gather_seq = int(end) - int(start)
    out = torch.zeros(
        block_table_long.shape[0],
        pages.shape[1],
        gather_seq,
        pages.shape[3],
        dtype=pages.dtype,
        device=pages.device,
    )
    if gather_seq == 0 or block_table_long.shape[0] == 0:
        return out
    page_size = int(pages.shape[2])
    for b in range(block_table_long.shape[0]):
        live_len = int(lengths_long[b].item())
        for t in range(gather_seq):
            pos = int(start) + t
            if pos >= live_len:
                continue
            block_idx = pos // page_size
            page_offset = pos % page_size
            page_id = int(block_table_long[b, block_idx])
            out[b, :, t, :] = pages[page_id, :, page_offset, :]
    return out


def paged_kv_append(
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    page_count: int,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    block_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if has_native_op("paged_kv_append"):
        module = native_module()
        if module is not None and hasattr(module, "paged_kv_append_forward"):
            next_k_pages, next_v_pages, next_block_table, next_lengths, next_page_count = module.paged_kv_append_forward(
                k_pages, v_pages, block_table, lengths, int(page_count), k_chunk, v_chunk, block_ids
            )
            return next_k_pages, next_v_pages, next_block_table, next_lengths, int(next_page_count.item())

    block_ids_long = block_ids.to(device=k_pages.device, dtype=torch.long).contiguous().view(-1)
    lengths_long = lengths.to(device=k_pages.device, dtype=torch.long).contiguous().view(-1)
    if block_ids_long.numel() == 0 or int(k_chunk.shape[2]) == 0:
        return k_pages, v_pages, block_table.to(device=k_pages.device, dtype=torch.long).contiguous(), lengths_long, int(page_count)
    starts_long = lengths_long.index_select(0, block_ids_long)
    total = int(k_chunk.shape[2])
    base = torch.arange(total, device=k_pages.device, dtype=torch.long).view(1, total)
    positions = starts_long.view(-1, 1) + base
    next_block_table, selected_block_table, next_page_count = paged_kv_assign_blocks(
        block_table,
        block_ids_long,
        starts_long,
        total,
        int(k_pages.shape[2]),
        int(page_count),
    )
    next_k_pages = paged_kv_reserve_pages(k_pages, int(page_count), int(next_page_count))
    next_v_pages = paged_kv_reserve_pages(v_pages, int(page_count), int(next_page_count))
    next_k_pages = paged_kv_write(next_k_pages, selected_block_table, positions, k_chunk)
    next_v_pages = paged_kv_write(next_v_pages, selected_block_table, positions, v_chunk)
    next_lengths = lengths_long.clone()
    next_lengths.index_copy_(0, block_ids_long, starts_long + total)
    return next_k_pages, next_v_pages, next_block_table, next_lengths, int(next_page_count)


def paged_kv_compact(
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    keep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if has_native_op("paged_kv_compact"):
        module = native_module()
        if module is not None and hasattr(module, "paged_kv_compact_forward"):
            return module.paged_kv_compact_forward(k_pages, v_pages, block_table, lengths, int(keep))

    block_table_long = block_table.to(device=k_pages.device, dtype=torch.long).contiguous()
    block_table_safe = block_table_long.clamp_min(0)
    lengths_long = lengths.to(device=k_pages.device, dtype=torch.long).contiguous().view(-1)
    if k_pages.dim() != 4 or v_pages.dim() != 4:
        raise ValueError("paged_kv_compact fallback requires k_pages and v_pages to be rank-4")
    if k_pages.shape != v_pages.shape:
        raise ValueError("paged_kv_compact fallback requires k_pages and v_pages shapes to match")
    if block_table_long.dim() != 2:
        raise ValueError("paged_kv_compact fallback requires block_table to be rank-2")
    if lengths_long.dim() != 1:
        raise ValueError("paged_kv_compact fallback requires lengths to be rank-1")
    if block_table_long.shape[0] != lengths_long.shape[0]:
        raise ValueError("paged_kv_compact fallback requires block_table and lengths batch sizes to match")
    if int(keep) < 0:
        raise ValueError("paged_kv_compact fallback requires keep to be non-negative")

    kept_k, kept_lengths = paged_kv_read_last(k_pages, block_table_safe, lengths_long, int(keep))
    kept_v, kept_lengths_v = paged_kv_read_last(v_pages, block_table_safe, lengths_long, int(keep))
    if not torch.equal(kept_lengths, kept_lengths_v):
        raise ValueError("paged_kv_compact fallback got mismatched kept lengths for K and V")

    next_block_table = torch.empty(block_table_long.shape[0], 0, dtype=torch.long, device=block_table_long.device)
    next_k_pages = torch.empty(0, k_pages.shape[1], k_pages.shape[2], k_pages.shape[3], dtype=k_pages.dtype, device=k_pages.device)
    next_v_pages = torch.empty(0, v_pages.shape[1], v_pages.shape[2], v_pages.shape[3], dtype=v_pages.dtype, device=v_pages.device)
    next_page_id = 0
    unique_lengths = sorted({int(x) for x in kept_lengths.tolist() if int(x) > 0})
    for length in unique_lengths:
        group_ids = (kept_lengths == length).nonzero(as_tuple=False).flatten()
        starts = torch.zeros(group_ids.numel(), dtype=torch.long, device=block_table_long.device)
        next_block_table, selected_block_table, next_page_id_new = paged_kv_assign_blocks(
            next_block_table,
            group_ids,
            starts,
            int(length),
            int(k_pages.shape[2]),
            int(next_page_id),
        )
        next_k_pages = paged_kv_reserve_pages(next_k_pages, int(next_page_id), int(next_page_id_new))
        next_v_pages = paged_kv_reserve_pages(next_v_pages, int(next_page_id), int(next_page_id_new))
        positions = torch.arange(int(length), dtype=torch.long, device=block_table_long.device)
        next_k_pages = paged_kv_write(
            next_k_pages,
            selected_block_table,
            positions,
            kept_k.index_select(0, group_ids)[:, :, : int(length), :].contiguous(),
        )
        next_v_pages = paged_kv_write(
            next_v_pages,
            selected_block_table,
            positions,
            kept_v.index_select(0, group_ids)[:, :, : int(length), :].contiguous(),
        )
        next_page_id = int(next_page_id_new)

    return (
        next_k_pages[:next_page_id].contiguous(),
        next_v_pages[:next_page_id].contiguous(),
        next_block_table,
        kept_lengths,
    )


def paged_kv_write(
    pages: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if has_native_op("paged_kv_write"):
        module = native_module()
        if module is not None and hasattr(module, "paged_kv_write_forward"):
            return module.paged_kv_write_forward(pages, block_table, positions, values)
    block_table_long = block_table.to(device=pages.device, dtype=torch.long)
    positions_long = positions.to(device=pages.device, dtype=torch.long)
    values_contig = values.to(device=pages.device, dtype=pages.dtype)
    if pages.dim() != 4:
        raise ValueError("paged_kv_write fallback requires pages to be rank-4 (P,H,page_size,D)")
    if block_table_long.dim() != 2:
        raise ValueError("paged_kv_write fallback requires block_table to be rank-2 (B,max_blocks)")
    if positions_long.dim() not in (1, 2):
        raise ValueError("paged_kv_write fallback requires positions to be rank-1 or rank-2")
    if values_contig.dim() != 4:
        raise ValueError("paged_kv_write fallback requires values to be rank-4 (B,H,T,D)")
    if positions_long.dim() == 2 and positions_long.shape[0] != values_contig.shape[0]:
        raise ValueError("paged_kv_write fallback requires rank-2 positions to match values batch size")
    if positions_long.shape[-1] != values_contig.shape[2]:
        raise ValueError("paged_kv_write fallback requires positions length to match values T")
    page_size = int(pages.shape[2])
    pages_out = pages
    for b in range(values_contig.shape[0]):
        for t in range(values_contig.shape[2]):
            pos = int(positions_long[t] if positions_long.dim() == 1 else positions_long[b, t])
            block_idx = pos // page_size
            page_offset = pos % page_size
            page_id = int(block_table_long[b, block_idx])
            pages_out[page_id, :, page_offset, :] = values_contig[b, :, t, :]
    return pages_out


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
    if _should_use_eager_autograd_fallback(q, k, v, attn_mask):
        op_name = ""
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
    if is_causal and q.shape[2] > 1:
        causal = torch.triu(
            torch.ones(q.shape[2], k_all.shape[2], device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal.view(1, 1, q.shape[2], k_all.shape[2]), float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v_all)


def attention_plan_info(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
) -> dict[str, object]:
    module = native_module()
    if module is not None and hasattr(module, "attention_plan_info"):
        return dict(module.attention_plan_info(q, k, v, attn_mask, is_causal))
    return {
        "backend": "aten",
        "kernel": "reference",
        "phase": "decode" if q.shape[2] == 1 else "prefill",
        "head_mode": "mha" if q.shape[1] == k.shape[1] else ("mqa" if k.shape[1] == 1 else "gqa"),
        "mask_kind": (
            "none"
            if attn_mask is None
            else ("bool" if attn_mask.dtype == torch.bool else "additive_same_dtype")
        ),
        "row_reduce_threads": 0,
        "head_dim_bucket": int(q.shape[3]),
        "batch": int(q.shape[0]),
        "q_heads": int(q.shape[1]),
        "kv_heads": int(k.shape[1]),
        "q_len": int(q.shape[2]),
        "kv_len": int(k.shape[2]),
        "head_dim": int(q.shape[3]),
        "causal": bool(is_causal),
    }


def prepare_attention_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_heads: int,
    tgt_len: int,
    src_len: int,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if has_native_op("prepare_attention_mask"):
        module = native_module()
        if module is not None and hasattr(module, "prepare_attention_mask_forward"):
            return module.prepare_attention_mask_forward(
                mask,
                int(batch_size),
                int(num_heads),
                int(tgt_len),
                int(src_len),
                position_ids,
            )

    prepared = mask
    if prepared.dtype == torch.bool:
        prepared = torch.where(
            prepared,
            torch.full_like(prepared, float("-inf"), dtype=torch.float32),
            torch.zeros_like(prepared, dtype=torch.float32),
        )
    else:
        prepared = prepared.to(dtype=torch.float32)
    if prepared.shape[-1] != src_len:
        prepared = prepared[..., :src_len]
    if prepared.ndim == 4 and prepared.shape[1] == 1:
        prepared = prepared.expand(int(batch_size), int(num_heads), int(tgt_len), int(src_len))
    if position_ids is not None:
        try:
            if position_ids.dim() == 2:
                pos_idx = position_ids.clamp_min(0).clamp_max(max(int(src_len) - 1, 0))
                zeros = torch.zeros(
                    int(batch_size), int(num_heads), int(tgt_len), 1,
                    dtype=prepared.dtype,
                    device=prepared.device,
                )
                idx = pos_idx.view(int(batch_size), 1, int(tgt_len), 1).expand(int(batch_size), int(num_heads), int(tgt_len), 1)
                prepared = prepared.scatter(dim=3, index=idx, src=zeros)
        except Exception:
            pass
    return prepared


def resolve_position_ids(
    *,
    batch_size: int,
    seq_len: int,
    reference: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    cache_position: torch.Tensor | None = None,
    past_length: int = 0,
) -> torch.Tensor:
    if has_native_op("resolve_position_ids"):
        module = native_module()
        if module is not None and hasattr(module, "resolve_position_ids_forward"):
            return module.resolve_position_ids_forward(
                int(batch_size),
                int(seq_len),
                reference,
                attention_mask,
                cache_position,
                int(past_length),
            )
    if cache_position is not None:
        return cache_position.view(1, -1).expand(int(batch_size), -1).to(device=reference.device, dtype=torch.long)
    if int(past_length) > 0:
        base = torch.arange(int(past_length), int(past_length) + int(seq_len), device=reference.device, dtype=torch.long)
        return base.view(1, -1).expand(int(batch_size), -1)
    if attention_mask is not None:
        lengths = attention_mask.to(torch.long).sum(dim=-1).view(int(batch_size), 1)
        starts = (lengths - int(seq_len)).clamp_min(0)
        base = torch.arange(int(seq_len), device=reference.device, dtype=torch.long).view(1, int(seq_len))
        return base + starts
    return torch.arange(int(seq_len), device=reference.device, dtype=torch.long).view(1, -1).expand(int(batch_size), -1)


def create_causal_mask(
    *,
    reference: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    cache_position: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if has_native_op("create_causal_mask"):
        module = native_module()
        if module is not None and hasattr(module, "create_causal_mask_forward"):
            return module.create_causal_mask_forward(reference, attention_mask, cache_position, position_ids)

    batch_size, tgt_len, _ = reference.shape
    src_len = tgt_len
    if position_ids is not None:
        try:
            src_len = int(position_ids.max().item()) + 1
        except Exception:
            src_len = max(tgt_len, 1)
    elif cache_position is not None:
        try:
            src_len = int(cache_position.max().item()) + 1
        except Exception:
            src_len = max(tgt_len, 1)
    if attention_mask is not None and attention_mask.shape[-1] > src_len:
        src_len = int(attention_mask.shape[-1])

    if position_ids is not None:
        pos = position_ids[0] if position_ids.dim() == 2 else position_ids
        qpos = pos.view(int(tgt_len), 1)
        kpos = torch.arange(int(src_len), device=reference.device, dtype=qpos.dtype).view(1, int(src_len))
        causal_bool = (kpos > qpos).view(1, 1, int(tgt_len), int(src_len))
    else:
        causal = torch.triu(
            torch.ones(int(src_len), int(src_len), device=reference.device, dtype=torch.bool),
            diagonal=1,
        )
        causal_bool = causal[-int(tgt_len):].view(1, 1, int(tgt_len), int(src_len))

    combined_bool = causal_bool
    if attention_mask is not None:
        pad = attention_mask
        if pad.shape[-1] < src_len:
            pad_fill = torch.ones(pad.shape[0], int(src_len) - pad.shape[-1], dtype=pad.dtype, device=pad.device)
            pad = torch.cat([pad, pad_fill], dim=-1)
        elif pad.shape[-1] > src_len:
            pad = pad[..., :int(src_len)]
        combined_bool = combined_bool | pad.eq(0).view(int(batch_size), 1, 1, int(src_len))

    return torch.where(
        combined_bool,
        torch.full(combined_bool.shape, float("-inf"), device=reference.device, dtype=torch.float32),
        torch.zeros(combined_bool.shape, device=reference.device, dtype=torch.float32),
    )


def resolve_rotary_embedding(
    *,
    reference: torch.Tensor,
    head_dim: int,
    base_theta: float,
    attention_scaling: float = 1.0,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if has_native_op("resolve_rotary_embedding"):
        module = native_module()
        if module is not None and hasattr(module, "resolve_rotary_embedding_forward"):
            return module.resolve_rotary_embedding_forward(
                reference,
                int(head_dim),
                float(base_theta),
                float(attention_scaling),
                position_ids,
            )

    seq_len = int(reference.shape[1])
    needed = seq_len
    gather_pos = None
    if position_ids is not None:
        gather_pos = position_ids[0] if position_ids.dim() == 2 else position_ids
        if gather_pos.numel() > 0:
            needed = int(gather_pos.max().item()) + 1
    t = torch.arange(needed, device=reference.device, dtype=torch.float32)
    inv_idx = torch.arange(0, int(head_dim), 2, device=reference.device, dtype=torch.int64).float()
    inv_freq = 1.0 / (float(base_theta) ** (inv_idx / float(head_dim)))
    freqs = torch.einsum("t,d->td", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    if float(attention_scaling) != 1.0:
        cos = cos * float(attention_scaling)
        sin = sin * float(attention_scaling)
    cos = cos.to(dtype=reference.dtype)
    sin = sin.to(dtype=reference.dtype)
    if gather_pos is not None:
        cos = cos.index_select(0, gather_pos.reshape(-1).to(torch.long)).view(seq_len, int(head_dim))
        sin = sin.index_select(0, gather_pos.reshape(-1).to(torch.long)).view(seq_len, int(head_dim))
    else:
        cos = cos[:seq_len]
        sin = sin[:seq_len]
    return cos, sin


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


def apply_sampling_mask(
    logits: torch.Tensor,
    *,
    topk_mask: torch.Tensor | None = None,
    topp_mask: torch.Tensor | None = None,
    no_repeat_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if has_native_op("sampling"):
        module = native_module()
        if module is not None and hasattr(module, "apply_sampling_mask_forward"):
            return module.apply_sampling_mask_forward(logits, topk_mask, topp_mask, no_repeat_mask)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    if topk_mask is not None:
        mask = mask | topk_mask
    if topp_mask is not None:
        mask = mask | topp_mask
    if no_repeat_mask is not None:
        mask = mask | no_repeat_mask
    if not mask.any():
        return logits
    min_val = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
    return logits.masked_fill(mask, min_val)


def beam_search_step(
    beams: torch.Tensor,
    logits: torch.Tensor,
    raw_scores: torch.Tensor,
    finished: torch.Tensor,
    lengths: torch.Tensor,
    *,
    beam_size: int,
    eos_id: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if has_native_op("beam_search_step"):
        module = native_module()
        if module is not None and hasattr(module, "beam_search_step_forward"):
            return module.beam_search_step_forward(
                beams,
                logits,
                raw_scores,
                finished,
                lengths,
                int(beam_size),
                int(eos_id),
                int(pad_id),
            )

    if logits.dim() == 3:
        next_logits = logits[:, -1, :]
    elif logits.dim() == 2:
        next_logits = logits
    else:
        raise ValueError("beam_search_step: logits must be rank-2 or rank-3")

    batch_size = raw_scores.shape[0]
    vocab_size = next_logits.shape[-1]
    logp = torch.log_softmax(next_logits, dim=-1).view(batch_size, beam_size, vocab_size)

    if finished.any():
        logp = logp.masked_fill(finished.unsqueeze(-1), float("-inf"))
        logp[:, :, pad_id] = torch.where(
            finished,
            torch.zeros_like(logp[:, :, pad_id]),
            logp[:, :, pad_id],
        )

    candidate_scores = raw_scores.unsqueeze(-1) + logp
    flat_scores = candidate_scores.view(batch_size, beam_size * vocab_size)
    best_raw_scores, best_idx = torch.topk(flat_scores, k=beam_size, dim=-1)

    parent_beams = best_idx // vocab_size
    next_tokens = best_idx % vocab_size

    prev_beams = beams.view(batch_size, beam_size, -1)
    gathered = prev_beams.gather(
        1,
        parent_beams.unsqueeze(-1).expand(-1, -1, prev_beams.shape[-1]),
    )

    parent_finished = finished.gather(1, parent_beams)
    appended_tokens = torch.where(
        parent_finished,
        torch.full_like(next_tokens, pad_id),
        next_tokens,
    )
    next_beams = torch.cat(
        [gathered.reshape(batch_size * beam_size, -1), appended_tokens.reshape(batch_size * beam_size, 1)],
        dim=1,
    )

    parent_lengths = lengths.gather(1, parent_beams)
    next_lengths = parent_lengths + (~parent_finished).long()
    next_finished = parent_finished | (next_tokens == eos_id)
    batch_offsets = torch.arange(batch_size, device=best_idx.device, dtype=torch.long).unsqueeze(-1) * int(beam_size)
    parent_rows = (batch_offsets + parent_beams).reshape(-1)
    return next_beams, best_raw_scores, next_finished, next_lengths, parent_rows


def sample_with_policies(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    no_repeat_ngram: int = 0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> torch.Tensor:
    if has_native_op("sampling"):
        module = native_module()
        if module is not None and hasattr(module, "sample_with_policies_forward"):
            return module.sample_with_policies_forward(
                logits,
                token_ids,
                bool(do_sample),
                float(temperature),
                None if top_k is None else int(top_k),
                None if top_p is None else float(top_p),
                int(no_repeat_ngram),
                float(repetition_penalty),
                float(presence_penalty),
                float(frequency_penalty),
            )

    temperature_fn = globals()["temperature"]
    repetition_penalty_fn = globals()["repetition_penalty"]
    x = logits
    if float(repetition_penalty) != 1.0:
        x = repetition_penalty_fn(x, token_ids, float(repetition_penalty))
    if bool(do_sample) and float(temperature) != 1.0:
        x = temperature_fn(x, float(temperature))
    topk_mask_tensor = topk_mask(x, int(top_k)) if bool(do_sample) and top_k is not None else None
    topp_mask_tensor = topp_mask(x, float(top_p)) if bool(do_sample) and top_p is not None else None
    no_repeat_mask_tensor = (
        no_repeat_ngram_mask(token_ids, vocab_size=x.shape[-1], n=int(no_repeat_ngram))
        if int(no_repeat_ngram) > 0
        else None
    )
    if topk_mask_tensor is not None or topp_mask_tensor is not None or no_repeat_mask_tensor is not None:
        x = apply_sampling_mask(
            x,
            topk_mask=topk_mask_tensor,
            topp_mask=topp_mask_tensor,
            no_repeat_mask=no_repeat_mask_tensor,
        )
    if float(presence_penalty) != 0.0 or float(frequency_penalty) != 0.0:
        counts = token_counts(token_ids, vocab_size=x.shape[-1], dtype=x.dtype)
        x = presence_frequency_penalty(x, counts, float(presence_penalty), float(frequency_penalty))
    return sample_next_token(x, bool(do_sample))


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


def no_repeat_ngram_mask(
    token_ids: torch.Tensor,
    *,
    vocab_size: int,
    n: int,
) -> torch.Tensor:
    if has_native_op("sampling"):
        module = native_module()
        if module is not None and hasattr(module, "no_repeat_ngram_mask_forward"):
            return module.no_repeat_ngram_mask_forward(token_ids, int(vocab_size), int(n))
    mask = torch.zeros(token_ids.shape[0], int(vocab_size), dtype=torch.bool, device=token_ids.device)
    if int(n) <= 0 or token_ids.shape[1] < int(n):
        return mask
    if int(n) == 1:
        ones = torch.ones_like(token_ids, dtype=torch.bool)
        return mask.scatter(1, token_ids.to(torch.long), ones)
    for b in range(token_ids.shape[0]):
        seq = token_ids[b].tolist()
        recent = tuple(seq[-(int(n) - 1):])
        for i in range(len(seq) - int(n) + 1):
            if tuple(seq[i:i + int(n) - 1]) == recent:
                nxt = seq[i + int(n) - 1]
                mask[b, nxt] = True
    return mask


def token_counts(
    token_ids: torch.Tensor,
    *,
    vocab_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if has_native_op("token_counts"):
        module = native_module()
        if module is not None and hasattr(module, "token_counts_forward"):
            return module.token_counts_forward(token_ids, int(vocab_size), dtype)
    counts = torch.zeros(token_ids.shape[0], int(vocab_size), dtype=dtype, device=token_ids.device)
    if token_ids.numel() == 0:
        return counts
    ones = torch.ones_like(token_ids, dtype=dtype)
    return counts.scatter_add(1, token_ids.to(torch.long), ones)


def append_tokens(
    seq: torch.Tensor,
    next_id: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    row_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if has_native_op("append_tokens"):
        module = native_module()
        if module is not None and hasattr(module, "append_tokens_forward"):
            next_seq, next_mask = module.append_tokens_forward(seq, next_id, attention_mask, row_ids)
            return next_seq, next_mask
    if row_ids is not None:
        row_ids_long = row_ids.to(device=seq.device, dtype=torch.long).contiguous().view(-1)
        seq = seq.index_select(0, row_ids_long)
        if attention_mask is not None:
            attention_mask = attention_mask.index_select(0, row_ids_long)
    next_seq = torch.cat([seq, next_id], dim=1)
    if attention_mask is None:
        return next_seq, None
    ones = torch.ones(next_id.shape[0], next_id.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
    return next_seq, torch.cat([attention_mask, ones], dim=1)


def decode_positions(
    *,
    batch_size: int,
    seq_len: int,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if has_native_op("decode_positions"):
        module = native_module()
        if module is not None and hasattr(module, "decode_positions_forward"):
            return module.decode_positions_forward(int(batch_size), int(seq_len), reference)
    pos_ids = torch.full((int(batch_size), 1), int(seq_len) - 1, device=reference.device, dtype=torch.long)
    return pos_ids, pos_ids.view(-1)


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
    if _should_use_eager_autograd_fallback(x, weight, bias):
        return F.linear(x, weight, bias)
    if has_native_op("linear"):
        module = native_module()
        if module is not None and hasattr(module, "linear_forward"):
            return module.linear_forward(x, weight, bias, str(backend or "auto"))
    return F.linear(x, weight, bias)


def _tensor_signature(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(tensor.shape),
        int(getattr(tensor, "_version", 0)),
        int(tensor.data_ptr()),
    )


def resolve_linear_module_tensors(
    module,
    *,
    reference: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if reference is not None:
        dtype = reference.dtype
        device = reference.device
    target_device = None if device is None else torch.device(device)

    runtime_weight = getattr(module, "runtime_weight", None)
    if callable(runtime_weight):
        weight = runtime_weight(dtype=dtype, device=target_device)
    else:
        weight = getattr(module, "weight")
        if target_device is not None or dtype is not None:
            weight = weight.to(
                device=weight.device if target_device is None else target_device,
                dtype=weight.dtype if dtype is None else dtype,
            )

    runtime_bias = getattr(module, "runtime_bias", None)
    if callable(runtime_bias):
        bias = runtime_bias(dtype=dtype, device=target_device)
    else:
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor) and (target_device is not None or dtype is not None):
            bias = bias.to(
                device=bias.device if target_device is None else target_device,
                dtype=bias.dtype if dtype is None else dtype,
            )
    return weight, bias


def linear_module_signature(module):
    runtime_signature = getattr(module, "runtime_signature", None)
    if callable(runtime_signature):
        return runtime_signature()
    weight, bias = resolve_linear_module_tensors(module)
    return (_tensor_signature(weight), _tensor_signature(bias))


def packed_linear_module_signature(
    module,
    *,
    backend: str | None = None,
):
    resolved_backend = resolve_linear_backend(backend)
    runtime_signature = getattr(module, "runtime_packed_linear_signature", None)
    if callable(runtime_signature):
        signature = runtime_signature(resolved_backend)
        if signature is not None:
            return signature
    return (resolved_backend, linear_module_signature(module))


def linear_module(
    x: torch.Tensor,
    module,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    runtime_forward = getattr(module, "runtime_linear", None)
    if callable(runtime_forward):
        return runtime_forward(x, backend=backend)
    weight, bias = resolve_linear_module_tensors(module, reference=x)
    return linear(x, weight, bias, backend=backend)


def resolve_packed_linear_module_spec(
    module,
    *,
    backend: str | None = None,
    reference: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    resolved_backend = resolve_linear_backend(backend)
    if reference is not None:
        dtype = reference.dtype
        device = reference.device
    target_device = None if device is None else torch.device(device)

    runtime_linear = getattr(module, "runtime_linear", None)
    if callable(runtime_linear):
        supports_packed_backend = getattr(module, "runtime_supports_packed_backend", None)
        if not callable(supports_packed_backend) or not bool(supports_packed_backend(resolved_backend)):
            return None
        runtime_spec = getattr(module, "runtime_packed_linear_spec", None)
        if not callable(runtime_spec):
            return None
        spec = runtime_spec(backend=resolved_backend, dtype=dtype, device=target_device)
        return None if spec is None else dict(spec)

    if resolved_backend == "cublaslt":
        weight, bias = resolve_linear_module_tensors(module, reference=reference, dtype=dtype, device=target_device)
        packed_weight, packed_bias = pack_linear_weight(weight, bias)
        return {
            "format": "dense_packed",
            "backend": resolved_backend,
            "packed_weight": packed_weight,
            "bias": packed_bias,
        }
    return None


def packed_qkv_module_signature(
    q_module,
    k_module,
    v_module,
    *,
    backend: str | None = None,
):
    resolved_backend = resolve_linear_backend(backend)
    return (
        resolved_backend,
        packed_linear_module_signature(q_module, backend=resolved_backend),
        packed_linear_module_signature(k_module, backend=resolved_backend),
        packed_linear_module_signature(v_module, backend=resolved_backend),
    )


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


def pack_bitnet_weight(
    weight: torch.Tensor,
    scale_values: torch.Tensor | None = None,
    layout_header: torch.Tensor | None = None,
    segment_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale_values is not None or layout_header is not None or segment_offsets is not None:
        raise NotImplementedError("Explicit BitNet metadata overrides are not yet supported in the Python fallback")
    if has_native_op("pack_bitnet_weight"):
        module = native_module()
        if module is not None and hasattr(module, "pack_bitnet_weight_forward"):
            try:
                packed_weight, packed_scale_values, packed_layout_header, packed_segment_offsets = (
                    module.pack_bitnet_weight_forward(weight)
                )
                return packed_weight, packed_scale_values, packed_layout_header, packed_segment_offsets
            except RuntimeError:
                # Keep the reference path usable even when the local native extension has not
                # been rebuilt after packer-side metadata changes.
                pass
    if weight.ndim != 2:
        raise ValueError("pack_bitnet_weight expects a rank-2 weight tensor")
    if not weight.dtype.is_floating_point:
        raise TypeError("pack_bitnet_weight expects a floating-point weight tensor")
    logical_out, logical_in = int(weight.shape[0]), int(weight.shape[1])
    padded_out = ((logical_out + 15) // 16) * 16
    padded_in = ((logical_in + 31) // 32) * 32
    scale = weight.detach().abs().amax().clamp_min(1e-8).to(device=weight.device, dtype=torch.float32)
    quantized = torch.round(weight.float() / scale).clamp_(-1, 1).to(dtype=torch.int16)
    codes = torch.ones((padded_out, padded_in), device=weight.device, dtype=torch.uint8)
    codes[:logical_out, :logical_in] = (quantized + 1).to(dtype=torch.uint8)
    packed_weight = (
        codes[:, 0::4]
        | (codes[:, 1::4] << 2)
        | (codes[:, 2::4] << 4)
        | (codes[:, 3::4] << 6)
    ).contiguous()
    packed_scale_values = scale.reshape(1).contiguous()
    packed_layout_header = torch.tensor(
        [
            1,
            16,
            32,
            logical_out,
            logical_in,
            padded_out,
            padded_in,
            0,
            logical_out,
            1,
            80,
            1,
            0,
        ],
        device=weight.device,
        dtype=torch.int32,
    )
    packed_segment_offsets = torch.tensor([0, logical_out], device=weight.device, dtype=torch.int32)
    return packed_weight, packed_scale_values, packed_layout_header, packed_segment_offsets


def split_heads(
    x: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if _should_use_eager_autograd_fallback(x):
        if x.ndim != 3:
            raise ValueError("split_heads expects x with shape (B, T, D)")
        bsz, seq, width = x.shape
        if width % int(num_heads) != 0:
            raise ValueError(f"Model dim {width} not divisible by heads {num_heads}")
        head_dim = width // int(num_heads)
        return x.view(bsz, seq, int(num_heads), head_dim).permute(0, 2, 1, 3).contiguous()
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
    if _should_use_eager_autograd_fallback(x):
        if x.ndim != 4:
            raise ValueError("merge_heads expects x with shape (B, H, T, Dh)")
        bsz, heads, seq, head_dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq, heads * head_dim)
    if has_native_op("merge_heads"):
        module = native_module()
        if module is not None and hasattr(module, "merge_heads_forward"):
            return module.merge_heads_forward(x)
    if x.ndim != 4:
        raise ValueError("merge_heads expects x with shape (B, H, T, Dh)")
    bsz, heads, seq, head_dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq, heads * head_dim)


def _quant_max(bits: int) -> int:
    bits = int(bits)
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return (1 << (bits - 1)) - 1


def _calibrate_activation_scale(
    x: torch.Tensor,
    *,
    method: str,
    bits: int,
    percentile: float,
) -> torch.Tensor:
    del percentile
    method_name = str(method).strip().lower()
    qmax = float(_quant_max(bits))
    values = x.detach().float()
    if method_name in {"", "absmax"}:
        clip = values.abs().amax()
    else:
        raise ValueError(f"Unsupported packed activation calibration method: {method}")
    return (clip.clamp_min(1e-8) / qmax).to(dtype=torch.float32)


def _fake_quantize_activation(
    x: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    bits: int,
) -> torch.Tensor:
    qmax = float(_quant_max(bits))
    scale_t = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, device=x.device, dtype=torch.float32)
    scale_t = scale_t.to(device=x.device, dtype=torch.float32).clamp_min(1e-8)
    while scale_t.ndim < x.ndim:
        scale_t = scale_t.unsqueeze(0)
    q = torch.round(x.float() / scale_t).clamp_(-qmax, qmax)
    return (q * scale_t).to(dtype=x.dtype)


def _power_of_two_segments(size: int) -> tuple[int, ...]:
    if size <= 0:
        return ()
    out: list[int] = []
    remaining = int(size)
    while remaining > 0:
        seg = 1 << (remaining.bit_length() - 1)
        out.append(seg)
        remaining -= seg
    return tuple(out)


def _hadamard_lastdim_power2(x: torch.Tensor) -> torch.Tensor:
    width = int(x.shape[-1])
    if width <= 1:
        return x
    if width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    y = x.reshape(-1, width)
    block = 1
    while block < width:
        y = y.view(-1, width // (block * 2), 2, block)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.cat((a + b, a - b), dim=-1)
        block *= 2
    return (y.view_as(x) / math.sqrt(float(width))).contiguous()


def _apply_spin_transform(x: torch.Tensor, spin_signs: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != spin_signs.numel():
        raise ValueError("spin_signs must match the last dimension of x")
    work_dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    signs = spin_signs.to(device=x.device, dtype=work_dtype)
    x_local = x.to(dtype=work_dtype) * signs
    segments = _power_of_two_segments(int(signs.numel()))
    if not segments:
        return x_local
    if len(segments) == 1:
        return _hadamard_lastdim_power2(x_local)
    parts = []
    start = 0
    for seg in segments:
        stop = start + seg
        part = x_local[..., start:stop]
        parts.append(_hadamard_lastdim_power2(part) if seg > 1 else part)
        start = stop
    return torch.cat(parts, dim=-1).contiguous()


def _packed_pre_scale_active(pre_scale: torch.Tensor | None) -> bool:
    if pre_scale is None:
        return False
    return bool((pre_scale.detach().float() - 1.0).abs().max().item() > 1e-6)


def _apply_pre_scale_to_input(x: torch.Tensor, pre_scale: torch.Tensor) -> torch.Tensor:
    view_shape = [1] * x.ndim
    view_shape[-1] = pre_scale.numel()
    return x / pre_scale.to(device=x.device, dtype=x.dtype).view(*view_shape)


def _apply_packed_activation_quantization(
    x: torch.Tensor,
    *,
    mode: str,
    act_scale: torch.Tensor | None,
    bits: int,
    method: str,
    percentile: float,
) -> torch.Tensor:
    mode_name = str(mode).strip().lower()
    if mode_name in {"", "none", "off"}:
        return x
    if mode_name == "dynamic_int8":
        scale = _calibrate_activation_scale(x, method=method, bits=bits, percentile=percentile)
        return _fake_quantize_activation(x, scale, bits=bits)
    if mode_name == "static_int8":
        if act_scale is None:
            raise ValueError("Packed static_int8 activation quantization requires act_scale")
        return _fake_quantize_activation(x, act_scale, bits=bits)
    raise ValueError(f"Unknown packed activation quantization mode: {mode}")


def _apply_packed_linear_input_spec(x: torch.Tensor, spec: dict) -> torch.Tensor:
    x_local = x
    if bool(spec.get("spin_enabled", False)):
        spin_signs = spec.get("spin_signs")
        if not isinstance(spin_signs, torch.Tensor):
            raise ValueError("Packed linear spec with spin_enabled requires spin_signs")
        x_local = _apply_spin_transform(x_local, spin_signs)
    pre_scale = spec.get("pre_scale")
    if isinstance(pre_scale, torch.Tensor) and _packed_pre_scale_active(pre_scale):
        x_local = _apply_pre_scale_to_input(x_local, pre_scale)
    x_local = _apply_packed_activation_quantization(
        x_local,
        mode=str(spec.get("act_quant_mode", "none")),
        act_scale=spec.get("act_scale") if isinstance(spec.get("act_scale"), torch.Tensor) else None,
        bits=int(spec.get("act_quant_bits", 8)),
        method=str(spec.get("act_quant_method", "absmax")),
        percentile=float(spec.get("act_quant_percentile", 0.999)),
    )
    return x_local


def linear_from_packed_spec(
    x: torch.Tensor,
    spec: dict,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    format_name = str(spec.get("format"))
    if format_name == "dense_packed":
        return linear(x, spec["packed_weight"], spec.get("bias"), backend=backend)
    if format_name == "bitnet_w2a8":
        x_local = _apply_packed_linear_input_spec(x, spec)
        return _bitnet_linear_from_transformed_input(x_local, spec)
    raise NotImplementedError(f"Packed linear format {format_name} is not implemented")


def _bitnet_linear_from_transformed_input(
    x: torch.Tensor,
    spec: dict,
) -> torch.Tensor:
    from runtime.quant import bitnet_linear as runtime_bitnet_linear

    return runtime_bitnet_linear(
        x,
        spec["packed_weight"],
        spec["scale_values"],
        spec["layout_header"],
        spec["segment_offsets"],
        bias=spec.get("bias"),
    )


def bitnet_qkv_packed_heads_projection(
    x: torch.Tensor,
    q_spec: dict,
    k_spec: dict,
    v_spec: dict,
    *,
    q_heads: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qx = _apply_packed_linear_input_spec(x, q_spec)
    kx = _apply_packed_linear_input_spec(x, k_spec)
    vx = _apply_packed_linear_input_spec(x, v_spec)
    q_bias = q_spec.get("bias")
    k_bias = k_spec.get("bias")
    v_bias = v_spec.get("bias")

    if not _should_use_eager_autograd_fallback(
        qx,
        kx,
        vx,
        q_spec["packed_weight"],
        q_bias,
        k_spec["packed_weight"],
        k_bias,
        v_spec["packed_weight"],
        v_bias,
    ):
        if has_native_op("bitnet_qkv_packed_heads_projection"):
            module = native_module()
            if module is not None and hasattr(module, "bitnet_qkv_packed_heads_projection_forward"):
                q, k, v = module.bitnet_qkv_packed_heads_projection_forward(
                    qx,
                    q_spec["packed_weight"],
                    q_spec["scale_values"],
                    q_spec["layout_header"],
                    q_spec["segment_offsets"],
                    q_bias,
                    kx,
                    k_spec["packed_weight"],
                    k_spec["scale_values"],
                    k_spec["layout_header"],
                    k_spec["segment_offsets"],
                    k_bias,
                    vx,
                    v_spec["packed_weight"],
                    v_spec["scale_values"],
                    v_spec["layout_header"],
                    v_spec["segment_offsets"],
                    v_bias,
                    int(q_heads),
                    int(kv_heads),
                )
                return q, k, v

    return (
        split_heads(_bitnet_linear_from_transformed_input(qx, q_spec), q_heads),
        split_heads(_bitnet_linear_from_transformed_input(kx, k_spec), kv_heads),
        split_heads(_bitnet_linear_from_transformed_input(vx, v_spec), kv_heads),
    )


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
    if _should_use_eager_autograd_fallback(
        x,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
    ):
        return (
            linear(x, q_weight, q_bias, backend=backend),
            linear(x, k_weight, k_bias, backend=backend),
            linear(x, v_weight, v_bias, backend=backend),
        )
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


def resolve_packed_qkv_module_spec(
    q_module,
    k_module,
    v_module,
    *,
    backend: str | None = None,
    reference: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    resolved_backend = resolve_linear_backend(backend)
    if reference is not None:
        dtype = reference.dtype
        device = reference.device
    target_device = None if device is None else torch.device(device)

    q_spec = resolve_packed_linear_module_spec(q_module, backend=resolved_backend, reference=reference, dtype=dtype, device=target_device)
    k_spec = resolve_packed_linear_module_spec(k_module, backend=resolved_backend, reference=reference, dtype=dtype, device=target_device)
    v_spec = resolve_packed_linear_module_spec(v_module, backend=resolved_backend, reference=reference, dtype=dtype, device=target_device)

    if q_spec is None and k_spec is None and v_spec is None:
        if resolved_backend != "cublaslt":
            return None
        q_weight, q_bias = resolve_linear_module_tensors(q_module, reference=reference, dtype=dtype, device=target_device)
        k_weight, k_bias = resolve_linear_module_tensors(k_module, reference=reference, dtype=dtype, device=target_device)
        v_weight, v_bias = resolve_linear_module_tensors(v_module, reference=reference, dtype=dtype, device=target_device)
        packed_weight, packed_bias, q_size, k_size, v_size = pack_qkv_weights(
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
        )
        return {
            "format": "dense_qkv_packed",
            "backend": resolved_backend,
            "packed_weight": packed_weight,
            "packed_bias": packed_bias,
            "q_size": int(q_size),
            "k_size": int(k_size),
            "v_size": int(v_size),
        }

    if q_spec is None or k_spec is None or v_spec is None:
        return None

    formats = {str(q_spec.get("format")), str(k_spec.get("format")), str(v_spec.get("format"))}
    if formats == {"dense_packed"}:
        packed_bias_parts = []
        packed_bias_any = False
        for spec in (q_spec, k_spec, v_spec):
            bias = spec.get("bias")
            if bias is None:
                packed_bias_parts.append(torch.zeros(spec["packed_weight"].shape[0], device=spec["packed_weight"].device, dtype=spec["packed_weight"].dtype))
            else:
                packed_bias_any = True
                packed_bias_parts.append(bias.to(device=spec["packed_weight"].device, dtype=spec["packed_weight"].dtype).contiguous())
        return {
            "format": "dense_qkv_packed",
            "backend": resolved_backend,
            "packed_weight": torch.cat([q_spec["packed_weight"], k_spec["packed_weight"], v_spec["packed_weight"]], dim=0).contiguous(),
            "packed_bias": torch.cat(packed_bias_parts, dim=0).contiguous() if packed_bias_any else None,
            "q_size": int(q_spec["packed_weight"].shape[0]),
            "k_size": int(k_spec["packed_weight"].shape[0]),
            "v_size": int(v_spec["packed_weight"].shape[0]),
        }

    if formats == {"bitnet_w2a8"} and resolved_backend == "bitnet":
        return {
            "format": "bitnet_qkv",
            "backend": resolved_backend,
            "q_spec": q_spec,
            "k_spec": k_spec,
            "v_spec": v_spec,
            "q_size": int(q_spec["layout_header"][3].item()),
            "k_size": int(k_spec["layout_header"][3].item()),
            "v_size": int(v_spec["layout_header"][3].item()),
        }
    return None


def qkv_packed_spec_heads_projection(
    x: torch.Tensor,
    spec: dict,
    *,
    q_heads: int,
    kv_heads: int,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    format_name = str(spec.get("format"))
    if format_name == "dense_qkv_packed":
        return qkv_packed_heads_projection(
            x,
            spec["packed_weight"],
            spec.get("packed_bias"),
            q_size=int(spec["q_size"]),
            k_size=int(spec["k_size"]),
            v_size=int(spec["v_size"]),
            q_heads=q_heads,
            kv_heads=kv_heads,
            backend=backend,
        )
    if format_name == "bitnet_qkv":
        if not all(key in spec for key in ("q_spec", "k_spec", "v_spec")):
            raise NotImplementedError("Packed QKV format bitnet_qkv requires q_spec, k_spec, and v_spec payloads")
        return bitnet_qkv_packed_heads_projection(
            x,
            spec["q_spec"],
            spec["k_spec"],
            spec["v_spec"],
            q_heads=q_heads,
            kv_heads=kv_heads,
        )
    raise NotImplementedError(f"Packed QKV format {format_name} is not implemented")


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
    if _should_use_eager_autograd_fallback(x, packed_weight, packed_bias):
        fused = linear(x, packed_weight, packed_bias, backend=backend)
        return (
            split_heads(fused[..., :q_size], q_heads),
            split_heads(fused[..., q_size: q_size + k_size], kv_heads),
            split_heads(fused[..., q_size + k_size: q_size + k_size + v_size], kv_heads),
        )
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
    if _should_use_eager_autograd_fallback(
        x,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
    ):
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
    if _should_use_eager_autograd_fallback(x, weight, bias):
        return linear(merge_heads(x), weight, bias, backend=backend)
    if has_native_op("head_output_projection"):
        module = native_module()
        if module is not None and hasattr(module, "head_output_projection_forward"):
            return module.head_output_projection_forward(x, weight, bias, str(backend or "auto"))
    return linear(merge_heads(x), weight, bias, backend=backend)


def head_output_packed_projection(
    x: torch.Tensor,
    spec: dict,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    format_name = str(spec.get("format"))
    if format_name == "dense_packed":
        return head_output_projection(x, spec["packed_weight"], spec.get("bias"), backend=backend)
    if format_name == "bitnet_w2a8":
        return linear_from_packed_spec(merge_heads(x), spec, backend=backend)
    raise NotImplementedError(f"Packed output projection format {format_name} is not implemented")


def activation(
    x: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    act = str(activation).lower()
    if act not in {"gelu", "geglu", "silu", "swish", "swiglu", "gated-silu", "relu", "reglu"}:
        raise ValueError(f"Unsupported activation: {activation}")
    if _should_use_eager_autograd_fallback(x):
        if act in {"gelu", "geglu"}:
            return F.gelu(x)
        if act in {"silu", "swish", "swiglu", "gated-silu"}:
            return F.silu(x)
        return F.relu(x)
    if has_native_op("activation"):
        module = native_module()
        if module is not None and hasattr(module, "activation_forward"):
            return module.activation_forward(x, act)
    if act in {"gelu", "geglu"}:
        return F.gelu(x)
    if act in {"silu", "swish", "swiglu", "gated-silu"}:
        return F.silu(x)
    return F.relu(x)


def gated_activation(
    x: torch.Tensor,
    gate: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    act = str(activation).lower()
    if act not in {"gelu", "geglu", "silu", "swish", "swiglu", "gated-silu", "relu", "reglu"}:
        raise ValueError(f"Unsupported activation: {activation}")
    if x.shape != gate.shape:
        raise ValueError(f"gated_activation requires x and gate to have the same shape, got {tuple(x.shape)} and {tuple(gate.shape)}")
    if _should_use_eager_autograd_fallback(x, gate):
        return globals()["activation"](x, act) * gate
    if has_native_op("gated_activation"):
        module = native_module()
        if module is not None and hasattr(module, "gated_activation_forward"):
            return module.gated_activation_forward(torch.cat([x, gate], dim=-1), act)
    return globals()["activation"](x, act) * gate


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
    if _should_use_eager_autograd_fallback(x, w_in_weight, w_in_bias, w_out_weight, w_out_bias):
        hidden = linear(x, w_in_weight, w_in_bias, backend=backend)
        act = str(activation).lower()
        if gated:
            a, b = hidden.chunk(2, dim=-1)
            if act in ("swiglu", "gated-silu"):
                hidden = gated_activation(a, b, "silu")
            elif act == "geglu":
                hidden = gated_activation(a, b, "gelu")
            elif act == "reglu":
                hidden = gated_activation(a, b, "relu")
            else:
                hidden = gated_activation(a, b, "silu")
        else:
            if act in ("gelu",):
                hidden = globals()["activation"](hidden, "gelu")
            elif act in ("silu", "swish"):
                hidden = globals()["activation"](hidden, "silu")
            else:
                hidden = globals()["activation"](hidden, "gelu")
        return linear(hidden, w_out_weight, w_out_bias, backend=backend)
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
        hidden = gated_activation(a, b, act)
    else:
        hidden = globals()["activation"](hidden, act)
    return linear(hidden, w_out_weight, w_out_bias, backend=backend)


def mlp_module(
    x: torch.Tensor,
    w_in_module,
    w_out_module,
    *,
    activation: str,
    gated: bool,
    backend: str | None = None,
) -> torch.Tensor:
    runtime_linear_in = getattr(w_in_module, "runtime_linear", None)
    runtime_linear_out = getattr(w_out_module, "runtime_linear", None)
    if callable(runtime_linear_in) or callable(runtime_linear_out):
        hidden = linear_module(x, w_in_module, backend=backend)
        act = str(activation).lower()
        if gated:
            a, b = hidden.chunk(2, dim=-1)
            hidden = gated_activation(a, b, act)
        else:
            hidden = globals()["activation"](hidden, act)
        return linear_module(hidden, w_out_module, backend=backend)
    w_in_weight, w_in_bias = resolve_linear_module_tensors(w_in_module, reference=x)
    w_out_weight, w_out_bias = resolve_linear_module_tensors(w_out_module, reference=x)
    return mlp(
        x,
        w_in_weight,
        w_in_bias,
        w_out_weight,
        w_out_bias,
        activation=activation,
        gated=gated,
        backend=backend,
    )


def embedding(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int | None = None,
) -> torch.Tensor:
    if _should_use_eager_autograd_fallback(weight):
        return F.embedding(indices, weight, padding_idx=padding_idx)
    if has_native_op("embedding"):
        module = native_module()
        if module is not None and hasattr(module, "embedding_forward"):
            return module.embedding_forward(weight, indices, -1 if padding_idx is None else int(padding_idx))
    return F.embedding(indices, weight, padding_idx=padding_idx)
