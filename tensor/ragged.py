import torch


def pack_sequences(x: torch.Tensor, lengths: torch.Tensor):
    # Sort batch by lengths desc for pack-friendly layout
    idx = torch.argsort(lengths, descending=True)
    rev = torch.empty_like(idx)
    rev[idx] = torch.arange(idx.numel(), device=idx.device)
    packed = x.index_select(0, idx)
    return packed, idx, rev


def unpack_sequences(packed: torch.Tensor, rev_idx: torch.Tensor, pad_to: int | None = None):
    x = packed.index_select(0, rev_idx)
    if pad_to is not None and x.shape[-2] != pad_to:
        pad = pad_to - x.shape[-2]
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        else:
            x = x[..., :pad_to, :]
    return x


def segment_sum(x: torch.Tensor, segments: torch.Tensor, n_segments: int | None = None) -> torch.Tensor:
    K = int(n_segments) if n_segments is not None else int(segments.max().item()) + 1
    out = torch.zeros((K, *x.shape[1:]), device=x.device, dtype=x.dtype)
    out.index_add_(0, segments, x)
    return out


def segment_mean(x: torch.Tensor, segments: torch.Tensor, n_segments: int | None = None) -> torch.Tensor:
    sums = segment_sum(x, segments, n_segments=n_segments)
    counts = torch.bincount(segments, minlength=sums.shape[0]).clamp_min(1)
    while counts.ndim < sums.ndim:
        counts = counts.unsqueeze(-1)
    return sums / counts


def segment_max(x: torch.Tensor, segments: torch.Tensor, n_segments: int | None = None) -> torch.Tensor:
    K = int(n_segments) if n_segments is not None else int(segments.max().item()) + 1
    out = torch.full((K, *x.shape[1:]), -float("inf"), device=x.device, dtype=x.dtype)
    out = out.index_reduce(0, segments, x, reduce="amax")
    return out


def segment_min(x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    K = int(segments.max().item()) + 1
    out = torch.full((K, *x.shape[1:]), float("inf"), device=x.device, dtype=x.dtype)
    out = out.index_reduce(0, segments, x, reduce="amin")
    return out


# Packed attention helpers
def pack_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = q.shape[0]
    offs = torch.cat([torch.tensor([0], device=lengths.device), torch.cumsum(lengths, dim=0)])
    q_list, k_list, v_list = [], [], []
    for b in range(B):
        t = int(lengths[b].item())
        q_list.append(q[b, :t])
        k_list.append(k[b, :t])
        v_list.append(v[b, :t])
    return torch.cat(q_list, dim=0), torch.cat(k_list, dim=0), torch.cat(v_list, dim=0), offs


def unpack_qkv(qp: torch.Tensor, kp: torch.Tensor, vp: torch.Tensor, offsets: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = offsets.numel() - 1
    out_q = qp.new_zeros(B, T, *qp.shape[1:])
    out_k = kp.new_zeros(B, T, *kp.shape[1:])
    out_v = vp.new_zeros(B, T, *vp.shape[1:])
    for b in range(B):
        s, e = int(offsets[b].item()), int(offsets[b + 1].item())
        t = e - s
        out_q[b, :t] = qp[s:e]
        out_k[b, :t] = kp[s:e]
        out_v[b, :t] = vp[s:e]
    return out_q, out_k, out_v


def packed_softmax(logits: torch.Tensor, lengths: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=dim).to(dtype=logits.dtype)


# Block-wise ragged reductions (GPU-friendly skeletons)
def ragged_block_sum(x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    return segment_sum(x, segments)


def ragged_block_mean(x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    return segment_mean(x, segments)


def ragged_block_max(x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    return segment_max(x, segments)


def packed_logsumexp(x: torch.Tensor, lengths: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.logsumexp(x.float(), dim=dim).to(dtype=x.dtype)


def ragged_inclusive_cumsum(values: torch.Tensor, lengths: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # values are concatenated sequences; lengths specify per-sequence sizes along dim
    return values.cumsum(dim=dim)


def ragged_exclusive_cumsum(values: torch.Tensor, lengths: torch.Tensor, dim: int = 0) -> torch.Tensor:
    cs = values.cumsum(dim=dim)
    shift = torch.zeros_like(values)
    idx = 0
    for l in lengths.tolist():
        if l > 0:
            sl = [slice(None)] * values.ndim
            sl[dim] = slice(idx, idx + l)
            s2 = [slice(None)] * values.ndim
            s2[dim] = slice(idx, idx + l)
            shift[tuple(sl)] = torch.cat([torch.zeros_like(values[tuple(s2)][...:1]), cs[tuple(s2)][...:-1]], dim=dim)
        idx += l
    return shift


def ragged_scatter(src: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor, T: int) -> torch.Tensor:
    """Inverse of ragged_gather: scatter back into (B,T,...) with zero fill.

    src: (sum(lengths), ...); offsets/lengths: (B,); returns (B,T,...)
    """
    B = int(lengths.numel())
    out = src.new_zeros((B, T, *src.shape[1:]))
    for b in range(B):
        l = int(lengths[b].item())
        if l <= 0:
            continue
        s = int(offsets[b].item())
        e = s + l
        out[b, :l] = src[s:e]
    return out


def ragged_gather(x: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Gather variable-length segments from a flat tensor into a padded batch.

    - x: (N, ...)
    - offsets, lengths: (B,) start indices and lengths for each segment in x
    Returns: (B, max_len, ...) zero-padded tensor
    """
    B = int(lengths.numel())
    max_len = int(lengths.max().item()) if B > 0 else 0
    out_shape = (B, max_len, *x.shape[1:])
    out = x.new_zeros(out_shape)
    for b in range(B):
        l = int(lengths[b].item())
        if l <= 0:
            continue
        s = int(offsets[b].item())
        e = s + l
        out[b, :l] = x[s:e]
    return out

