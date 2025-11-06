import torch


def build_causal_mask(seq_len: int, device=None, dtype=torch.bool) -> torch.Tensor:
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    return mask.to(dtype)


def build_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: (B, T) where 1=token, 0=pad
    return attention_mask == 0


def apply_mask(attn_scores: torch.Tensor, mask: torch.Tensor, fill_value: float = -1e9) -> torch.Tensor:
    # attn_scores: (B, H, T, S) or (B, T, S)
    return attn_scores.masked_fill(mask, fill_value)


def broadcast_mask(
    *,
    batch_size: int,
    num_heads: int,
    tgt_len: int,
    src_len: int,
    causal_mask: torch.Tensor | None = None,  # (T,S) or (T,T)
    padding_mask: torch.Tensor | None = None,  # (B,S) where True is PAD or input 0/1
    padding_mask_is_1_for_token: bool = True,
) -> torch.Tensor:
    """Return a boolean mask broadcastable to (B,H,T,S).

    - causal_mask: True means masked; shape (T,S).
    - padding_mask: if shape (B,S) and is 1 for tokens, convert to True for pads.
    """
    device = None
    masks = []
    if causal_mask is not None:
        device = causal_mask.device
        cm = causal_mask.to(torch.bool).view(1, 1, tgt_len, src_len)
        masks.append(cm)
    if padding_mask is not None:
        device = padding_mask.device
        pm = padding_mask
        if pm.dtype != torch.bool:
            pm = pm == (0 if padding_mask_is_1_for_token else 1)
        pm = pm.view(batch_size, 1, 1, src_len)  # (B,1,1,S)
        masks.append(pm)
    if not masks:
        return torch.zeros(batch_size, num_heads, tgt_len, src_len, dtype=torch.bool, device=device)
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m  # True means masked
    return combined.expand(batch_size, num_heads, tgt_len, src_len)


def build_sliding_window_causal_mask(seq_len: int, window_size: int, device=None, dtype=torch.bool) -> torch.Tensor:
    # True means masked. Allow attention only to last `window_size` tokens including self, and no future.
    i = torch.arange(seq_len, device=device).view(seq_len, 1)
    j = torch.arange(seq_len, device=device).view(1, seq_len)
    too_far_past = j < (i - (window_size - 1))
    future = j > i
    mask = too_far_past | future
    return mask.to(dtype)


def build_prefix_lm_mask(seq_len: int, prefix_len: int, device=None, dtype=torch.bool) -> torch.Tensor:
    # True means masked. Prefix tokens see only prefix; continuation sees prefix and past continuation, not future
    i = torch.arange(seq_len, device=device).view(seq_len, 1)
    j = torch.arange(seq_len, device=device).view(1, seq_len)
    future = j > i
    # tokens in prefix cannot see continuation
    prefix_row = i < prefix_len
    cont_col = j >= prefix_len
    mask = torch.where(prefix_row, cont_col, future)
    return mask.to(dtype)


def attention_mask_from_lengths(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    # lengths: (B,) int -> attention_mask (B, T) with 1 for tokens, 0 for pads
    B = lengths.numel()
    T = int(max_len) if max_len is not None else int(lengths.max().item())
    rng = torch.arange(T, device=lengths.device).view(1, T)
    return (rng < lengths.view(B, 1)).to(torch.long)


def lengths_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: (B, T) with 1 token, 0 pad
    return attention_mask.to(torch.long).sum(dim=-1)


def pack_sequences(x: torch.Tensor, lengths: torch.Tensor):
    # x: (B, T, ...), lengths: (B,) -> packed: (sumL, ...), idx and rev_idx for unpack
    B, T = x.shape[0], x.shape[1]
    device = x.device
    idx_list = []
    for b in range(B):
        L = int(lengths[b].item())
        idx_list.append(torch.stack([torch.full((L,), b, device=device), torch.arange(L, device=device)], dim=1))
    idx = torch.cat(idx_list, dim=0)  # (sumL, 2)
    packed = x[idx[:, 0], idx[:, 1]]
    # rev_idx maps back to (B,T) with -1 for pads
    rev_idx = torch.full((B, T), -1, device=device, dtype=torch.long)
    cursor = 0
    for b in range(B):
        L = int(lengths[b].item())
        if L > 0:
            rev_idx[b, :L] = torch.arange(cursor, cursor + L, device=device)
        cursor += L
    return packed, idx, rev_idx


def unpack_sequences(packed: torch.Tensor, rev_idx: torch.Tensor, pad_to: int | None = None):
    # rev_idx: (B,T) with -1 for pads
    B, T = rev_idx.shape
    T_out = pad_to if pad_to is not None else T
    out_shape = (B, T_out) + packed.shape[1:]
    out = packed.new_zeros(out_shape)
    mask = rev_idx >= 0
    out[mask] = packed[rev_idx[mask]]
    return out


def make_block_indices(lengths: torch.Tensor, block_size: int):
    # Return list of (start,end) per example for blockwise attention windows
    blocks = []
    for L in lengths.tolist():
        spans = []
        for s in range(0, int(L), block_size):
            e = min(s + block_size, int(L))
            spans.append((s, e))
        blocks.append(spans)
    return blocks


def build_block_causal_mask(seq_len: int, block: int, device=None, dtype=torch.bool) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    for s in range(0, seq_len, block):
        e = min(s + block, seq_len)
        # within block: causal
        block_mask = torch.triu(torch.ones(e - s, e - s, device=device, dtype=torch.bool), diagonal=1)
        mask[s:e, s:e] = block_mask
        # prevent cross-block (no attention to previous blocks beyond immediate?) keep simple: block-local
        mask[s:e, :s] = True
        mask[s:e, e:] = True
    return mask.to(dtype)


def build_dilated_causal_mask(seq_len: int, window: int, dilation: int, device=None, dtype=torch.bool) -> torch.Tensor:
    i = torch.arange(seq_len, device=device).view(seq_len, 1)
    j = torch.arange(seq_len, device=device).view(1, seq_len)
    future = j > i
    dist = i - j
    too_far_past = dist > (window - 1) * dilation
    off_dilation = (dist % dilation) != 0
    mask = future | too_far_past | off_dilation
    return mask.to(dtype)


def combine_masks(masks: list[torch.Tensor], target_shape: tuple[int, ...] | None = None) -> torch.Tensor:
    # Broadcast OR across masks; enforce final shape if provided
    if not masks:
        raise ValueError("No masks provided")
    combined = None
    for m in masks:
        if m is None:
            continue
        m_bool = m.to(torch.bool)
        combined = m_bool if combined is None else (combined | m_bool)
    if combined is None:
        raise ValueError("All masks were None")
    if target_shape is not None and tuple(combined.shape) != tuple(target_shape):
        combined = combined.expand(*target_shape)
    return combined


def build_banded_mask(seq_len: int, bandwidth: int, device=None, dtype=torch.bool) -> torch.Tensor:
    # Allow only |i-j| <= bandwidth; everything else masked
    i = torch.arange(seq_len, device=device).view(seq_len, 1)
    j = torch.arange(seq_len, device=device).view(1, seq_len)
    return (torch.abs(i - j) > bandwidth).to(dtype)


def build_strided_mask(seq_len: int, stride: int, device=None, dtype=torch.bool) -> torch.Tensor:
    # Allow tokens to see positions with j % stride == i % stride and j <= i
    i = torch.arange(seq_len, device=device).view(seq_len, 1)
    j = torch.arange(seq_len, device=device).view(1, seq_len)
    future = j > i
    modulo = (i % stride) != (j % stride)
    return (future | modulo).to(dtype)


def build_segment_bidir_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    # segment_ids: (B, T) ints; mask positions where segments differ (bidirectional within segment)
    B, T = segment_ids.shape
    seg_i = segment_ids.unsqueeze(-1).expand(B, T, T)
    seg_j = segment_ids.unsqueeze(-2).expand(B, T, T)
    return (seg_i != seg_j)


def window_pattern_from_spans(spans: list[tuple[int, int]], T: int) -> torch.Tensor:
    """Build a boolean mask (T,T) where True indicates masked.

    Spans are inclusive ranges [start, end) that are allowed; others masked.
    """
    allow = torch.zeros(T, T, dtype=torch.bool)
    for s, e in spans:
        s0 = max(int(s), 0)
        e0 = min(int(e), T)
        if e0 > s0:
            allow[s0:e0, s0:e0] = True
    return ~allow


def invert_mask(mask: torch.Tensor) -> torch.Tensor:
    return ~mask.to(torch.bool)


def as_bool_mask(x: torch.Tensor, true_means: str = "masked") -> torch.Tensor:
    # convert int/float masks to bool; true_means="masked" or "keep"
    if x.dtype == torch.bool:
        return x
    if true_means not in ("masked", "keep"):
        raise ValueError("true_means must be 'masked' or 'keep'")
    if x.dtype.is_floating_point or x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        b = x != 0
        return b if true_means == "keep" else (~b)
    raise TypeError("Unsupported mask dtype")


def to_additive_mask(bool_mask: torch.Tensor, neg_inf_value: float = float('-inf')) -> torch.Tensor:
    # True (masked) -> -inf, False -> 0 (float32; callers may cast to attn dtype)
    x = bool_mask.to(torch.float32)
    if neg_inf_value == float('-inf'):
        # Multiply by -inf via where to avoid NaNs on 0 * inf
        return torch.where(x > 0, torch.full_like(x, float('-inf')), torch.zeros_like(x))
    return x.mul(neg_inf_value)


def apply_additive_mask_(scores: torch.Tensor, add_mask: torch.Tensor) -> torch.Tensor:
    # In-place add; assumes broadcastable shapes
    scores.add_(add_mask.to(dtype=scores.dtype))
    return scores


def build_block_sparse_mask(seq_len: int, block: int, pattern: torch.Tensor, device=None, dtype=torch.bool) -> torch.Tensor:
    """Build a block-sparse mask from a block connectivity pattern.

    pattern: (N,N) with True meaning masked or 0/1 connectivity? Here we interpret
    pattern[bq, bk] == 1 as ALLOW attention from block bq to block bk (within causal), else masked.
    """
    N = (seq_len + block - 1) // block
    pat = pattern.to(torch.bool)
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    for bq in range(N):
        qs, qe = bq * block, min((bq + 1) * block, seq_len)
        for bk in range(N):
            ks, ke = bk * block, min((bk + 1) * block, seq_len)
            if pat[bq, bk]:
                # allow within block region (unmask), respecting causal
                sub = torch.triu(torch.ones(qe - qs, ke - ks, device=device, dtype=torch.bool), diagonal=max(0, ks - qs))
                # sub True = masked; we want to allow -> set False where allowed
                mask[qs:qe, ks:ke] = mask[qs:qe, ks:ke] & sub
            # else keep masked
    return mask.to(dtype)


def create_causal_mask(
    *,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    cache_position: torch.Tensor | None = None,
    past_key_values=None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """HF-like 4D additive causal mask builder handling cached keys.

    - Returns additive mask shaped (B, 1, T, S) where S is total key length (past + present).
    - Uses absolute `position_ids` or `cache_position` to determine S and future masking.
    - Combines causal and padding masks. If `attention_mask` is shorter than S, right-pad with ones (keep).
    """
    B, T, _ = input_embeds.shape
    device = input_embeds.device

    # Determine total key length S from position context
    S = T
    if position_ids is not None:
        pos = position_ids
        if pos.dim() == 2:
            pos = pos[0]
        try:
            S = int(pos.max().item()) + 1
        except Exception:
            S = max(T, 1)
    elif cache_position is not None:
        cp = cache_position
        if cp.dim() > 0:
            try:
                S = int(cp.max().item()) + 1
            except Exception:
                S = max(T, 1)
    # If an attention_mask is provided and longer than S, expand S to match
    if attention_mask is not None:
        try:
            am_len = int(attention_mask.shape[-1])
            if am_len > S:
                S = am_len
        except Exception:
            pass

    # Build future mask using absolute positions if provided
    if position_ids is not None:
        pos = position_ids
        if pos.dim() == 2:
            pos = pos[0]
        qpos = pos.view(T, 1)  # (T,1)
        kpos = torch.arange(S, device=device, dtype=qpos.dtype).view(1, S)  # (1,S)
        future = kpos > qpos  # (T,S)
        causal_bool = future.view(1, 1, T, S)
    else:
        # Fallback to standard causal when absolute positions are unavailable
        causal_bool = build_causal_mask(S, device=device, dtype=torch.bool)  # (S,S)
        # Take the last T rows as queries
        causal_bool = causal_bool[-T:].view(1, 1, T, S)

    # Padding mask (True means masked)
    if attention_mask is not None:
        pm = attention_mask
        # Right-pad to length S with ones (keep)
        if pm.shape[-1] < S:
            pad_len = S - pm.shape[-1]
            pad = torch.ones(pm.shape[0], pad_len, dtype=pm.dtype, device=pm.device)
            pm = torch.cat([pm, pad], dim=-1)
        elif pm.shape[-1] > S:
            # Truncate to match S if longer
            pm = pm[..., :S]
        pad_bool = build_padding_mask(pm).view(B, 1, 1, S)
        combined_bool = causal_bool | pad_bool
    else:
        combined_bool = causal_bool

    # Convert to additive mask (neg_inf where masked)
    add = to_additive_mask(combined_bool)
    return add


