import torch


def ensure_even_last_dim(x: torch.Tensor, name: str = "tensor"):
    if x.size(-1) % 2 != 0:
        raise ValueError(f"{name} last dimension must be even, got {x.size(-1)}")


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # x: (B, T, D) -> (B, H, T, Dh)
    B, T, D = x.shape
    if D % num_heads != 0:
        raise ValueError(f"Model dim {D} not divisible by heads {num_heads}")
    Dh = D // num_heads
    return x.view(B, T, num_heads, Dh).permute(0, 2, 1, 3).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh) -> (B, T, H*Dh)
    B, H, T, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)


def assert_mask_shape(mask: torch.Tensor, B: int, H: int, T: int, S: int):
    if tuple(mask.shape) != (B, H, T, S):
        raise ValueError(f"mask shape {tuple(mask.shape)} != ({B},{H},{T},{S})")


def assert_boolean_mask(mask: torch.Tensor):
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")


def assert_broadcastable(x: torch.Tensor, target_shape: tuple[int, ...]):
    xs = list(x.shape)
    ts = list(target_shape)
    while len(xs) < len(ts):
        xs = [1] + xs
    for a, b in zip(xs, ts):
        if a != 1 and a != b:
            raise ValueError(f"shape {tuple(x.shape)} is not broadcastable to {target_shape}")


def assert_same_dtype(*tensors: torch.Tensor):
    if not tensors:
        return
    d = tensors[0].dtype
    for t in tensors[1:]:
        if t.dtype != d:
            raise TypeError("Tensors must have the same dtype")


def assert_contiguous_lastdim(x: torch.Tensor):
    if not x.is_contiguous():
        # relax: ensure last dim contiguous
        if x.stride(-1) != 1:
            raise ValueError("Expected last dimension to be contiguous")


def ensure_contiguous_lastdim(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


def assert_sequence_lengths_match(mask: torch.Tensor, lengths: torch.Tensor):
    if mask.ndim < 2:
        raise ValueError("mask must be at least 2D (B,T,...)")
    B, T = mask.shape[0], mask.shape[1]
    if lengths.numel() != B:
        raise ValueError("lengths must have shape (B,)")
    # Optional: verify that mask agrees with lengths if it's a padding mask format


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -1, value: float | int = 0):
    size = x.size(dim)
    pad = (multiple - (size % multiple)) % multiple
    if pad == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad
    pad_tensor = x.new_full(pad_shape, value)
    return torch.cat([x, pad_tensor], dim=dim)


def left_pad(x: torch.Tensor, pad: int, dim: int = -2, value: float | int = 0):
    if pad <= 0:
        return x
    shape = list(x.shape)
    shape[dim] = pad
    filler = x.new_full(shape, value)
    return torch.cat([filler, x], dim=dim)


def right_trim_to(x: torch.Tensor, length: int, dim: int = -2):
    sl = [slice(None)] * x.ndim
    sl[dim] = slice(0, length)
    return x[tuple(sl)]


def reorder_to_channels_last_2d(x: torch.Tensor) -> torch.Tensor:
    # No-op for 2D tensors; for 4D NHWC/NCHW scenarios this is a placeholder
    return x


def stride_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.stride()) == tuple(b.stride())


def is_view_of(a: torch.Tensor, b: torch.Tensor) -> bool:
    try:
        return (a._is_view() and b._is_view() and a.storage().data_ptr() == b.storage().data_ptr()) or (
            a.storage().data_ptr() == b.storage().data_ptr()
        )
    except Exception:
        return False


def expect_shape(x: torch.Tensor, pattern: str):
    """Validate a minimal shape pattern like "... h d" against `x`.

    - "..." matches any prefix (possibly empty)
    - tokens count (excluding ellipsis) must not exceed x.ndim
    """
    tokens = [t for t in pattern.strip().split() if t]
    has_ellipsis = tokens and tokens[0] == "..."
    tail = tokens[1:] if has_ellipsis else tokens
    if len(tail) > x.ndim:
        raise ValueError(f"pattern {pattern!r} expects at least {len(tail)} dims, got {x.ndim}")
    # If no ellipsis, exact dims count must match
    if not has_ellipsis and len(tail) != x.ndim:
        raise ValueError(f"pattern {pattern!r} expects {len(tail)} dims, got {x.ndim}")
    return True


def same_shape(a: torch.Tensor, b: torch.Tensor, *more: torch.Tensor) -> bool:
    s = tuple(a.shape)
    if tuple(b.shape) != s:
        return False
    for t in more:
        if tuple(t.shape) != s:
            return False
    return True


def enforce_static_shape(x: torch.Tensor, spec: tuple[int, ...]) -> torch.Tensor:
    """Assert that x.shape matches spec where non-negative entries are fixed; -1 or 0 acts as wildcard.

    Returns x for chaining.
    """
    sx = tuple(x.shape)
    if len(sx) != len(spec):
        raise ValueError(f"Rank mismatch: {sx} vs {spec}")
    for a, b in zip(sx, spec):
        if b > 0 and a != b:
            raise ValueError(f"Shape mismatch: {sx} vs {spec}")
    return x


def trace_shape(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(s) for s in x.shape)


def expect_memory_format(x: torch.Tensor, fmt: str = "contig") -> bool:
    if fmt == "contig":
        if not x.is_contiguous():
            raise AssertionError("Expected contiguous tensor")
        return True
    if fmt == "channels_last":
        if x.ndim < 4:
            return True
        if not x.is_contiguous(memory_format=torch.channels_last):
            raise AssertionError("Expected channels_last format")
        return True
    return True


def window_partition(x: torch.Tensor, win: int) -> tuple[torch.Tensor, int]:
    # x: (B, T, ...) -> (B*n, win, ...), returns num_windows per batch item
    B, T = x.shape[0], x.shape[1]
    n = (T + win - 1) // win
    pad_T = n * win
    x_pad = pad_to_multiple(x, win, dim=1)
    xw = x_pad.view(B, n, win, *x.shape[2:]).reshape(B * n, win, *x.shape[2:])
    return xw, n


def window_merge(xw: torch.Tensor, win: int, B: int, T: int):
    n = (T + win - 1) // win
    return xw.view(B, n, win, *xw.shape[2:]).reshape(B, n * win, *xw.shape[2:])[:, :T]


def segment_sum(x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(int(segments.max().item()) + 1, *x.shape[1:], device=x.device, dtype=x.dtype)
    out.index_add_(0, segments, x)
    return out


def segment_mean(x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    sums = segment_sum(x, segments)
    counts = torch.bincount(segments, minlength=sums.shape[0]).clamp_min(1).view(-1, *([1] * (x.ndim - 1)))
    return sums / counts


def segment_max(x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    out = torch.full((int(segments.max().item()) + 1, *x.shape[1:]), -float('inf'), device=x.device, dtype=x.dtype)
    out = out.index_reduce(0, segments, x, reduce="amax")
    return out


def segment_min(x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    out = torch.full((int(segments.max().item()) + 1, *x.shape[1:]), float('inf'), device=x.device, dtype=x.dtype)
    out = out.index_reduce(0, segments, x, reduce="amin")
    return out


def split_qkv(x: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: (B, T, 3*D) -> q,k,v each (B,H,T,Dh)
    B, T, threeD = x.shape
    D = threeD // 3
    q, k, v = x.split(D, dim=-1)
    return split_heads(q, num_heads), split_heads(k, num_heads), split_heads(v, num_heads)


def merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q,k,v: (B,H,T,Dh) -> (B,T,3*D)
    q_ = merge_heads(q)
    k_ = merge_heads(k)
    v_ = merge_heads(v)
    return torch.cat([q_, k_, v_], dim=-1)


def split_gqa_heads(q: torch.Tensor, k: torch.Tensor, num_q_heads: int, num_kv_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    # q: (B,T,Dq) -> (B,Hq,T,Dhq), k: (B,T,Dk) -> (B,Hkv,T,Dhkv)
    qh = split_heads(q, num_q_heads)
    kh = split_heads(k, num_kv_heads)
    return qh, kh


def merge_gqa_heads(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q: (B,Hq,T,Dhq), k: (B,Hkv,T,Dhkv) -> (B,T,Dq), (B,T,Dk)
    return merge_heads(q), merge_heads(k)


def center_pad(x: torch.Tensor, total: int, dim: int = -2, value: float | int = 0):
    size = x.size(dim)
    if total <= size:
        return x
    left = (total - size) // 2
    right = total - size - left
    shape = list(x.shape)
    shape[dim] = left
    left_pad_tensor = x.new_full(shape, value)
    shape[dim] = right
    right_pad_tensor = x.new_full(shape, value)
    return torch.cat([left_pad_tensor, x, right_pad_tensor], dim=dim)


def bhdt_to_bthd(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("bhdt_to_bthd expects a 4D tensor (B,H,D,T)")
    return x.permute(0, 3, 1, 2).contiguous()


def contiguous_lastdim(x: torch.Tensor) -> torch.Tensor:
    return ensure_contiguous_lastdim(x)


def assert_shape(x: torch.Tensor, pattern: str) -> bool:
    return expect_shape(x, pattern)


def assert_mask(mask: torch.Tensor, shape: tuple[int, int, int, int]) -> None:
    assert_mask_shape(mask, *shape)


def pack_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack heads into groups: (B,H,T,Dh) -> (B,G,T,H//G,Dh)."""
    if groups <= 0:
        raise ValueError("groups must be > 0")
    def _pack(t: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = t.shape
        if H % groups != 0:
            raise ValueError("H not divisible by groups")
        G = groups
        Hg = H // G
        return t.view(B, G, Hg, T, Dh).permute(0, 1, 3, 2, 4).contiguous()
    return _pack(q), _pack(k), _pack(v)


def unpack_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of pack_heads: (B,G,T,Hg,Dh) -> (B,H,T,Dh)."""
    def _unpack(t: torch.Tensor) -> torch.Tensor:
        B, G, T, Hg, Dh = t.shape
        H = G * Hg
        return t.permute(0, 1, 3, 2, 4).contiguous().view(B, H, T, Dh)
    return _unpack(q), _unpack(k), _unpack(v)


def rechunk(x: torch.Tensor, block: int, dim: int = -2) -> torch.Tensor:
    """Partition along dim into blocks of size ``block`` and flatten batch blocks.

    Returns shape with dim split into (n_blocks*B, block, ...) similar to window_partition for sequences.
    """
    if block <= 0:
        return x
    d = dim if dim >= 0 else x.ndim + dim
    if d != 1:
        # generic: pad and reshape along arbitrary dim
        size = x.size(d)
        n = (size + block - 1) // block
        pad = n * block - size
        if pad:
            pad_shape = list(x.shape)
            pad_shape[d] = pad
            filler = x.new_zeros(pad_shape)
            x = torch.cat([x, filler], dim=d)
        shape = list(x.shape)
        shape[d] = n
        shape.insert(d + 1, block)
        y = x.view(*shape)
        # flatten the new leading block dimension into batch-like dim
        return y
    # fast path for (B,T,...) along dim=1
    from .windows import window_partition
    B, T = x.shape[0], x.shape[1]
    y, _ = window_partition(x, block)
    return y


# Symbolic shapes & simple inference stubs
def S(spec: str):
    """Parse a simple symbolic shape spec like "B,H,T,D" into a tuple of strings."""
    return tuple(t.strip() for t in spec.replace(" ", "").split(",") if t.strip())


def unify(a, b):
    """Unify two symbolic shape tuples; returns a merged tuple and notes.

    This is a placeholder emitting a human-readable description.
    """
    ta = tuple(a)
    tb = tuple(b)
    n = min(len(ta), len(tb))
    merged = []
    notes = []
    for i in range(n):
        if ta[i] == tb[i]:
            merged.append(ta[i])
        else:
            merged.append(f"{ta[i]}|{tb[i]}")
            notes.append(f"dim[{i}]: {ta[i]} != {tb[i]}")
    merged += list(ta[n:]) + list(tb[n:])
    return tuple(merged), ("; ".join(notes) if notes else "consistent")


def graph_shapes(*tensors: torch.Tensor) -> str:
    """Emit a human-readable summary of shapes for debugging."""
    lines = [f"tensor[{i}]: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}" for i, t in enumerate(tensors)]
    return "\n".join(lines)


# ---- Symbolic inference and broadcast planning ----
def _parse_eq(eq: str) -> tuple[list[str], list[str]]:
    eq = eq.replace(" ", "")
    left, right = eq.split("==")
    return left.split("*"), right.split("*")


def infer(eq: str | list[str], given: dict[str, int]) -> tuple[dict[str, int], str]:
    """Solve simple product equalities for one unknown symbol using given bindings.

    Example: infer("B*H==G*Q", {"B":2,"H":8,"G":4}) -> {"Q":4}, derivation
    """
    eqs = [eq] if isinstance(eq, str) else list(eq)
    solved: dict[str, int] = {}
    steps: list[str] = []
    env = {k: int(v) for k, v in given.items()}
    for e in eqs:
        L, R = _parse_eq(e)
        def prod(side: list[str]) -> tuple[int, list[str]]:
            unknowns: list[str] = []
            p = 1
            for s in side:
                if s.isnumeric():
                    p *= int(s)
                elif s in env:
                    p *= int(env[s])
                else:
                    unknowns.append(s)
            return p, unknowns
        pl, ul = prod(L)
        pr, ur = prod(R)
        if len(ul) + len(ur) != 1:
            steps.append(f"skip {e}: expected single unknown, got ul={ul}, ur={ur}")
            continue
        if len(ul) == 1:
            sym = ul[0]
            val = (pr // max(pl, 1)) if pl != 0 else 0
            env[sym] = int(val)
            solved[sym] = int(val)
            steps.append(f"{e} => {sym}={(pr)}/{max(pl,1)} -> {val}")
        else:
            sym = ur[0]
            val = (pl // max(pr, 1)) if pr != 0 else 0
            env[sym] = int(val)
            solved[sym] = int(val)
            steps.append(f"{e} => {sym}={(pl)}/{max(pr,1)} -> {val}")
    deriv = "\n".join(steps) if steps else "no solvable equations"
    return solved, deriv


def broadcast_plan(sig_a: str, sig_b: str) -> tuple[dict[str, str], str]:
    """Plan broadcasting between two symbolic signatures.

    Returns mapping {dim: 'match|expand_a|expand_b'} and a derivation string.
    """
    A = [t for t in sig_a.replace(" ", "").split(",") if t]
    B = [t for t in sig_b.replace(" ", "").split(",") if t]
    n = max(len(A), len(B))
    A = ["1"] * (n - len(A)) + A
    B = ["1"] * (n - len(B)) + B
    plan: dict[str, str] = {}
    steps: list[str] = []
    for i, (a, b) in enumerate(zip(A, B)):
        key = f"dim[{i}]"
        if a == b:
            plan[key] = "match"
            steps.append(f"{key}: {a} == {b} -> match")
        elif a == "1":
            plan[key] = "expand_a"
            steps.append(f"{key}: a=1, b={b} -> expand_a")
        elif b == "1":
            plan[key] = "expand_b"
            steps.append(f"{key}: b=1, a={a} -> expand_b")
        else:
            plan[key] = "incompatible"
            steps.append(f"{key}: {a} vs {b} -> incompatible")
    return plan, "\n".join(steps)


# ---- Layout helpers ----
def tile(x: torch.Tensor, reps: tuple[object, ...]) -> torch.Tensor:
    """Tile tensor with symbol-friendly reps; strings are treated as no-op (1x).

    Example: tile(x, ("B", 2, "T", 1)) -> repeats=(1,2,1,1)
    """
    r: list[int] = []
    for rep in reps:
        if isinstance(rep, int):
            r.append(int(rep))
        else:
            r.append(1)
    return x.repeat(*r)


def reorder(x: torch.Tensor, policy: str = "batch_first") -> torch.Tensor:
    """Canonical reorder; for now, batch_first is identity for (B, ...)."""
    return x


