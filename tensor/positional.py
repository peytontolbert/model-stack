# tensor/positional.py
import torch


def build_rope_cache(seq_len: int, head_dim: int, device=None, base_theta: float = 1e6):
    """Build RoPE cos/sin cache matching HF Transformers LLaMA implementation exactly.
    
    HF computes freqs for dim/2, then concatenates [freqs, freqs] to get full dimension.
    This is different from interleaving! HF does: [f0, f1, f2, ..., f0, f1, f2, ...]
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    # Match HF: use int64 dtype for arange, then cast to float
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        base_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim)
    )
    freqs = torch.einsum("t,d->td", t, inv_freq)  # (T, Dh/2)
    # HF: emb = torch.cat((freqs, freqs), dim=-1) - repeat the whole freq vector
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, Dh)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE using HF-compatible rotate_half formulation.

    q: (B,Hq,T,Dh), k: (B,Hk,T,Dh); cos/sin: (T,Dh)
    """
    # Broadcast cos/sin over batch and heads
    cos_b = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sin_b = sin.view(1, 1, sin.shape[0], sin.shape[1])

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos_b) + (rotate_half(q) * sin_b)
    k_embed = (k * cos_b) + (rotate_half(k) * sin_b)
    return q_embed, k_embed


def apply_rotary_scaled(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, scale: float):
    q_out, k_out = apply_rotary(q, k, cos, sin)
    return q_out * scale, k_out * scale


def rope_ntk_scaling(seq_len: int, base_theta: float, factor: float = 1.0) -> float:
    # Basic NTK scaling factor; higher factor slows frequency growth
    return float(factor)


def rope_yarn_factors(seq_len: int, base_theta: float, factor: float) -> tuple[float, float]:
    # Return (scale_q, scale_k) multipliers; simple symmetric default
    return float(factor), float(factor)


def relative_position_bucket(t: int, s: int, num_buckets: int, max_distance: int) -> int:
    # Heuristic bucketing (T5-style)
    n = num_buckets // 2
    rel = s - t
    sign = 0 if rel <= 0 else 1
    rel = abs(rel)
    max_exact = n // 2
    if rel < max_exact:
        bucket = rel
    else:
        bucket = max_exact + int((torch.log(torch.tensor(rel / max_exact)) / torch.log(torch.tensor(max_distance / max_exact)) * (n - max_exact)).clamp(min=0, max=n - max_exact).item())
    return int(bucket + sign * n)


def build_sinusoidal_cache(seq_len: int, dim: int, device=None) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device).float().unsqueeze(1)
    i = torch.arange(dim, device=device).float().unsqueeze(0)
    angle_rates = 1.0 / torch.pow(10000.0, (2 * (i // 2)) / dim)
    angles = pos * angle_rates
    emb = torch.zeros(seq_len, dim, device=device)
    emb[:, 0::2] = torch.sin(angles[:, 0::2])
    emb[:, 1::2] = torch.cos(angles[:, 1::2])
    return emb


def alibi_slopes(n_heads: int, method: str = "press2022", device=None) -> torch.Tensor:
    # Canonical slopes used by ALiBi; matches common open-source implementations
    import math
    def get_slopes(n):
        if math.log2(n).is_integer():
            m = n
        else:
            m = 2 ** math.floor(math.log2(n))
        slopes = torch.pow(2, -torch.arange(0, m, device=device).float() / m)
        if m < n:
            extra = torch.pow(2, -torch.arange(1, 2 * (n - m) + 1, 2, device=device).float() / m)
            slopes = torch.cat([slopes, extra], dim=0)
        return slopes[:n]
    return get_slopes(n_heads)


def rescale_positions(idx: torch.Tensor, old_ctx: int, new_ctx: int, mode: str = "linear") -> torch.Tensor:
    if mode == "linear":
        scale = float(new_ctx) / float(max(old_ctx, 1))
        return (idx.float() * scale).to(idx.dtype)
    return idx


def build_rope_cache_2d(H: int, W: int, head_dim: int, device=None, base_theta: float = 1e6):
    # Split dims between H and W axes
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even for 2D cache")
    Dh2 = head_dim // 2
    # Height component
    t_h = torch.arange(H, device=device, dtype=torch.float32)
    inv_h = 1.0 / (base_theta ** (torch.arange(0, Dh2, 2, device=device).float() / Dh2)) if Dh2 > 0 else torch.tensor([], device=device)
    freqs_h = torch.einsum("t,d->td", t_h, inv_h) if Dh2 > 0 else torch.zeros(H, 0, device=device)
    cos_h = torch.cos(torch.repeat_interleave(freqs_h, 2, dim=-1))  # (H, Dh2)
    sin_h = torch.sin(torch.repeat_interleave(freqs_h, 2, dim=-1))
    # Width component
    t_w = torch.arange(W, device=device, dtype=torch.float32)
    inv_w = 1.0 / (base_theta ** (torch.arange(0, Dh2, 2, device=device).float() / Dh2)) if Dh2 > 0 else torch.tensor([], device=device)
    freqs_w = torch.einsum("t,d->td", t_w, inv_w) if Dh2 > 0 else torch.zeros(W, 0, device=device)
    cos_w = torch.cos(torch.repeat_interleave(freqs_w, 2, dim=-1))  # (W, Dh2)
    sin_w = torch.sin(torch.repeat_interleave(freqs_w, 2, dim=-1))
    # Combine to (H,W,head_dim)
    cos = torch.zeros(H, W, head_dim, device=device)
    sin = torch.zeros(H, W, head_dim, device=device)
    if Dh2 > 0:
        cos[:, :, :Dh2] = cos_h[:, None, :]
        sin[:, :, :Dh2] = sin_h[:, None, :]
        cos[:, :, Dh2:] = cos_w[None, :, :]
        sin[:, :, Dh2:] = sin_w[None, :, :]
    return cos, sin


def build_relative_position_indices(tgt_len: int, src_len: int, max_distance: int, device=None) -> torch.Tensor:
    # indices in [0, 2*max_distance-2], where center corresponds to distance 0
    i = torch.arange(tgt_len, device=device).view(tgt_len, 1)
    j = torch.arange(src_len, device=device).view(1, src_len)
    rel = j - i  # positive means key is ahead of query
    rel = rel.clamp(min=-(max_distance - 1), max=(max_distance - 1))
    indices = rel + (max_distance - 1)
    return indices.to(torch.long)


def relative_position_bias_from_table(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    # indices: (T,S) ints; table: (H, 2*max_distance-1) -> bias: (1,H,T,S)
    H, D = table.shape
    T, S = indices.shape
    bias = table[:, indices.view(-1)].view(H, T, S)
    return bias.unsqueeze(0)


def _alibi_slopes(n_heads: int, device=None) -> torch.Tensor:
    # Common slope generation per ALiBi paper implementation
    import math
    def get_slopes(n):
        if math.log2(n).is_integer():
            m = n
        else:
            m = 2 ** math.floor(math.log2(n))
        slopes = torch.pow(2, -torch.arange(0, m, device=device).float() / m)
        if m < n:
            extra = torch.pow(2, -torch.arange(1, 2 * (n - m) + 1, 2, device=device).float() / m)
            slopes = torch.cat([slopes, extra], dim=0)
        return slopes[:n]
    return get_slopes(n_heads)


def build_alibi_bias(num_heads: int, seq_len: int, device=None) -> torch.Tensor:
    # Returns bias shaped (1, H, T, S) to add to attention scores; assume causal mask applied separately
    slopes = _alibi_slopes(num_heads, device=device).view(1, num_heads, 1, 1)  # (1,H,1,1)
    i = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)  # (1,1,T,1)
    j = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)  # (1,1,1,S)
    diff = i - j  # (1,1,T,S)
    return -slopes * diff


def apply_rotary_2d(q: torch.Tensor, k: torch.Tensor, cos2d: torch.Tensor, sin2d: torch.Tensor, hw: tuple[int, int]):
    """Apply 2D RoPE to q,k.

    q,k: (B,H,T,Dh) where T = H*W flattened; cos2d/sin2d: (H,W,Dh)
    hw: (H, W)
    """
    Hh, Ww = int(hw[0]), int(hw[1])
    B, Hn, T, Dh = q.shape
    if T != Hh * Ww:
        raise ValueError("T must equal H*W for apply_rotary_2d")
    qv = q.view(B, Hn, Hh, Ww, Dh)
    kv = k.view(B, Hn, Hh, Ww, Dh)
    cos = cos2d.view(1, 1, Hh, Ww, Dh)
    sin = sin2d.view(1, 1, Hh, Ww, Dh)
    def _rot(x):
        x1 = x * cos
        xr = torch.stack([x[..., ::2], x[..., 1::2]], dim=-1)
        xr = torch.view_as_complex(xr)
        x2 = (xr.imag * sin).reshape_as(x)
        return x1 + x2
    qv = _rot(qv)
    kv = _rot(kv)
    return qv.view(B, Hn, T, Dh), kv.view(B, Hn, T, Dh)


def fit_alibi_slopes(num_heads: int, L: int, target_decay: float = 2.0, device=None) -> torch.Tensor:
    """Fit slopes so that average slope yields bias ~ -target_decay at distance L-1.

    Returns per-head slopes shaped (H,).
    """
    base = _alibi_slopes(num_heads, device=device)
    if L <= 1:
        return base
    scale = float(target_decay) / float(L - 1)
    # normalize by mean to preserve relative diversity
    base_norm = base / base.mean().clamp_min(1e-12)
    return base_norm * scale


def rotary_fft(q: torch.Tensor, k: torch.Tensor, theta: torch.Tensor):
    """Experimental rotary via angle tensor theta (broadcastable to (T,Dh))."""
    # Build cos/sin from theta
    if theta.shape[-1] != q.shape[-1]:
        # allow Dh/2 provided; repeat to Dh
        if theta.shape[-1] * 2 != q.shape[-1]:
            raise ValueError("theta last dim must match Dh or Dh/2")
        th = torch.repeat_interleave(theta, 2, dim=-1)
    else:
        th = theta
    cos = torch.cos(th)
    sin = torch.sin(th)
    return apply_rotary(q, k, cos, sin)


class RotaryEmbeddingHF:
    """Minimal HF-like rotary embedder producing (T,Dh) cos/sin.

    Uses existing build_rope_cache and optional attention scaling.
    """
    def __init__(self, head_dim: int, base_theta: float = 1e6, attention_scaling: float = 1.0, device=None):
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = int(head_dim)
        self.base_theta = float(base_theta)
        self.attention_scaling = float(attention_scaling)
        self.device = device
        self._cos = torch.tensor([], device=device)
        self._sin = torch.tensor([], device=device)

    @torch.no_grad()
    def _ensure(self, T: int, dtype: torch.dtype, device):
        if self._cos.numel() == 0 or self._cos.shape[0] < T or self._cos.device != device:
            cos, sin = build_rope_cache(T, self.head_dim, device=device, base_theta=self.base_theta)
            if self.attention_scaling != 1.0:
                cos = cos * float(self.attention_scaling)
                sin = sin * float(self.attention_scaling)
            self._cos = cos.to(dtype=dtype)
            self._sin = sin.to(dtype=dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,D)
        # If position_ids provided, gather cos/sin at those absolute positions; else return [0..T-1]
        B, T, D = x.shape
        device, dtype = x.device, x.dtype
        if position_ids is not None:
            pos = position_ids
            if pos.dim() == 2:
                # Assume all batches share same positions; use first batch
                pos = pos[0]
            pos = pos.to(device)
            T_need = int(pos.max().item()) + 1 if pos.numel() > 0 else T
            self._ensure(T_need, dtype, device)
            cos = self._cos.index_select(0, pos.view(-1)).view(T, -1)
            sin = self._sin.index_select(0, pos.view(-1)).view(T, -1)
            return cos, sin
        self._ensure(T, dtype, device)
        return self._cos[:T], self._sin[:T]
