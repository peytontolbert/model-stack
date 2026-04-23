from __future__ import annotations

import math

import torch

from runtime.ops import apply_rotary as runtime_apply_rotary


def build_rope_cache(seq_len: int, head_dim: int, device=None, base_theta: float = 1e6):
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        base_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim)
    )
    freqs = torch.einsum("t,d->td", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return runtime_apply_rotary(q, k, cos, sin)


def apply_rotary_scaled(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, scale: float):
    q_out, k_out = apply_rotary(q, k, cos, sin)
    return q_out * scale, k_out * scale


def rope_ntk_scaling(
    seq_len: int,
    base_theta: float,
    factor: float = 1.0,
    *,
    original_max_position_embeddings: int | None = None,
    head_dim: int | None = None,
) -> float:
    base = float(base_theta)
    scale_factor = float(factor)
    if scale_factor <= 1.0:
        return base
    orig = int(original_max_position_embeddings) if original_max_position_embeddings is not None else int(seq_len)
    if orig <= 0 or int(seq_len) <= orig:
        return base
    ratio = (scale_factor * float(seq_len) / float(orig)) - (scale_factor - 1.0)
    ratio = max(ratio, 1.0)
    exponent = 1.0
    if head_dim is not None and int(head_dim) > 2:
        exponent = float(int(head_dim)) / float(int(head_dim) - 2)
    return base * (ratio ** exponent)


def rope_yarn_factors(
    seq_len: int,
    base_theta: float,
    factor: float,
    *,
    original_max_position_embeddings: int | None = None,
    low_freq_factor: float | None = None,
    high_freq_factor: float | None = None,
) -> tuple[float, float]:
    del base_theta, low_freq_factor, high_freq_factor
    scale = float(factor)
    orig = int(original_max_position_embeddings) if original_max_position_embeddings is not None else 0
    if orig > 0:
        scale = max(scale, float(seq_len) / float(orig))
    scale = max(scale, 1.0)
    if scale <= 1.0:
        return 1.0, 1.0
    attention_factor = 0.1 * math.log(scale) + 1.0
    return float(attention_factor), float(attention_factor)


def resolve_rope_seq_len(seq_len: int, position_ids: torch.Tensor | None = None) -> int:
    needed = int(seq_len)
    if position_ids is None or not isinstance(position_ids, torch.Tensor) or position_ids.numel() == 0:
        return needed
    try:
        gather_pos = position_ids
        if gather_pos.dim() == 2:
            gather_pos = gather_pos[0]
        if gather_pos.numel() > 0:
            needed = max(needed, int(gather_pos.max().item()) + 1)
    except Exception:
        return needed
    return needed


def resolve_rope_parameters(
    *,
    seq_len: int,
    head_dim: int,
    base_theta: float,
    attention_scaling: float = 1.0,
    scaling_type: str | None = None,
    scaling_factor: float | None = None,
    original_max_position_embeddings: int | None = None,
    low_freq_factor: float | None = None,
    high_freq_factor: float | None = None,
) -> tuple[float, float]:
    base = float(base_theta)
    attn = float(attention_scaling)
    st = (scaling_type or "").lower() if isinstance(scaling_type, str) else None
    factor = None if scaling_factor is None else float(scaling_factor)
    if st == "linear" and factor is not None:
        base = float(base) * factor
    elif st in {"dynamic", "ntk"} and factor is not None:
        base = rope_ntk_scaling(
            int(seq_len),
            float(base),
            factor=factor,
            original_max_position_embeddings=original_max_position_embeddings,
            head_dim=int(head_dim),
        )
    elif st == "yarn" and factor is not None:
        sq, sk = rope_yarn_factors(
            int(seq_len),
            float(base),
            factor=factor,
            original_max_position_embeddings=original_max_position_embeddings,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
        )
        attn *= math.sqrt(float(sq) * float(sk))
    return float(base), float(attn)


def relative_position_bucket(t: int, s: int, num_buckets: int, max_distance: int) -> int:
    n = num_buckets // 2
    rel = s - t
    sign = 0 if rel <= 0 else 1
    rel = abs(rel)
    max_exact = n // 2
    if rel < max_exact:
        bucket = rel
    else:
        bucket = max_exact + int(
            (
                torch.log(torch.tensor(rel / max_exact))
                / torch.log(torch.tensor(max_distance / max_exact))
                * (n - max_exact)
            )
            .clamp(min=0, max=n - max_exact)
            .item()
        )
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
    del method
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
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even for 2D cache")
    Dh2 = head_dim // 2
    t_h = torch.arange(H, device=device, dtype=torch.float32)
    inv_h = (
        1.0 / (base_theta ** (torch.arange(0, Dh2, 2, device=device).float() / Dh2))
        if Dh2 > 0
        else torch.tensor([], device=device)
    )
    freqs_h = torch.einsum("t,d->td", t_h, inv_h) if Dh2 > 0 else torch.zeros(H, 0, device=device)
    cos_h = torch.cos(torch.repeat_interleave(freqs_h, 2, dim=-1))
    sin_h = torch.sin(torch.repeat_interleave(freqs_h, 2, dim=-1))

    t_w = torch.arange(W, device=device, dtype=torch.float32)
    inv_w = (
        1.0 / (base_theta ** (torch.arange(0, Dh2, 2, device=device).float() / Dh2))
        if Dh2 > 0
        else torch.tensor([], device=device)
    )
    freqs_w = torch.einsum("t,d->td", t_w, inv_w) if Dh2 > 0 else torch.zeros(W, 0, device=device)
    cos_w = torch.cos(torch.repeat_interleave(freqs_w, 2, dim=-1))
    sin_w = torch.sin(torch.repeat_interleave(freqs_w, 2, dim=-1))

    cos = torch.zeros(H, W, head_dim, device=device)
    sin = torch.zeros(H, W, head_dim, device=device)
    if Dh2 > 0:
        cos[:, :, :Dh2] = cos_h[:, None, :]
        sin[:, :, :Dh2] = sin_h[:, None, :]
        cos[:, :, Dh2:] = cos_w[None, :, :]
        sin[:, :, Dh2:] = sin_w[None, :, :]
    return cos, sin


def build_relative_position_indices(tgt_len: int, src_len: int, max_distance: int, device=None) -> torch.Tensor:
    i = torch.arange(tgt_len, device=device).view(tgt_len, 1)
    j = torch.arange(src_len, device=device).view(1, src_len)
    rel = j - i
    rel = rel.clamp(min=-(max_distance - 1), max=(max_distance - 1))
    return (rel + (max_distance - 1)).to(torch.long)


def relative_position_bias_from_table(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    H, _ = table.shape
    T, S = indices.shape
    bias = table[:, indices.view(-1)].view(H, T, S)
    return bias.unsqueeze(0)


def _alibi_slopes(n_heads: int, device=None) -> torch.Tensor:
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
    slopes = _alibi_slopes(num_heads, device=device).view(1, num_heads, 1, 1)
    i = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
    j = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)
    return -slopes * (i - j)


def apply_rotary_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos2d: torch.Tensor,
    sin2d: torch.Tensor,
    hw: tuple[int, int],
):
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
    base = _alibi_slopes(num_heads, device=device)
    if L <= 1:
        return base
    scale = float(target_decay) / float(L - 1)
    base_norm = base / base.mean().clamp_min(1e-12)
    return base_norm * scale


def rotary_fft(q: torch.Tensor, k: torch.Tensor, theta: torch.Tensor):
    if theta.shape[-1] != q.shape[-1]:
        if theta.shape[-1] * 2 != q.shape[-1]:
            raise ValueError("theta last dim must match Dh or Dh/2")
        th = torch.repeat_interleave(theta, 2, dim=-1)
    else:
        th = theta
    return apply_rotary(q, k, torch.cos(th), torch.sin(th))


class RotaryEmbeddingHF:
    def __init__(
        self,
        head_dim: int,
        base_theta: float = 1e6,
        attention_scaling: float = 1.0,
        device=None,
        scaling_type: str | None = None,
        scaling_factor: float | None = None,
        original_max_position_embeddings: int | None = None,
        low_freq_factor: float | None = None,
        high_freq_factor: float | None = None,
    ):
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = int(head_dim)
        self.base_theta = float(base_theta)
        self.attention_scaling = float(attention_scaling)
        self.device = device
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self._cos = torch.tensor([], device=device)
        self._sin = torch.tensor([], device=device)

    @torch.no_grad()
    def _ensure(self, T: int, dtype: torch.dtype, device):
        if self._cos.numel() == 0 or self._cos.shape[0] < T or self._cos.device != device:
            base, attn = resolve_rope_parameters(
                seq_len=int(T),
                head_dim=int(self.head_dim),
                base_theta=float(self.base_theta),
                attention_scaling=float(self.attention_scaling),
                scaling_type=self.scaling_type,
                scaling_factor=self.scaling_factor,
                original_max_position_embeddings=self.original_max_position_embeddings,
                low_freq_factor=self.low_freq_factor,
                high_freq_factor=self.high_freq_factor,
            )
            cos, sin = build_rope_cache(T, self.head_dim, device=device, base_theta=base)
            if attn != 1.0:
                cos = cos * float(attn)
                sin = sin * float(attn)
            self._cos = cos.to(dtype=dtype)
            self._sin = sin.to(dtype=dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, T, _ = x.shape
        device, dtype = x.device, x.dtype
        if position_ids is not None:
            pos = position_ids
            if pos.dim() == 2:
                pos = pos[0]
            pos = pos.to(device)
            T_need = int(pos.max().item()) + 1 if pos.numel() > 0 else T
            self._ensure(T_need, dtype, device)
            cos = self._cos.index_select(0, pos.view(-1)).view(T, -1)
            sin = self._sin.index_select(0, pos.view(-1)).view(T, -1)
            return cos, sin
        self._ensure(T, dtype, device)
        return self._cos[:T], self._sin[:T]


__all__ = [
    "RotaryEmbeddingHF",
    "alibi_slopes",
    "apply_rotary",
    "apply_rotary_2d",
    "apply_rotary_scaled",
    "build_alibi_bias",
    "build_relative_position_indices",
    "build_rope_cache",
    "build_rope_cache_2d",
    "build_sinusoidal_cache",
    "fit_alibi_slopes",
    "relative_position_bias_from_table",
    "relative_position_bucket",
    "resolve_rope_seq_len",
    "rescale_positions",
    "rope_ntk_scaling",
    "resolve_rope_parameters",
    "rope_yarn_factors",
    "rotary_fft",
]
