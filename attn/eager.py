from typing import Optional
import torch
import torch.nn as nn

from specs.config import ModelConfig
from .interfaces import Attention, KVCache
from .backends import scaled_dot_product_attention, select_attention_backend
from tensor.shape import split_heads, merge_heads
from tensor.positional import build_rope_cache, apply_rotary


class EagerAttention(nn.Module):
    """Projection-internal self-attention supporting MHA/GQA, optional RoPE.

    Implements Attention.forward(q,k,v,mask,cache) with the convention that when k/v are None,
    `q` is the input hidden state (B,T,D) and this module performs QKV projection internally.
    """

    def __init__(self, cfg: ModelConfig, n_kv_heads: int | None = None, use_rope: bool | None = None, rope_theta: float | None = None, attn_dropout: float | None = None, is_causal: bool | None = None, backend_override: str | None = None):
        super().__init__()
        self.d_model = int(cfg.d_model)
        self.n_heads = int(cfg.n_heads)
        self.n_kv_heads = int(n_kv_heads if n_kv_heads is not None else getattr(cfg, "n_kv_heads", cfg.n_heads))
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = self.d_model // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be multiple of n_kv_heads")
        self.use_rope = bool(use_rope if use_rope is not None else getattr(cfg, "use_rope", True))
        self.rope_theta = float(rope_theta if rope_theta is not None else getattr(cfg, "rope_theta", 1e6))
        self.attn_dropout_p = float(attn_dropout if attn_dropout is not None else getattr(cfg, "attn_dropout", 0.0))
        self.is_causal = bool(is_causal if is_causal is not None else True)
        self.backend_override = backend_override.lower() if isinstance(backend_override, str) else None

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        self.register_buffer("_rope_cos", torch.tensor([]), persistent=False)
        self.register_buffer("_rope_sin", torch.tensor([]), persistent=False)

    def _ensure_rope_cache(self, seq_len: int, device, dtype):
        if not self.use_rope:
            return
        if self._rope_cos.numel() == 0 or self._rope_cos.shape[0] < seq_len or self._rope_cos.device != device:
            cos, sin = build_rope_cache(seq_len, self.head_dim, device=device, base_theta=self.rope_theta)
            self._rope_cos = cos.to(dtype=dtype)
            self._rope_sin = sin.to(dtype=dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        x = q
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        q_lin = self.w_q(x)
        k_lin = self.w_k(x if k is None else k)
        v_lin = self.w_v(x if v is None else v)

        qh = split_heads(q_lin, self.n_heads)          # (B, H, T, Dh)
        kh_new = split_heads(k_lin, self.n_kv_heads)   # (B, Hk, T, Dh)
        vh_new = split_heads(v_lin, self.n_kv_heads)   # (B, Hk, T, Dh)

        # Apply RoPE to queries and only the newly produced keys
        if self.use_rope:
            self._ensure_rope_cache(T, device, dtype)
            qh, kh_new = apply_rotary(qh, kh_new, self._rope_cos[:T], self._rope_sin[:T])

        # Read previously cached KV (already RoPE-applied) and concatenate
        if cache is not None:
            k_old, v_old = cache.read(0, cache.length())  # (B, Hk, Sold, Dh)
            if k_old is not None and k_old.shape[2] > 0:
                kh_all = torch.cat([k_old, kh_new], dim=2)
                vh_all = torch.cat([v_old, vh_new], dim=2)
            else:
                kh_all, vh_all = kh_new, vh_new
        else:
            kh_all, vh_all = kh_new, vh_new

        # Expand KV heads to match attention heads if using GQA
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            kh_all = kh_all.repeat_interleave(repeat, dim=1)
            vh_all = vh_all.repeat_interleave(repeat, dim=1)

        backend = self.backend_override or select_attention_backend(is_causal=self.is_causal, dtype=dtype, seq=T, heads=self.n_heads, device=device)
        out = scaled_dot_product_attention(
            qh, kh_all, vh_all, attn_mask=mask, dropout_p=self.attn_dropout_p if self.training else 0.0, backend=backend, is_causal=self.is_causal
        )
        y = merge_heads(out)

        # Append newly produced KV to cache (stored with Hk heads)
        if cache is not None and T > 0:
            cache.append(kh_new, vh_new)

        return self.w_o(y)


