from typing import Optional
import os
import torch
import torch.nn as nn

from specs.config import ModelConfig
from .interfaces import Attention, KVCache
from .backends import scaled_dot_product_attention, select_attention_backend
from tensor.shape import split_heads, merge_heads
from tensor.masking import to_additive_mask
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
        # Prefer explicit head_dim if provided
        hd = getattr(cfg, "head_dim", None)
        if hd is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.head_dim = self.d_model // self.n_heads
        else:
            self.head_dim = int(hd)
            # Ensure consistency
            if self.n_heads * self.head_dim != self.d_model:
                raise ValueError(f"n_heads*head_dim ({self.n_heads*self.head_dim}) != d_model ({self.d_model})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be multiple of n_kv_heads")
        
        # HF-compatible explicit scaling factor for attention scores
        self.scaling = self.head_dim ** -0.5
        
        self.use_rope = bool(use_rope if use_rope is not None else getattr(cfg, "use_rope", True))
        self.rope_theta = float(rope_theta if rope_theta is not None else getattr(cfg, "rope_theta", 1e6))
        # Optional RoPE scaling from config
        self.rope_scaling_type = getattr(cfg, "rope_scaling_type", None)
        self.rope_scaling_factor = getattr(cfg, "rope_scaling_factor", None)
        self.rope_scaling_low_freq_factor = getattr(cfg, "rope_scaling_low_freq_factor", None)
        self.rope_scaling_high_freq_factor = getattr(cfg, "rope_scaling_high_freq_factor", None)
        self.attn_dropout_p = float(attn_dropout if attn_dropout is not None else getattr(cfg, "attn_dropout", 0.0))
        # Optional HF-style rotary attention scaling (multiply cos/sin)
        self.rope_attention_scaling = float(getattr(cfg, "rope_attention_scaling", 1.0) or 1.0)
        self.is_causal = bool(is_causal if is_causal is not None else True)
        self.backend_override = backend_override.lower() if isinstance(backend_override, str) else None

        # Use n_heads * head_dim for Q projection (matches HF exactly)
        attn_bias = bool(getattr(cfg, "attention_bias", False))
        self.w_q = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=attn_bias)
        self.w_k = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=attn_bias)
        self.w_v = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=attn_bias)
        self.w_o = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=attn_bias)

        self.register_buffer("_rope_cos", torch.tensor([]), persistent=False)
        self.register_buffer("_rope_sin", torch.tensor([]), persistent=False)

    def _ensure_rope_cache(self, seq_len: int, device, dtype):
        if not self.use_rope:
            return
        if self._rope_cos.numel() == 0 or self._rope_cos.shape[0] < seq_len or self._rope_cos.device != device:
            # Apply simple HF-compatible scaling for linear interpolation by stretching base_theta
            base_theta = self.rope_theta
            st = (self.rope_scaling_type or "").lower() if isinstance(self.rope_scaling_type, str) else None
            if st == "linear" and self.rope_scaling_factor is not None:
                try:
                    base_theta = float(base_theta) * float(self.rope_scaling_factor)
                except Exception:
                    base_theta = float(base_theta)
            cos, sin = build_rope_cache(seq_len, self.head_dim, device=device, base_theta=base_theta)
            if self.rope_attention_scaling != 1.0:
                cos = cos * float(self.rope_attention_scaling)
                sin = sin * float(self.rope_attention_scaling)
            self._rope_cos = cos.to(dtype=dtype)
            self._rope_sin = sin.to(dtype=dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        cache: Optional[KVCache] = None,
        *,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        import os, traceback
        trace = (os.getenv("GEN_TRACE_ATTENTION", os.getenv("GEN_TRACE", "0")) == "1")
        x = q
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        q_lin = self.w_q(x)
        k_lin = self.w_k(x if k is None else k)
        v_lin = self.w_v(x if v is None else v)

        qh = split_heads(q_lin, self.n_heads)          # (B, H, T, Dh)
        kh_new = split_heads(k_lin, self.n_kv_heads)   # (B, Hk, T, Dh)
        vh_new = split_heads(v_lin, self.n_kv_heads)   # (B, Hk, T, Dh)

        # Apply RoPE using provided embeddings or internal cache
        if self.use_rope:
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cos_t = cos[:T]
                sin_t = sin[:T]
                qh, kh_new = apply_rotary(qh, kh_new, cos_t, sin_t)
            else:
                self._ensure_rope_cache(T, device, dtype)
                st = (self.rope_scaling_type or "").lower() if isinstance(self.rope_scaling_type, str) else None
                if st == "yarn" and self.rope_scaling_factor is not None:
                    # Scale outputs per YaRN after applying rotary
                    qh_rot, kh_rot = apply_rotary(qh, kh_new, self._rope_cos[:T], self._rope_sin[:T])
                    from tensor.positional import rope_yarn_factors
                    sq, sk = rope_yarn_factors(T, self.rope_theta, float(self.rope_scaling_factor))
                    qh = qh_rot * float(sq)
                    kh_new = kh_rot * float(sk)
                else:
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

        # Use explicit matmul + float32 softmax (HF-eager parity) if enabled; else route to SDPA
        S = kh_all.shape[2]
        add = None
        if mask is not None:
            add = to_additive_mask(mask) if mask.dtype == torch.bool else mask
            if add.shape[-1] != S:
                add = add[..., :S]
            # Ensure mask shape is (B,H,T,S) and dtype matches query dtype
            if add.ndim == 4 and add.shape[1] == 1:
                add = add.expand(B, self.n_heads, T, S)
            # Keep additive mask in float32 for numerical stability (bf16 can underflow -inf)
            # SDPA accepts float masks; avoid downcasting to attention dtype
            import torch as _torch
            add = add.to(dtype=_torch.float32)
            # Guarantee self (diagonal) is unmasked to avoid fully-masked rows (HF parity)
            if position_ids is not None:
                try:
                    pos = position_ids
                    if pos.dim() == 2:
                        # (B,T)
                        pos_idx = pos.clamp_min(0).clamp_max(S - 1)
                        zeros = torch.zeros(B, self.n_heads, T, 1, dtype=add.dtype, device=add.device)
                        idx = pos_idx.view(B, 1, T, 1).expand(B, self.n_heads, T, 1)
                        add = add.scatter(dim=3, index=idx, src=zeros)
                except Exception:
                    pass
        parity_exact = os.environ.get("ATTN_PARITY_EXACT", "0") == "1"
        if parity_exact:
            # HF-like: explicit attention scores + additive mask + float32 softmax
            scores = torch.matmul(qh, kh_all.transpose(2, 3)) * float(self.scaling)  # (B,H,T,S)
            if add is not None:
                scores = scores + add.to(dtype=scores.dtype)
            attn_probs = torch.nn.functional.softmax(scores.float(), dim=-1).to(dtype)
            if self.training and self.attn_dropout_p > 0.0:
                attn_probs = torch.nn.functional.dropout(attn_probs, p=self.attn_dropout_p)
            out = torch.matmul(attn_probs, vh_all)
            y = merge_heads(out)
        else:
            # Choose backend; avoid double causal when explicit mask is provided
            is_causal_flag = add is None
            backend = self.backend_override or select_attention_backend(
                is_causal=is_causal_flag, dtype=dtype, seq=T, heads=self.n_heads, device=device
            )
            mask_for_backend = add
            # Some backends (torch SDPA) may expect float/bool mask with dtype matching q/k/v
            if mask_for_backend is not None and backend == "torch" and mask_for_backend.dtype != qh.dtype:
                mask_for_backend = mask_for_backend.to(dtype=qh.dtype)
          #  if trace:
           #     try:
            #        print(f"[attn] backend={backend} causal={is_causal_flag} qdtype={qh.dtype} mdtype={(mask_for_backend.dtype if mask_for_backend is not None else None)} T={T} S={kh_all.shape[2]}")
            #    except Exception:
            #        pass
            try:
                out = scaled_dot_product_attention(
                    qh, kh_all, vh_all,
                    attn_mask=mask_for_backend,
                    dropout_p=(self.attn_dropout_p if self.training else 0.0),
                    backend=backend,
                    is_causal=is_causal_flag,
                    scale=float(self.scaling),
                )
            except Exception:
                if trace:
                    print("[attn] exception in SDPA/backend call")
                    traceback.print_exc()
                raise
            y = merge_heads(out)

        # Append newly produced KV to cache (stored with Hk heads)
        if cache is not None and T > 0:
            cache.append(kh_new, vh_new)

        return self.w_o(y)


