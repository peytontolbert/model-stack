from __future__ import annotations

from typing import Optional
import os

import torch
import torch.nn as nn

from runtime.attention import _read_backend_from_env_or_file, scaled_dot_product_attention, select_attention_backend
from runtime.attention_interfaces import Attention, KVCache
from runtime.native import resolve_linear_backend
from runtime.ops import (
    attention as runtime_attention,
    apply_rotary as runtime_apply_rotary,
    head_output_projection as runtime_head_output_projection,
    linear as runtime_linear,
    pack_linear_weight as runtime_pack_linear_weight,
    pack_qkv_weights as runtime_pack_qkv_weights,
    prepare_attention_mask as runtime_prepare_attention_mask,
    qkv_heads_projection as runtime_qkv_heads_projection,
    qkv_packed_heads_projection as runtime_qkv_packed_heads_projection,
    resolve_rotary_embedding as runtime_resolve_rotary_embedding,
    split_heads as runtime_split_heads,
)
from specs.config import ModelConfig


class EagerAttention(nn.Module):
    """Projection-internal self-attention supporting MHA/GQA, optional RoPE.

    Implements Attention.forward(q,k,v,mask,cache) with the convention that when k/v are None,
    `q` is the input hidden state (B,T,D) and this module performs QKV projection internally.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        n_kv_heads: int | None = None,
        use_rope: bool | None = None,
        rope_theta: float | None = None,
        attn_dropout: float | None = None,
        is_causal: bool | None = None,
        backend_override: str | None = None,
    ):
        super().__init__()
        self.d_model = int(cfg.d_model)
        self.n_heads = int(cfg.n_heads)
        self.n_kv_heads = int(n_kv_heads if n_kv_heads is not None else getattr(cfg, "n_kv_heads", cfg.n_heads))
        hd = getattr(cfg, "head_dim", None)
        if hd is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.head_dim = self.d_model // self.n_heads
        else:
            self.head_dim = int(hd)
            if self.n_heads * self.head_dim != self.d_model:
                raise ValueError(f"n_heads*head_dim ({self.n_heads*self.head_dim}) != d_model ({self.d_model})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be multiple of n_kv_heads")

        self.scaling = self.head_dim ** -0.5
        self.use_rope = bool(use_rope if use_rope is not None else getattr(cfg, "use_rope", True))
        self.rope_theta = float(rope_theta if rope_theta is not None else getattr(cfg, "rope_theta", 1e6))
        self.rope_scaling_type = getattr(cfg, "rope_scaling_type", None)
        self.rope_scaling_factor = getattr(cfg, "rope_scaling_factor", None)
        self.rope_scaling_low_freq_factor = getattr(cfg, "rope_scaling_low_freq_factor", None)
        self.rope_scaling_high_freq_factor = getattr(cfg, "rope_scaling_high_freq_factor", None)
        self.attn_dropout_p = float(attn_dropout if attn_dropout is not None else getattr(cfg, "attn_dropout", 0.0))
        self.rope_attention_scaling = float(getattr(cfg, "rope_attention_scaling", 1.0) or 1.0)
        self.is_causal = bool(is_causal if is_causal is not None else True)
        self.backend_override = backend_override.lower() if isinstance(backend_override, str) else None

        attn_bias = bool(getattr(cfg, "attention_bias", False))
        self.w_q = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=attn_bias)
        self.w_k = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=attn_bias)
        self.w_v = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=attn_bias)
        self.w_o = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=attn_bias)

        self.register_buffer("_rope_cos", torch.tensor([]), persistent=False)
        self.register_buffer("_rope_sin", torch.tensor([]), persistent=False)
        self._packed_qkv_signature = None
        self._packed_qkv_weight = None
        self._packed_qkv_bias = None
        self._packed_q_sizes = None
        self._packed_o_signature = None
        self._packed_o_weight = None
        self._packed_o_bias = None

    @staticmethod
    def _tensor_signature(t: torch.Tensor | None):
        if t is None:
            return None
        return (
            str(t.device),
            str(t.dtype),
            tuple(t.shape),
            int(getattr(t, "_version", 0)),
            int(t.data_ptr()),
        )

    def _packed_backend(self, x: torch.Tensor) -> str | None:
        if self.training or (not x.is_cuda):
            return None
        backend = resolve_linear_backend("auto")
        return backend if backend == "cublaslt" else None

    def _ensure_packed_qkv(self, backend: str):
        signature = (
            backend,
            self._tensor_signature(self.w_q.weight),
            self._tensor_signature(self.w_q.bias),
            self._tensor_signature(self.w_k.weight),
            self._tensor_signature(self.w_k.bias),
            self._tensor_signature(self.w_v.weight),
            self._tensor_signature(self.w_v.bias),
        )
        if signature != self._packed_qkv_signature:
            packed_weight, packed_bias, q_size, k_size, v_size = runtime_pack_qkv_weights(
                self.w_q.weight,
                self.w_q.bias,
                self.w_k.weight,
                self.w_k.bias,
                self.w_v.weight,
                self.w_v.bias,
            )
            self._packed_qkv_signature = signature
            self._packed_qkv_weight = packed_weight
            self._packed_qkv_bias = packed_bias
            self._packed_q_sizes = (q_size, k_size, v_size)
        return self._packed_qkv_weight, self._packed_qkv_bias, self._packed_q_sizes

    def _ensure_packed_output(self, backend: str):
        signature = (
            backend,
            self._tensor_signature(self.w_o.weight),
            self._tensor_signature(self.w_o.bias),
        )
        if signature != self._packed_o_signature:
            packed_weight, packed_bias = runtime_pack_linear_weight(self.w_o.weight, self.w_o.bias)
            self._packed_o_signature = signature
            self._packed_o_weight = packed_weight
            self._packed_o_bias = packed_bias
        return self._packed_o_weight, self._packed_o_bias

    def _ensure_rope_cache(self, seq_len: int, reference: torch.Tensor):
        if not self.use_rope:
            return
        if (
            self._rope_cos.numel() == 0
            or self._rope_cos.shape[0] < seq_len
            or self._rope_cos.device != reference.device
            or self._rope_cos.dtype != reference.dtype
        ):
            base_theta = self.rope_theta
            st = (self.rope_scaling_type or "").lower() if isinstance(self.rope_scaling_type, str) else None
            if st == "linear" and self.rope_scaling_factor is not None:
                try:
                    base_theta = float(base_theta) * float(self.rope_scaling_factor)
                except Exception:
                    base_theta = float(base_theta)
            cos, sin = runtime_resolve_rotary_embedding(
                reference=reference[:, :seq_len, :],
                head_dim=self.head_dim,
                base_theta=float(base_theta),
                attention_scaling=float(self.rope_attention_scaling),
            )
            self._rope_cos = cos
            self._rope_sin = sin

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
        import traceback

        trace = os.getenv("GEN_TRACE_ATTENTION", os.getenv("GEN_TRACE", "0")) == "1"
        x = q
        B, T, D = x.shape
        device, dtype = x.device, x.dtype
        packed_backend = self._packed_backend(x)

        if k is None and v is None:
            if packed_backend is not None:
                packed_weight, packed_bias, packed_sizes = self._ensure_packed_qkv(packed_backend)
                q_size, k_size, v_size = packed_sizes
                qh, kh_new, vh_new = runtime_qkv_packed_heads_projection(
                    x,
                    packed_weight,
                    packed_bias,
                    q_size=q_size,
                    k_size=k_size,
                    v_size=v_size,
                    q_heads=self.n_heads,
                    kv_heads=self.n_kv_heads,
                    backend=packed_backend,
                )
            else:
                qh, kh_new, vh_new = runtime_qkv_heads_projection(
                    x,
                    self.w_q.weight,
                    self.w_q.bias,
                    self.w_k.weight,
                    self.w_k.bias,
                    self.w_v.weight,
                    self.w_v.bias,
                    q_heads=self.n_heads,
                    kv_heads=self.n_kv_heads,
                )
        else:
            q_lin = runtime_linear(x, self.w_q.weight, self.w_q.bias)
            k_lin = runtime_linear(x if k is None else k, self.w_k.weight, self.w_k.bias)
            v_lin = runtime_linear(x if v is None else v, self.w_v.weight, self.w_v.bias)
            qh = runtime_split_heads(q_lin, self.n_heads)
            kh_new = runtime_split_heads(k_lin, self.n_kv_heads)
            vh_new = runtime_split_heads(v_lin, self.n_kv_heads)

        if self.use_rope:
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cos_t = cos[:T]
                sin_t = sin[:T]
                qh, kh_new = runtime_apply_rotary(qh, kh_new, cos_t, sin_t)
            else:
                self._ensure_rope_cache(T, x)
                st = (self.rope_scaling_type or "").lower() if isinstance(self.rope_scaling_type, str) else None
                if st == "yarn" and self.rope_scaling_factor is not None:
                    qh_rot, kh_rot = runtime_apply_rotary(qh, kh_new, self._rope_cos[:T], self._rope_sin[:T])
                    from runtime.positional import rope_yarn_factors

                    sq, sk = rope_yarn_factors(T, self.rope_theta, float(self.rope_scaling_factor))
                    qh = qh_rot * float(sq)
                    kh_new = kh_rot * float(sk)
                else:
                    qh, kh_new = runtime_apply_rotary(qh, kh_new, self._rope_cos[:T], self._rope_sin[:T])

        appended_to_cache = False
        if cache is not None:
            if T > 0 and hasattr(cache, "append_and_read"):
                kh_all, vh_all = cache.append_and_read(kh_new, vh_new, 0)
                appended_to_cache = True
            else:
                k_old, v_old = cache.read(0, cache.length())
                if k_old is not None and k_old.shape[2] > 0:
                    kh_all = torch.cat([k_old, kh_new], dim=2)
                    vh_all = torch.cat([v_old, vh_new], dim=2)
                else:
                    kh_all, vh_all = kh_new, vh_new
        else:
            kh_all, vh_all = kh_new, vh_new

        def _expanded_kv_heads() -> tuple[torch.Tensor, torch.Tensor]:
            if self.n_kv_heads == self.n_heads:
                return kh_all, vh_all
            repeat = self.n_heads // self.n_kv_heads
            return (
                kh_all.repeat_interleave(repeat, dim=1),
                vh_all.repeat_interleave(repeat, dim=1),
            )

        S = kh_all.shape[2]
        add = runtime_prepare_attention_mask(
            mask,
            batch_size=B,
            num_heads=self.n_heads,
            tgt_len=T,
            src_len=S,
            position_ids=position_ids,
        )
        parity_exact = os.environ.get("ATTN_PARITY_EXACT", "0") == "1"
        if parity_exact:
            kh_attn, vh_attn = _expanded_kv_heads()
            scores = torch.matmul(qh, kh_attn.transpose(2, 3)) * float(self.scaling)
            if add is not None:
                scores = scores + add.to(dtype=scores.dtype)
            attn_probs = torch.nn.functional.softmax(scores.float(), dim=-1).to(dtype)
            if self.training and self.attn_dropout_p > 0.0:
                attn_probs = torch.nn.functional.dropout(attn_probs, p=self.attn_dropout_p)
            out = torch.matmul(attn_probs, vh_attn)
        else:
            is_causal_flag = add is None
            forced_backend = self.backend_override or _read_backend_from_env_or_file()
            use_runtime_attention = forced_backend is None and (self.attn_dropout_p if self.training else 0.0) == 0.0
            if use_runtime_attention:
                kh_backend, vh_backend = kh_all, vh_all
                out = runtime_attention(
                    qh,
                    kh_backend,
                    vh_backend,
                    attn_mask=add,
                    is_causal=is_causal_flag,
                    scale=float(self.scaling),
                )
            else:
                backend = forced_backend or select_attention_backend(
                    is_causal=is_causal_flag,
                    dtype=dtype,
                    seq=T,
                    heads=self.n_heads,
                    device=device,
                )
                use_native_gqa = backend == "torch" and (self.attn_dropout_p if self.training else 0.0) == 0.0
                if use_native_gqa:
                    kh_backend, vh_backend = kh_all, vh_all
                else:
                    kh_backend, vh_backend = _expanded_kv_heads()
                mask_for_backend = add
                if mask_for_backend is not None and backend == "torch" and mask_for_backend.dtype != qh.dtype:
                    mask_for_backend = mask_for_backend.to(dtype=qh.dtype)
                try:
                    out = scaled_dot_product_attention(
                        qh,
                        kh_backend,
                        vh_backend,
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

        if cache is not None and T > 0 and not appended_to_cache:
            cache.append(kh_new, vh_new)

        if packed_backend is not None:
            packed_o_weight, packed_o_bias = self._ensure_packed_output(packed_backend)
            return runtime_head_output_projection(out, packed_o_weight, packed_o_bias, backend=packed_backend)
        return runtime_head_output_projection(out, self.w_o.weight, self.w_o.bias)


class FlashAttention(EagerAttention):
    def __init__(self, cfg: ModelConfig, **overrides):
        super().__init__(cfg, backend_override="flash2", **overrides)


class TritonAttention(EagerAttention):
    def __init__(self, cfg: ModelConfig, **overrides):
        super().__init__(cfg, backend_override="triton", **overrides)


class XFormersAttention(EagerAttention):
    def __init__(self, cfg: ModelConfig, **overrides):
        super().__init__(cfg, backend_override="xformers", **overrides)


__all__ = [
    "Attention",
    "KVCache",
    "EagerAttention",
    "FlashAttention",
    "TritonAttention",
    "XFormersAttention",
]
