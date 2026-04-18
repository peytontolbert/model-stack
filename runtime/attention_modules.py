from __future__ import annotations

from typing import Optional
import os

import torch
import torch.nn as nn

from runtime.attention import _read_backend_from_env_or_file, scaled_dot_product_attention, select_attention_backend
from runtime.hardware import prefer_hopper_library_attention
from runtime.attention_interfaces import Attention, KVCache
from runtime.native import resolve_linear_backend
from runtime.quant import int8_matmul_qkv as runtime_int8_matmul_qkv
from runtime.quant import quantize_activation_int8_rowwise as runtime_quantize_activation_int8_rowwise
from runtime.ops import (
    attention as runtime_attention,
    apply_rotary as runtime_apply_rotary,
    head_output_packed_projection as runtime_head_output_packed_projection,
    head_output_projection as runtime_head_output_projection,
    linear_module as runtime_linear_module,
    packed_qkv_module_signature as runtime_packed_qkv_module_signature,
    packed_linear_module_signature as runtime_packed_linear_module_signature,
    merge_heads as runtime_merge_heads,
    prepare_attention_mask as runtime_prepare_attention_mask,
    qkv_heads_projection as runtime_qkv_heads_projection,
    qkv_packed_spec_heads_projection as runtime_qkv_packed_spec_heads_projection,
    resolve_packed_qkv_module_spec as runtime_resolve_packed_qkv_module_spec,
    resolve_packed_linear_module_spec as runtime_resolve_packed_linear_module_spec,
    resolve_linear_module_tensors as runtime_resolve_linear_module_tensors,
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
        self._packed_qkv_spec = None
        self._packed_q_sizes = None
        self._packed_o_signature = None
        self._packed_o_spec = None

    def _uses_module_runtime_linear(self) -> bool:
        for projection in (self.w_q, self.w_k, self.w_v, self.w_o):
            runtime_linear = getattr(projection, "runtime_linear", None)
            if callable(runtime_linear):
                return True
        return False

    def _shared_int8_qkv_input_signature(self):
        shared_signature = None
        for projection in (self.w_q, self.w_k, self.w_v):
            runtime_signature = getattr(projection, "runtime_shared_int8_input_signature", None)
            runtime_linear = getattr(projection, "runtime_linear_from_quantized_input", None)
            runtime_quantize = getattr(projection, "runtime_quantize_int8_input", None)
            if not callable(runtime_signature) or not callable(runtime_linear) or not callable(runtime_quantize):
                return None
            current = runtime_signature()
            if current is None:
                return None
            if shared_signature is None:
                shared_signature = current
            elif current != shared_signature:
                return None
        return shared_signature

    def _shared_int8_qkv_projection(self, x: torch.Tensor):
        qx, row_scale, out_dtype = self.w_q.runtime_quantize_int8_input(x)
        q_lin = self.w_q.runtime_linear_from_quantized_input(qx, row_scale, out_dtype=out_dtype)
        k_lin = self.w_k.runtime_linear_from_quantized_input(qx, row_scale, out_dtype=out_dtype)
        v_lin = self.w_v.runtime_linear_from_quantized_input(qx, row_scale, out_dtype=out_dtype)
        return (
            runtime_split_heads(q_lin, self.n_heads),
            runtime_split_heads(k_lin, self.n_kv_heads),
            runtime_split_heads(v_lin, self.n_kv_heads),
        )

    def _supports_int8_attention_core(self) -> bool:
        for projection in (self.w_q, self.w_k, self.w_v):
            runtime_signature = getattr(projection, "runtime_shared_int8_input_signature", None)
            if not callable(runtime_signature):
                return False
            if runtime_signature() is None:
                return False
        return True

    def _int8_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        qq, q_scale = runtime_quantize_activation_int8_rowwise(q)
        kq, k_scale = runtime_quantize_activation_int8_rowwise(k)
        vq, v_scale = runtime_quantize_activation_int8_rowwise(v)
        return runtime_int8_matmul_qkv(
            qq,
            kq,
            vq,
            q_scale,
            k_scale,
            v_scale,
            attn_mask,
            is_causal=is_causal,
            scale=scale,
            out_dtype=q.dtype,
        )

    def _packed_backend(self, x: torch.Tensor) -> str | None:
        if self.training or (not getattr(x, "is_cuda", False)):
            return None
        preferred_backend = resolve_linear_backend("auto")

        def _supports_backend(candidate: str) -> bool:
            for projection in (self.w_q, self.w_k, self.w_v, self.w_o):
                runtime_linear = getattr(projection, "runtime_linear", None)
                supports_packed_backend = getattr(projection, "runtime_supports_packed_backend", None)
                if callable(runtime_linear):
                    if not callable(supports_packed_backend):
                        return False
                    if not bool(supports_packed_backend(candidate)):
                        return False
                elif candidate == "cublaslt":
                    if callable(supports_packed_backend) and not bool(supports_packed_backend(candidate)):
                        return False
                else:
                    return False
            return True

        candidates: list[str] = []
        if preferred_backend == "cublaslt":
            candidates.append("cublaslt")
        if preferred_backend == "bitnet":
            candidates.append("bitnet")
        if "bitnet" not in candidates:
            candidates.append("bitnet")

        for candidate in candidates:
            if _supports_backend(candidate):
                return candidate
        return None

    def _ensure_packed_qkv(self, backend: str, reference: torch.Tensor):
        signature = (
            backend,
            str(reference.device),
            str(reference.dtype),
            runtime_packed_qkv_module_signature(self.w_q, self.w_k, self.w_v, backend=backend),
        )
        if signature != self._packed_qkv_signature:
            spec = runtime_resolve_packed_qkv_module_spec(self.w_q, self.w_k, self.w_v, backend=backend, reference=reference)
            if spec is None:
                raise RuntimeError(f"Unable to resolve packed QKV spec for backend {backend}")
            self._packed_qkv_signature = signature
            self._packed_qkv_spec = spec
            self._packed_q_sizes = (int(spec["q_size"]), int(spec["k_size"]), int(spec["v_size"]))
        return self._packed_qkv_spec, self._packed_q_sizes

    def _ensure_packed_output(self, backend: str, reference: torch.Tensor):
        signature = (
            backend,
            str(reference.device),
            str(reference.dtype),
            runtime_packed_linear_module_signature(self.w_o, backend=backend),
        )
        if signature != self._packed_o_signature:
            spec = runtime_resolve_packed_linear_module_spec(self.w_o, backend=backend, reference=reference)
            if spec is None:
                raise RuntimeError(f"Unable to resolve packed output projection spec for backend {backend}")
            self._packed_o_signature = signature
            self._packed_o_spec = spec
        return self._packed_o_spec

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
                packed_qkv_spec, _ = self._ensure_packed_qkv(packed_backend, x)
                qh, kh_new, vh_new = runtime_qkv_packed_spec_heads_projection(
                    x,
                    packed_qkv_spec,
                    q_heads=self.n_heads,
                    kv_heads=self.n_kv_heads,
                    backend=packed_backend,
                )
            elif self._shared_int8_qkv_input_signature() is not None:
                qh, kh_new, vh_new = self._shared_int8_qkv_projection(x)
            elif self._uses_module_runtime_linear():
                q_lin = runtime_linear_module(x, self.w_q)
                k_lin = runtime_linear_module(x, self.w_k)
                v_lin = runtime_linear_module(x, self.w_v)
                qh = runtime_split_heads(q_lin, self.n_heads)
                kh_new = runtime_split_heads(k_lin, self.n_kv_heads)
                vh_new = runtime_split_heads(v_lin, self.n_kv_heads)
            else:
                q_weight, q_bias = runtime_resolve_linear_module_tensors(self.w_q, reference=x)
                k_weight, k_bias = runtime_resolve_linear_module_tensors(self.w_k, reference=x)
                v_weight, v_bias = runtime_resolve_linear_module_tensors(self.w_v, reference=x)
                qh, kh_new, vh_new = runtime_qkv_heads_projection(
                    x,
                    q_weight,
                    q_bias,
                    k_weight,
                    k_bias,
                    v_weight,
                    v_bias,
                    q_heads=self.n_heads,
                    kv_heads=self.n_kv_heads,
                )
        else:
            q_lin = runtime_linear_module(x, self.w_q)
            k_lin = runtime_linear_module(x if k is None else k, self.w_k)
            v_lin = runtime_linear_module(x if v is None else v, self.w_v)
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

        parity_exact = os.environ.get("ATTN_PARITY_EXACT", "0") == "1"
        appended_to_cache = False
        add = None
        out = None
        if (
            cache is not None
            and T == 1
            and not parity_exact
            and hasattr(cache, "paged_attention_decode")
            and getattr(cache, "supports_paged_attention_decode", lambda: False)()
            and (self.attn_dropout_p if self.training else 0.0) == 0.0
        ):
            S = int(cache.length()) + T
            add = runtime_prepare_attention_mask(
                mask,
                batch_size=B,
                num_heads=self.n_heads,
                tgt_len=T,
                src_len=S,
                position_ids=position_ids,
            )
            try:
                out = cache.paged_attention_decode(
                    qh,
                    kh_new,
                    vh_new,
                    attn_mask=add,
                    scale=float(self.scaling),
                )
                appended_to_cache = True
            except Exception:
                if trace:
                    print("[attn] exception in paged decode attention path; falling back")
                    traceback.print_exc()
                out = None

        if out is None:
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
                prefer_library_backend = prefer_hopper_library_attention(
                    device=device,
                    dtype=dtype,
                    q_seq=T,
                    kv_seq=S,
                    forced_backend=forced_backend,
                )
                use_runtime_attention = (
                    forced_backend is None
                    and (self.attn_dropout_p if self.training else 0.0) == 0.0
                    and not prefer_library_backend
                )
                if use_runtime_attention:
                    kh_backend, vh_backend = kh_all, vh_all
                    if self._supports_int8_attention_core():
                        out = self._int8_attention(
                            qh,
                            kh_backend,
                            vh_backend,
                            attn_mask=add,
                            is_causal=is_causal_flag,
                            scale=float(self.scaling),
                        )
                    else:
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
                        kv_seq=S,
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
            packed_o_spec = self._ensure_packed_output(packed_backend, out)
            return runtime_head_output_packed_projection(out, packed_o_spec, backend=packed_backend)
        runtime_linear = getattr(self.w_o, "runtime_linear", None)
        if callable(runtime_linear):
            return runtime_linear_module(runtime_merge_heads(out), self.w_o)
        o_weight, o_bias = runtime_resolve_linear_module_tensors(self.w_o, reference=out)
        return runtime_head_output_projection(out, o_weight, o_bias)


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
