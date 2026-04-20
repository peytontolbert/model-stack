from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import attn.eager as eager_mod
import compress.quantization as quant_mod
import runtime.ops as runtime_ops_mod
import runtime.quant as runtime_quant_mod
from attn.eager import EagerAttention
from specs.config import ModelConfig


def _build_cfg(attn_dropout: float = 0.0) -> ModelConfig:
    return ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        vocab_size=64,
        dtype="float32",
        attn_dropout=attn_dropout,
    )


def _stub_output_projection(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, backend: str | None = None) -> torch.Tensor:
    del weight, bias, backend
    return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)


def test_eager_attention_prefers_runtime_attention_in_eval(monkeypatch):
    monkeypatch.delenv("ATTN_BACKEND", raising=False)
    monkeypatch.delenv("ATTN_BACKEND_FILE", raising=False)
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()
    calls = {"runtime": 0, "legacy": 0}

    def fake_runtime(q, k, v, attn_mask=None, is_causal=False, scale=None):
        del k, v, attn_mask, is_causal, scale
        calls["runtime"] += 1
        return torch.zeros_like(q)

    def fail_legacy(*args, **kwargs):
        del args, kwargs
        calls["legacy"] += 1
        raise AssertionError("legacy attention path should not be used on the default eval path")

    monkeypatch.setattr(eager_mod, "runtime_attention", fake_runtime)
    monkeypatch.setattr(eager_mod, "scaled_dot_product_attention", fail_legacy)
    monkeypatch.setattr(eager_mod, "prefer_hopper_library_attention", lambda **kwargs: False)
    monkeypatch.setattr(eager_mod, "runtime_head_output_projection", _stub_output_projection)

    x = torch.randn(2, 3, 16)
    y = attn(x, None, None, None)

    assert calls["runtime"] == 1
    assert calls["legacy"] == 0
    assert y.shape == x.shape


def test_eager_attention_respects_explicit_backend_override(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False, backend_override="torch")
    attn.eval()
    calls = {"runtime": 0, "legacy": 0}

    def fail_runtime(*args, **kwargs):
        del args, kwargs
        calls["runtime"] += 1
        raise AssertionError("runtime attention path should not run when a backend is explicitly forced")

    def fake_legacy(q, k, v, attn_mask=None, dropout_p=0.0, backend=None, is_causal=None, scale=None):
        del k, v, attn_mask, dropout_p, backend, is_causal, scale
        calls["legacy"] += 1
        return torch.zeros_like(q)

    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime)
    monkeypatch.setattr(eager_mod, "scaled_dot_product_attention", fake_legacy)
    monkeypatch.setattr(eager_mod, "runtime_head_output_projection", _stub_output_projection)

    x = torch.randn(2, 3, 16)
    y = attn(x, None, None, None)

    assert calls["runtime"] == 0
    assert calls["legacy"] == 1
    assert y.shape == x.shape


def test_eager_attention_training_dropout_uses_legacy_path(monkeypatch):
    monkeypatch.delenv("ATTN_BACKEND", raising=False)
    monkeypatch.delenv("ATTN_BACKEND_FILE", raising=False)
    attn = EagerAttention(_build_cfg(attn_dropout=0.1), use_rope=False)
    attn.train()
    calls = {"runtime": 0, "legacy": 0}

    def fail_runtime(*args, **kwargs):
        del args, kwargs
        calls["runtime"] += 1
        raise AssertionError("runtime attention path should not run when training-time dropout is active")

    def fake_legacy(q, k, v, attn_mask=None, dropout_p=0.0, backend=None, is_causal=None, scale=None):
        del k, v, attn_mask, backend, is_causal, scale
        calls["legacy"] += 1
        assert float(dropout_p) == 0.1
        return torch.zeros_like(q)

    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime)
    monkeypatch.setattr(eager_mod, "scaled_dot_product_attention", fake_legacy)
    monkeypatch.setattr(eager_mod, "select_attention_backend", lambda **kwargs: "torch")
    monkeypatch.setattr(eager_mod, "prefer_hopper_library_attention", lambda **kwargs: False)
    monkeypatch.setattr(eager_mod, "runtime_head_output_projection", _stub_output_projection)

    x = torch.randn(2, 3, 16)
    y = attn(x, None, None, None)

    assert calls["runtime"] == 0
    assert calls["legacy"] == 1
    assert y.shape == x.shape


def test_eager_attention_internal_rope_cache_uses_runtime_resolution(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=True)
    seen = {}

    def fake_resolve(*, reference, head_dim, base_theta, attention_scaling=1.0, position_ids=None):
        seen["reference_shape"] = tuple(reference.shape)
        seen["head_dim"] = head_dim
        seen["base_theta"] = base_theta
        seen["attention_scaling"] = attention_scaling
        seen["position_ids"] = position_ids
        cos = torch.zeros(reference.shape[1], head_dim, dtype=reference.dtype, device=reference.device)
        sin = torch.ones(reference.shape[1], head_dim, dtype=reference.dtype, device=reference.device)
        return cos, sin

    monkeypatch.setattr(eager_mod, "runtime_resolve_rotary_embedding", fake_resolve)

    x = torch.randn(2, 3, 16)
    attn._ensure_rope_cache(3, x)

    assert seen["reference_shape"] == (2, 3, 16)
    assert seen["head_dim"] == 4
    assert seen["base_theta"] == attn.rope_theta
    assert seen["attention_scaling"] == attn.rope_attention_scaling
    assert seen["position_ids"] is None
    assert attn._rope_cos.shape == (3, 4)
    assert attn._rope_sin.shape == (3, 4)


def test_eager_attention_grad_enabled_training_path_stays_eager_safe(monkeypatch):
    monkeypatch.delenv("ATTN_BACKEND", raising=False)
    monkeypatch.delenv("ATTN_BACKEND_FILE", raising=False)
    attn = EagerAttention(_build_cfg(attn_dropout=0.0), use_rope=True)
    attn.train()

    class FakeModule:
        def __getattr__(self, name):
            if name.endswith("_forward"):
                raise AssertionError(f"native {name} should not run on grad-enabled attention path")
            raise AttributeError(name)

    monkeypatch.setattr(
        runtime_ops_mod,
        "has_native_op",
        lambda name: name in {
            "linear",
            "split_heads",
            "merge_heads",
            "qkv_projection",
            "qkv_heads_projection",
            "head_output_projection",
            "rope",
            "attention_prefill",
            "attention_decode",
        },
    )
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(eager_mod, "prefer_hopper_library_attention", lambda **kwargs: False)

    x = torch.randn(2, 3, 16, requires_grad=True)
    y = attn(x, None, None, None)

    assert y.shape == x.shape

    y.sum().backward()
    assert x.grad is not None
    assert attn.w_q.weight.grad is not None
    assert attn.w_k.weight.grad is not None
    assert attn.w_v.weight.grad is not None
    assert attn.w_o.weight.grad is not None


def test_eager_attention_packed_backend_requires_runtime_linear_opt_in(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()

    class FakeRef:
        is_cuda = True

    class RuntimeOnlyProjection(torch.nn.Module):
        def runtime_linear(self, x, *, backend=None):
            del x, backend
            raise AssertionError("should not be called in helper test")

    class RuntimePackedProjection(RuntimeOnlyProjection):
        def runtime_supports_packed_backend(self, backend: str) -> bool:
            return str(backend) == "cublaslt"

    monkeypatch.setattr(eager_mod, "resolve_linear_backend", lambda requested="auto": "cublaslt")

    attn.w_q = RuntimeOnlyProjection()
    attn.w_k = RuntimeOnlyProjection()
    attn.w_v = RuntimeOnlyProjection()
    attn.w_o = RuntimeOnlyProjection()
    assert attn._packed_backend(FakeRef()) is None

    attn.w_q = RuntimePackedProjection()
    attn.w_k = RuntimePackedProjection()
    attn.w_v = RuntimePackedProjection()
    attn.w_o = RuntimePackedProjection()
    assert attn._packed_backend(FakeRef()) == "cublaslt"


def test_eager_attention_prefers_library_backend_on_hopper(monkeypatch):
    monkeypatch.delenv("ATTN_BACKEND", raising=False)
    monkeypatch.delenv("ATTN_BACKEND_FILE", raising=False)
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()
    calls = {"runtime": 0, "legacy": 0}

    class FakeRef:
        is_cuda = True

    class RuntimeOnlyProjection(torch.nn.Module):
        def runtime_linear(self, x, *, backend=None):
            del x, backend
            raise AssertionError("should not be called in helper test")

    def fail_runtime(*args, **kwargs):
        del args, kwargs
        calls["runtime"] += 1
        raise AssertionError("generic runtime attention path should be skipped when Hopper fast path is preferred")

    def fake_legacy(q, k, v, attn_mask=None, dropout_p=0.0, backend=None, is_causal=None, scale=None):
        del k, v, attn_mask, dropout_p, is_causal, scale
        calls["legacy"] += 1
        assert backend == "torch"
        return torch.zeros_like(q)

    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime)
    monkeypatch.setattr(eager_mod, "scaled_dot_product_attention", fake_legacy)
    monkeypatch.setattr(eager_mod, "prefer_hopper_library_attention", lambda **kwargs: True)
    monkeypatch.setattr(eager_mod, "select_attention_backend", lambda **kwargs: "torch")
    monkeypatch.setattr(eager_mod, "runtime_head_output_projection", _stub_output_projection)

    x = torch.randn(2, 128, 16)
    y = attn(x, None, None, None)

    assert calls["runtime"] == 0
    assert calls["legacy"] == 1
    assert y.shape == x.shape

    class BitNetPackedProjection(RuntimeOnlyProjection):
        def runtime_supports_packed_backend(self, backend: str) -> bool:
            return str(backend) == "bitnet"

    monkeypatch.setattr(eager_mod, "resolve_linear_backend", lambda requested="auto": "aten")

    attn.w_q = BitNetPackedProjection()
    attn.w_k = BitNetPackedProjection()
    attn.w_v = BitNetPackedProjection()
    attn.w_o = BitNetPackedProjection()
    assert attn._packed_backend(FakeRef()) == "bitnet"
    assert attn._packed_output_backend(FakeRef()) == "bitnet"


def test_eager_attention_bitnet_wrappers_use_module_runtime_projections(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()

    monkeypatch.setattr(eager_mod, "runtime_attention", lambda q, k, v, attn_mask=None, is_causal=False, scale=None: torch.zeros_like(q))
    monkeypatch.setattr(
        eager_mod,
        "runtime_head_output_projection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed output projection should not run for BitNet wrappers")),
    )

    calls = []

    def fake_runtime_bitnet_linear(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None):
        del packed_weight, scale_values, segment_offsets, bias
        out_features = int(layout_header[3].item())
        calls.append((tuple(x.shape), out_features))
        return torch.full(x.shape[:-1] + (out_features,), float(len(calls)), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(quant_mod, "runtime_bitnet_linear", fake_runtime_bitnet_linear)

    attn.w_q = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(attn.w_q)
    attn.w_k = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(attn.w_k)
    attn.w_v = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(attn.w_v)
    attn.w_o = quant_mod.QuantizedLinearBitNet(attn.n_heads * attn.head_dim, attn.d_model, bias=True).from_float(attn.w_o)

    x = torch.randn(2, 3, attn.d_model)
    with torch.no_grad():
        y = attn(x, None, None, None)

    assert y.shape == x.shape
    assert len(calls) == 4
    assert calls[0] == ((2, 3, attn.d_model), attn.n_heads * attn.head_dim)
    assert calls[1] == ((2, 3, attn.d_model), attn.n_kv_heads * attn.head_dim)
    assert calls[2] == ((2, 3, attn.d_model), attn.n_kv_heads * attn.head_dim)
    assert calls[3] == ((2, 3, attn.d_model), attn.d_model)


def test_eager_attention_bitnet_wrappers_use_packed_spec_path_when_enabled(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()
    monkeypatch.setattr(attn, "_packed_backend", lambda x: "bitnet")
    monkeypatch.setattr(attn, "_packed_output_backend", lambda x: None)
    monkeypatch.setattr(eager_mod, "runtime_attention", lambda q, k, v, attn_mask=None, is_causal=False, scale=None: torch.zeros_like(q))
    orig_has_native_op = runtime_ops_mod.has_native_op
    monkeypatch.setattr(
        runtime_ops_mod,
        "has_native_op",
        lambda name: False if name == "bitnet_fused_qkv_packed_heads_projection" else orig_has_native_op(name),
    )

    calls = []

    def fake_runtime_bitnet_linear(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None):
        del packed_weight, scale_values, segment_offsets, bias
        out_features = int(layout_header[3].item())
        calls.append((tuple(x.shape), out_features))
        return torch.full(x.shape[:-1] + (out_features,), float(len(calls)), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "bitnet_linear", fake_runtime_bitnet_linear)

    attn.w_q = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(attn.w_q)
    attn.w_k = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(attn.w_k)
    attn.w_v = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(attn.w_v)
    attn.w_o = quant_mod.QuantizedLinearBitNet(attn.n_heads * attn.head_dim, attn.d_model, bias=True).from_float(attn.w_o)

    x = torch.randn(2, 3, attn.d_model)
    with torch.no_grad():
        y = attn(x, None, None, None)

    assert y.shape == x.shape
    assert len(calls) == 1
    assert calls[0] == ((2, 3, attn.d_model), attn.n_heads * attn.head_dim + 2 * attn.n_kv_heads * attn.head_dim)


def test_eager_attention_bitnet_wrappers_use_native_packed_qkv_when_available(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()
    monkeypatch.setattr(attn, "_packed_backend", lambda x: "bitnet")
    monkeypatch.setattr(attn, "_packed_output_backend", lambda x: None)
    monkeypatch.setattr(eager_mod, "runtime_attention", lambda q, k, v, attn_mask=None, is_causal=False, scale=None: torch.zeros_like(q))

    native_calls = {"qkv": 0}
    output_calls = []

    class FakeNativeModule:
        def bitnet_fused_qkv_packed_heads_projection_forward(
            self,
            x,
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
            packed_bias,
            q_size,
            k_size,
            v_size,
            q_heads,
            kv_heads,
        ):
            del packed_weight, scale_values, segment_offsets, packed_bias, layout_header
            native_calls["qkv"] += 1
            q_width = int(q_size) // int(q_heads)
            kv_width = int(k_size) // int(kv_heads)
            return (
                torch.full((x.shape[0], int(q_heads), x.shape[1], q_width), 1.0, dtype=x.dtype, device=x.device),
                torch.full((x.shape[0], int(kv_heads), x.shape[1], kv_width), 2.0, dtype=x.dtype, device=x.device),
                torch.full((x.shape[0], int(kv_heads), x.shape[1], int(v_size) // int(kv_heads)), 3.0, dtype=x.dtype, device=x.device),
            )

    def fake_runtime_bitnet_linear(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None):
        del packed_weight, scale_values, segment_offsets, bias
        out_features = int(layout_header[3].item())
        output_calls.append((tuple(x.shape), out_features))
        return torch.full(x.shape[:-1] + (out_features,), 4.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "bitnet_fused_qkv_packed_heads_projection")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeNativeModule())
    monkeypatch.setattr(runtime_quant_mod, "bitnet_linear", fake_runtime_bitnet_linear)

    attn.w_q = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(attn.w_q)
    attn.w_k = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(attn.w_k)
    attn.w_v = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(attn.w_v)
    attn.w_o = quant_mod.QuantizedLinearBitNet(attn.n_heads * attn.head_dim, attn.d_model, bias=True).from_float(attn.w_o)

    x = torch.randn(2, 3, attn.d_model)
    with torch.no_grad():
        y = attn(x, None, None, None)

    assert y.shape == x.shape
    assert native_calls["qkv"] == 1
    assert output_calls == []


def test_eager_attention_bitnet_w2a8_packed_backend_uses_fused_int8_qkv_and_int8_attention(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()
    monkeypatch.setattr(attn, "_packed_backend", lambda x: "bitnet")
    monkeypatch.setattr(attn, "_packed_output_backend", lambda x: "bitnet")

    def fail_runtime_attention(*args, **kwargs):
        del args, kwargs
        raise AssertionError("BitNet W2A8 attention should use the int8 attention core")

    calls = {"resolve_qkv": [], "packed_qkv": [], "resolve_o": [], "packed_o": 0, "runtime_linear": [], "int8_attention": 0}

    def fake_int8_attention(q, k, v, attn_mask=None, *, is_causal=False, scale=None, out_dtype=None, q_scale=None, k_scale=None, v_scale=None):
        del attn_mask, is_causal, scale, q_scale, k_scale, v_scale
        calls["int8_attention"] += 1
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.zeros_like(v, dtype=dtype)

    def fake_resolve_packed_qkv_module_spec(q_module, k_module, v_module, *, backend=None, reference=None, dtype=None, device=None):
        del q_module, k_module, v_module, dtype, device
        calls["resolve_qkv"].append((backend, None if reference is None else tuple(reference.shape)))
        return {
            "format": "bitnet_qkv_fused_int8",
            "backend": "bitnet",
            "qweight": torch.zeros(
                attn.n_heads * attn.head_dim + 2 * (attn.n_kv_heads * attn.head_dim),
                attn.d_model,
                dtype=torch.int8,
            ),
            "inv_scale": torch.ones(
                attn.n_heads * attn.head_dim + 2 * (attn.n_kv_heads * attn.head_dim),
                dtype=torch.float32,
            ),
            "packed_bias": None,
            "q_size": attn.n_heads * attn.head_dim,
            "k_size": attn.n_kv_heads * attn.head_dim,
            "v_size": attn.n_kv_heads * attn.head_dim,
            "spin_enabled": False,
            "spin_signs": attn.w_q.spin_signs,
            "pre_scale": attn.w_q.pre_scale,
            "act_quant_mode": "dynamic_int8",
            "act_quant_method": "absmax",
            "act_quant_bits": 6,
            "act_quant_percentile": 0.999,
            "act_scale": attn.w_q.act_scale,
        }

    def fake_qkv_packed_spec_heads_projection(x, spec, *, q_heads, kv_heads, backend=None):
        calls["packed_qkv"].append((str(spec.get("format")), backend, tuple(x.shape), q_heads, kv_heads))
        return (
            torch.zeros((x.shape[0], q_heads, x.shape[1], attn.head_dim), dtype=x.dtype, device=x.device),
            torch.zeros((x.shape[0], kv_heads, x.shape[1], attn.head_dim), dtype=x.dtype, device=x.device),
            torch.zeros((x.shape[0], kv_heads, x.shape[1], attn.head_dim), dtype=x.dtype, device=x.device),
        )

    def fake_resolve_packed_linear_module_spec(module, *, backend=None, reference=None, dtype=None, device=None):
        del module, dtype, device
        calls["resolve_o"].append((backend, None if reference is None else tuple(reference.shape)))
        return {
            "format": "bitnet_w2a8_int8",
            "backend": "bitnet",
            "qweight": torch.zeros(attn.d_model, attn.n_heads * attn.head_dim, dtype=torch.int8),
            "inv_scale": torch.ones(attn.d_model, dtype=torch.float32),
            "bias": None,
            "spin_enabled": False,
            "spin_signs": attn.w_o.spin_signs,
            "pre_scale": attn.w_o.pre_scale,
            "act_quant_mode": "dynamic_int8",
            "act_quant_method": "absmax",
            "act_quant_bits": 6,
            "act_quant_percentile": 0.999,
            "act_scale": attn.w_o.act_scale,
        }

    def fake_head_output_packed_projection(x, spec, *, backend=None):
        calls["packed_o"] += 1
        assert str(spec.get("format")) == "bitnet_w2a8_int8"
        assert backend == "bitnet"
        return torch.zeros((x.shape[0], x.shape[2], attn.d_model), dtype=x.dtype, device=x.device)

    def fake_runtime_linear_module(x, module, *, backend=None):
        del backend
        assert module is attn.w_o
        calls["runtime_linear"].append(tuple(x.shape))
        return torch.zeros((x.shape[0], x.shape[1], attn.d_model), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(eager_mod, "runtime_resolve_packed_qkv_module_spec", fake_resolve_packed_qkv_module_spec)
    monkeypatch.setattr(eager_mod, "runtime_qkv_packed_spec_heads_projection", fake_qkv_packed_spec_heads_projection)
    monkeypatch.setattr(eager_mod, "runtime_resolve_packed_linear_module_spec", fake_resolve_packed_linear_module_spec)
    monkeypatch.setattr(eager_mod, "runtime_head_output_packed_projection", fake_head_output_packed_projection)
    monkeypatch.setattr(eager_mod, "runtime_linear_module", fake_runtime_linear_module)
    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime_attention)
    monkeypatch.setattr(eager_mod, "runtime_int8_attention", fake_int8_attention)

    attn.w_q = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(
        attn.w_q,
        activation_quant="dynamic_int8",
        activation_quant_bits=6,
    )
    attn.w_k = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_k,
        activation_quant="dynamic_int8",
        activation_quant_bits=6,
    )
    attn.w_v = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_v,
        activation_quant="dynamic_int8",
        activation_quant_bits=6,
    )
    attn.w_o = quant_mod.QuantizedLinearBitNet(attn.n_heads * attn.head_dim, attn.d_model, bias=True).from_float(
        attn.w_o,
        activation_quant="dynamic_int8",
        activation_quant_bits=6,
    )
    monkeypatch.setattr(
        attn,
        "_shared_int8_qkv_projection",
        lambda x: (_ for _ in ()).throw(AssertionError("packed BitNet W2A8 path should not use shared qkv projection")),
    )

    x = torch.randn(2, 3, attn.d_model)
    with torch.no_grad():
        y = attn(x, None, None, None)

    assert y.shape == x.shape
    assert calls["resolve_qkv"] == [("bitnet", (2, 3, attn.d_model))]
    assert calls["packed_qkv"] == [("bitnet_qkv_fused_int8", "bitnet", (2, 3, attn.d_model), attn.n_heads, attn.n_kv_heads)]
    assert calls["resolve_o"] == [("bitnet", (2, attn.n_heads, 3, attn.head_dim))]
    assert calls["packed_o"] == 1
    assert calls["runtime_linear"] == []
    assert calls["int8_attention"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for single-token BitNet W2A8 decode path")
def test_eager_attention_bitnet_w2a8_single_token_prefers_packed_qkv_over_runtime_linear(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False).to(device="cuda")
    attn.eval()
    monkeypatch.setattr(attn, "_packed_backend", lambda x: "bitnet")
    monkeypatch.setattr(attn, "_packed_output_backend", lambda x: None)
    monkeypatch.setattr(eager_mod, "has_native_op", lambda name: name == "int8_linear_from_float")
    monkeypatch.setattr(
        eager_mod,
        "native_module",
        lambda: type("FakeNativeModule", (), {"int8_linear_from_float_forward": object()})(),
    )
    monkeypatch.setattr(eager_mod, "runtime_attention", lambda q, k, v, attn_mask=None, is_causal=False, scale=None: torch.zeros_like(v))

    calls = {"resolve": [], "packed": [], "runtime_linear": []}

    def fake_resolve_packed_qkv(q_module, k_module, v_module, *, backend=None, reference=None, dtype=None, device=None):
        del dtype, device
        calls["resolve"].append((q_module, k_module, v_module, backend, tuple(reference.shape)))
        return {
            "format": "bitnet_qkv_fused_int8",
            "backend": "bitnet",
            "q_size": attn.n_heads * attn.head_dim,
            "k_size": attn.n_kv_heads * attn.head_dim,
            "v_size": attn.n_kv_heads * attn.head_dim,
        }

    def fake_packed_qkv(x, spec, *, q_heads, kv_heads, backend=None):
        calls["packed"].append((tuple(x.shape), spec["format"], q_heads, kv_heads, backend))
        q = torch.zeros((x.shape[0], attn.n_heads, x.shape[1], attn.head_dim), dtype=x.dtype, device=x.device)
        k = torch.zeros((x.shape[0], attn.n_kv_heads, x.shape[1], attn.head_dim), dtype=x.dtype, device=x.device)
        v = torch.zeros((x.shape[0], attn.n_kv_heads, x.shape[1], attn.head_dim), dtype=x.dtype, device=x.device)
        return q, k, v

    def fake_runtime_linear_module(x, module, *, backend=None):
        calls["runtime_linear"].append((module, tuple(x.shape), backend))
        out_features = module.out_features
        return torch.zeros((*x.shape[:-1], out_features), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(eager_mod, "runtime_resolve_packed_qkv_module_spec", fake_resolve_packed_qkv)
    monkeypatch.setattr(eager_mod, "runtime_qkv_packed_spec_heads_projection", fake_packed_qkv)
    monkeypatch.setattr(eager_mod, "runtime_linear_module", fake_runtime_linear_module)

    attn.w_q = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(
        attn.w_q,
        activation_quant="dynamic_int8",
        activation_quant_bits=6,
    )
    attn.w_k = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_k,
        activation_quant="dynamic_int8",
        activation_quant_bits=6,
    )
    attn.w_v = quant_mod.QuantizedLinearBitNet(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_v,
        activation_quant="dynamic_int8",
        activation_quant_bits=6,
    )
    attn.w_o = quant_mod.QuantizedLinearBitNet(attn.n_heads * attn.head_dim, attn.d_model, bias=True).from_float(
        attn.w_o,
        activation_quant="dynamic_int8",
        activation_quant_bits=6,
    )

    x = torch.randn(2, 1, attn.d_model, device="cuda")
    with torch.no_grad():
        y = attn(x, None, None, None)

    assert y.shape == x.shape
    assert calls["resolve"] == [
        (attn.w_q, attn.w_k, attn.w_v, "bitnet", (2, 1, attn.d_model)),
    ]
    assert calls["packed"] == [
        ((2, 1, attn.d_model), "bitnet_qkv_fused_int8", attn.n_heads, attn.n_kv_heads, "bitnet"),
    ]
    assert calls["runtime_linear"] == [
        (attn.w_o, (2, 1, attn.n_heads * attn.head_dim), None),
    ]


def test_eager_attention_int8_wrappers_share_qkv_input_quantization(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()
    monkeypatch.setattr(eager_mod, "runtime_attention", lambda q, k, v, attn_mask=None, is_causal=False, scale=None: torch.zeros_like(q))

    calls = {"quantize": [], "project": []}

    def fake_quantize_activation_int8_rowwise(x, *, scale=None, method="absmax", percentile=0.999, eps=1e-8):
        del scale, method, percentile, eps
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        qx = torch.ones_like(x, dtype=torch.int8)
        calls["quantize"].append(int(qx.data_ptr()))
        return qx, torch.full((rows,), 0.25, dtype=torch.float32, device=x.device)

    def fake_int8_linear_from_quantized_activation(qx, x_scale, qweight, inv_scale, bias=None, *, out_dtype=None):
        del x_scale, inv_scale, bias
        calls["project"].append((int(qx.data_ptr()), int(qweight.shape[0]), out_dtype))
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.full((*qx.shape[:-1], qweight.shape[0]), float(len(calls["project"])), dtype=dtype, device=qx.device)

    monkeypatch.setattr(quant_mod, "runtime_quantize_activation_int8_rowwise", fake_quantize_activation_int8_rowwise)
    monkeypatch.setattr(quant_mod, "runtime_int8_linear_from_quantized_activation", fake_int8_linear_from_quantized_activation)
    monkeypatch.setattr(
        eager_mod,
        "runtime_int8_attention",
        lambda q, k, v, **kwargs: torch.zeros_like(v, dtype=torch.float32),
    )

    attn.w_q = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(
        attn.w_q,
        activation_quant="dynamic_int8",
    )
    attn.w_k = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_k,
        activation_quant="dynamic_int8",
    )
    attn.w_v = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_v,
        activation_quant="dynamic_int8",
    )
    attn.w_o = quant_mod.QuantizedLinearInt8(attn.n_heads * attn.head_dim, attn.d_model, bias=True).from_float(
        attn.w_o,
        activation_quant="dynamic_int8",
    )

    x = torch.randn(2, 3, attn.d_model)
    with torch.no_grad():
        y = attn(x, None, None, None)

    assert y.shape == x.shape
    assert len(calls["quantize"]) == 1
    assert len(calls["project"]) == 3
    assert calls["project"][0][0] == calls["project"][1][0] == calls["project"][2][0]
    assert calls["project"][0][1] == attn.n_heads * attn.head_dim
    assert calls["project"][1][1] == attn.n_kv_heads * attn.head_dim
    assert calls["project"][2][1] == attn.n_kv_heads * attn.head_dim


def test_eager_attention_int8_wrappers_share_qkv_input_quantization_when_irrelevant_metadata_differs(monkeypatch):
    monkeypatch.setattr(
        eager_mod,
        "runtime_attention",
        lambda q, k, v, attn_mask=None, is_causal=False, scale=None: torch.zeros_like(q),
    )

    def fake_quantize_activation_int8_rowwise(x, *, scale=None, method="absmax", percentile=0.999, eps=1e-8):
        del scale, method, percentile, eps
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        qx = torch.ones_like(x, dtype=torch.int8)
        fake_quantize_activation_int8_rowwise.calls.append(int(qx.data_ptr()))
        return qx, torch.full((rows,), 0.25, dtype=torch.float32, device=x.device)

    fake_quantize_activation_int8_rowwise.calls = []

    def fake_int8_linear_from_quantized_activation(qx, x_scale, qweight, inv_scale, bias=None, *, out_dtype=None):
        del x_scale, inv_scale, bias
        fake_int8_linear_from_quantized_activation.calls.append((int(qx.data_ptr()), int(qweight.shape[0]), out_dtype))
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.full((*qx.shape[:-1], qweight.shape[0]), float(len(fake_int8_linear_from_quantized_activation.calls)), dtype=dtype, device=qx.device)

    fake_int8_linear_from_quantized_activation.calls = []

    monkeypatch.setattr(quant_mod, "runtime_quantize_activation_int8_rowwise", fake_quantize_activation_int8_rowwise)
    monkeypatch.setattr(quant_mod, "runtime_int8_linear_from_quantized_activation", fake_int8_linear_from_quantized_activation)
    monkeypatch.setattr(
        eager_mod,
        "runtime_int8_attention",
        lambda q, k, v, **kwargs: torch.zeros_like(v, dtype=torch.float32),
    )

    def _run_case(mode: str) -> None:
        fake_quantize_activation_int8_rowwise.calls.clear()
        fake_int8_linear_from_quantized_activation.calls.clear()

        attn = EagerAttention(_build_cfg(), use_rope=False)
        attn.eval()
        attn.w_q = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(
            attn.w_q,
            activation_quant=mode,
        )
        attn.w_k = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
            attn.w_k,
            activation_quant=mode,
        )
        attn.w_v = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
            attn.w_v,
            activation_quant=mode,
        )
        attn.w_o = quant_mod.QuantizedLinearInt8(attn.n_heads * attn.head_dim, attn.d_model, bias=True).from_float(
            attn.w_o,
            activation_quant=mode,
        )
        if mode == "dynamic_int8":
            attn.w_q.act_scale.fill_(0.125)
            attn.w_k.act_scale.fill_(0.25)
            attn.w_v.act_scale.fill_(0.5)
        else:
            for projection in (attn.w_q, attn.w_k, attn.w_v):
                projection.act_scale.fill_(0.125)
            attn.w_q.act_quant_method = "absmax"
            attn.w_k.act_quant_method = "percentile"
            attn.w_v.act_quant_method = "mse"
            attn.w_q.act_quant_percentile = 0.999
            attn.w_k.act_quant_percentile = 0.95
            attn.w_v.act_quant_percentile = 0.9

        x = torch.randn(2, 3, attn.d_model)
        with torch.no_grad():
            y = attn(x, None, None, None)

        assert y.shape == x.shape
        assert len(fake_quantize_activation_int8_rowwise.calls) == 1
        assert len(fake_int8_linear_from_quantized_activation.calls) == 3
        assert (
            fake_int8_linear_from_quantized_activation.calls[0][0]
            == fake_int8_linear_from_quantized_activation.calls[1][0]
            == fake_int8_linear_from_quantized_activation.calls[2][0]
        )

    _run_case("dynamic_int8")
    _run_case("static_int8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused int8 attention dispatch")
def test_eager_attention_prefers_cuda_int8_runtime_linear_over_shared_qkv_quantization(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False).to(device="cuda", dtype=torch.float16)
    attn.eval()

    monkeypatch.setattr(
        eager_mod,
        "runtime_int8_attention",
        lambda q, k, v, **kwargs: torch.zeros_like(v, dtype=torch.float16),
    )
    monkeypatch.setattr(eager_mod, "runtime_head_output_projection", _stub_output_projection)

    def fail_quantize_activation_int8_rowwise(x, *, scale=None, method="absmax", percentile=0.999, eps=1e-8):
        del x, scale, method, percentile, eps
        raise AssertionError("shared int8 qkv quantization path should not run on CUDA when the fused frontend is available")

    calls = {"runtime_linear": 0}

    def fake_runtime_int8_linear(x, qweight, inv_scale, bias=None, *, act_scale=None, act_method="absmax", act_percentile=0.999):
        del qweight, inv_scale, bias, act_scale, act_method, act_percentile
        calls["runtime_linear"] += 1
        return torch.zeros((*x.shape[:-1], attn.n_heads * attn.head_dim), dtype=x.dtype, device=x.device)

    class FakeNativeModule:
        int8_linear_from_float_forward = object()

    monkeypatch.setattr(quant_mod, "runtime_quantize_activation_int8_rowwise", fail_quantize_activation_int8_rowwise)
    monkeypatch.setattr(quant_mod, "runtime_int8_linear", fake_runtime_int8_linear)
    monkeypatch.setattr(eager_mod, "has_native_op", lambda name: name == "int8_linear_from_float")
    monkeypatch.setattr(eager_mod, "native_module", lambda: FakeNativeModule())

    attn.w_q = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(
        attn.w_q,
        activation_quant="dynamic_int8",
    ).to(device="cuda")
    attn.w_k = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_k,
        activation_quant="dynamic_int8",
    ).to(device="cuda")
    attn.w_v = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_v,
        activation_quant="dynamic_int8",
    ).to(device="cuda")
    attn.w_q.act_scale.fill_(0.125)
    attn.w_k.act_scale.fill_(0.25)
    attn.w_v.act_scale.fill_(0.5)

    x = torch.randn(2, 3, attn.d_model, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        y = attn(x, None, None, None)

    assert y.shape == x.shape
    assert calls["runtime_linear"] == 3


def test_eager_attention_int8_wrappers_use_int8_attention_core(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()

    calls = {"int8_attention": 0}

    def fail_runtime_attention(*args, **kwargs):
        del args, kwargs
        raise AssertionError("float runtime attention path should not run for int8 attention-core path")

    def fake_int8_attention(q, k, v, attn_mask=None, *, is_causal=False, scale=None, out_dtype=None, q_scale=None, k_scale=None, v_scale=None):
        del attn_mask, scale, q_scale, k_scale, v_scale
        calls["int8_attention"] += 1
        assert q.dtype == torch.float32
        assert k.dtype == torch.float32
        assert v.dtype == torch.float32
        assert is_causal is True
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.zeros(q.shape, dtype=dtype, device=q.device)

    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime_attention)
    monkeypatch.setattr(eager_mod, "runtime_int8_attention", fake_int8_attention)

    attn.w_q = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(
        attn.w_q,
        activation_quant="dynamic_int8",
    )
    attn.w_k = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_k,
        activation_quant="dynamic_int8",
    )
    attn.w_v = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_v,
        activation_quant="dynamic_int8",
    )
    attn.w_o = quant_mod.QuantizedLinearInt8(attn.n_heads * attn.head_dim, attn.d_model, bias=True).from_float(
        attn.w_o,
        activation_quant="dynamic_int8",
    )

    x = torch.randn(2, 3, attn.d_model)
    with torch.no_grad():
        y = attn(x, None, None, None)

    assert y.shape == x.shape
    assert calls["int8_attention"] == 1


def test_eager_attention_int8_wrappers_use_int8_attention_core_with_mask(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()

    calls = {"int8_attention": 0, "mask_shape": None, "is_causal": None}

    def fail_runtime_attention(*args, **kwargs):
        del args, kwargs
        raise AssertionError("float runtime attention path should not run for masked int8 attention-core path")

    def fake_int8_attention(q, k, v, attn_mask=None, *, is_causal=False, scale=None, out_dtype=None, q_scale=None, k_scale=None, v_scale=None):
        del scale, q_scale, k_scale, v_scale
        calls["int8_attention"] += 1
        calls["mask_shape"] = None if attn_mask is None else tuple(attn_mask.shape)
        calls["is_causal"] = is_causal
        assert q.dtype == torch.float32
        assert k.dtype == torch.float32
        assert v.dtype == torch.float32
        assert attn_mask is not None
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.zeros(q.shape, dtype=dtype, device=q.device)

    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime_attention)
    monkeypatch.setattr(eager_mod, "runtime_int8_attention", fake_int8_attention)

    attn.w_q = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_heads * attn.head_dim, bias=True).from_float(
        attn.w_q,
        activation_quant="dynamic_int8",
    )
    attn.w_k = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_k,
        activation_quant="dynamic_int8",
    )
    attn.w_v = quant_mod.QuantizedLinearInt8(attn.d_model, attn.n_kv_heads * attn.head_dim, bias=True).from_float(
        attn.w_v,
        activation_quant="dynamic_int8",
    )
    attn.w_o = quant_mod.QuantizedLinearInt8(attn.n_heads * attn.head_dim, attn.d_model, bias=True).from_float(
        attn.w_o,
        activation_quant="dynamic_int8",
    )

    x = torch.randn(2, 3, attn.d_model)
    mask = torch.zeros(2, 1, 3, 3, dtype=torch.float32)
    mask[..., -1] = float("-inf")
    with torch.no_grad():
        y = attn(x, None, None, mask)

    assert y.shape == x.shape
    assert calls["int8_attention"] == 1
    assert calls["mask_shape"] == (2, attn.n_heads, 3, 3)
    assert calls["is_causal"] is False


def test_qkv_packed_spec_projection_rejects_unimplemented_format():
    x = torch.randn(2, 3, 16)
    try:
        runtime_ops_mod.qkv_packed_spec_heads_projection(
            x,
            {"format": "unknown_qkv"},
            q_heads=4,
            kv_heads=4,
        )
    except NotImplementedError as exc:
        assert "unknown_qkv" in str(exc)
    else:
        raise AssertionError("expected unsupported packed_qkv format to raise NotImplementedError")


def test_eager_attention_rejects_unimplemented_packed_qkv_format(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()
    monkeypatch.setattr(attn, "_packed_backend", lambda x: "bitnet")

    monkeypatch.setattr(eager_mod, "runtime_attention", lambda q, k, v, attn_mask=None, is_causal=False, scale=None: torch.zeros_like(q))

    monkeypatch.setattr(
        eager_mod,
        "runtime_resolve_packed_qkv_module_spec",
        lambda *args, **kwargs: {
            "format": "bitnet_qkv",
            "backend": "bitnet",
            "q_size": 4,
            "k_size": 4,
            "v_size": 4,
        },
    )

    try:
        attn(torch.randn(2, 3, 16), None, None, None)
    except NotImplementedError as exc:
        assert "bitnet_qkv" in str(exc)
    else:
        raise AssertionError("expected unsupported packed_qkv format to raise NotImplementedError")
