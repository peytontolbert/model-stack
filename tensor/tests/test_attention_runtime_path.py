from pathlib import Path
import sys

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
    monkeypatch.setattr(eager_mod, "runtime_attention", lambda q, k, v, attn_mask=None, is_causal=False, scale=None: torch.zeros_like(q))
    orig_has_native_op = runtime_ops_mod.has_native_op
    monkeypatch.setattr(
        runtime_ops_mod,
        "has_native_op",
        lambda name: False if name == "bitnet_qkv_packed_heads_projection" else orig_has_native_op(name),
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
    assert len(calls) == 4
    assert calls[0] == ((2, 3, attn.d_model), attn.n_heads * attn.head_dim)
    assert calls[1] == ((2, 3, attn.d_model), attn.n_kv_heads * attn.head_dim)
    assert calls[2] == ((2, 3, attn.d_model), attn.n_kv_heads * attn.head_dim)
    assert calls[3] == ((2, 3, attn.d_model), attn.d_model)


def test_eager_attention_bitnet_wrappers_use_native_packed_qkv_when_available(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()
    monkeypatch.setattr(attn, "_packed_backend", lambda x: "bitnet")
    monkeypatch.setattr(eager_mod, "runtime_attention", lambda q, k, v, attn_mask=None, is_causal=False, scale=None: torch.zeros_like(q))

    native_calls = {"qkv": 0}
    output_calls = []

    class FakeNativeModule:
        def bitnet_qkv_packed_heads_projection_forward(
            self,
            q_x,
            q_packed_weight,
            q_scale_values,
            q_layout_header,
            q_segment_offsets,
            q_bias,
            k_x,
            k_packed_weight,
            k_scale_values,
            k_layout_header,
            k_segment_offsets,
            k_bias,
            v_x,
            v_packed_weight,
            v_scale_values,
            v_layout_header,
            v_segment_offsets,
            v_bias,
            q_heads,
            kv_heads,
        ):
            del q_packed_weight, q_scale_values, q_segment_offsets, q_bias
            del k_packed_weight, k_scale_values, k_segment_offsets, k_bias
            del v_packed_weight, v_scale_values, v_segment_offsets, v_bias
            native_calls["qkv"] += 1
            q_width = int(q_layout_header[3].item()) // int(q_heads)
            kv_width = int(k_layout_header[3].item()) // int(kv_heads)
            return (
                torch.full((q_x.shape[0], int(q_heads), q_x.shape[1], q_width), 1.0, dtype=q_x.dtype, device=q_x.device),
                torch.full((k_x.shape[0], int(kv_heads), k_x.shape[1], kv_width), 2.0, dtype=k_x.dtype, device=k_x.device),
                torch.full((v_x.shape[0], int(kv_heads), v_x.shape[1], kv_width), 3.0, dtype=v_x.dtype, device=v_x.device),
            )

    def fake_runtime_bitnet_linear(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None):
        del packed_weight, scale_values, segment_offsets, bias
        out_features = int(layout_header[3].item())
        output_calls.append((tuple(x.shape), out_features))
        return torch.full(x.shape[:-1] + (out_features,), 4.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "bitnet_qkv_packed_heads_projection")
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
    assert output_calls == [((2, 3, attn.d_model), attn.d_model)]


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
    monkeypatch.setattr(eager_mod, "runtime_quantize_activation_int8_rowwise", fake_quantize_activation_int8_rowwise)
    monkeypatch.setattr(quant_mod, "runtime_int8_linear_from_quantized_activation", fake_int8_linear_from_quantized_activation)
    monkeypatch.setattr(
        eager_mod,
        "runtime_int8_matmul_qkv",
        lambda q, k, v, *args, **kwargs: torch.zeros_like(v, dtype=torch.float32),
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
    assert len(calls["quantize"]) >= 2
    assert len(calls["project"]) == 4
    assert calls["project"][0][0] == calls["project"][1][0] == calls["project"][2][0]
    assert calls["project"][0][1] == attn.n_heads * attn.head_dim
    assert calls["project"][1][1] == attn.n_kv_heads * attn.head_dim
    assert calls["project"][2][1] == attn.n_kv_heads * attn.head_dim
    assert calls["project"][3][1] == attn.d_model


def test_eager_attention_int8_wrappers_use_int8_attention_core(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()

    calls = {"quantize": 0, "int8_attention": 0}

    def fail_runtime_attention(*args, **kwargs):
        del args, kwargs
        raise AssertionError("float runtime attention path should not run for int8 attention-core path")

    def fake_quantize_activation_int8_rowwise(x, *, scale=None, method="absmax", percentile=0.999, eps=1e-8):
        del scale, method, percentile, eps
        calls["quantize"] += 1
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        return torch.ones_like(x, dtype=torch.int8), torch.full((rows,), 0.25, dtype=torch.float32, device=x.device)

    def fake_int8_matmul_qkv(q, k, v, q_scales, k_scales, v_scales, attn_mask=None, *, is_causal=False, scale=None, out_dtype=None):
        del q_scales, k_scales, v_scales, attn_mask, scale
        calls["int8_attention"] += 1
        assert q.dtype == torch.int8
        assert k.dtype == torch.int8
        assert v.dtype == torch.int8
        assert is_causal is True
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.zeros(q.shape, dtype=dtype, device=q.device)

    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime_attention)
    monkeypatch.setattr(eager_mod, "runtime_quantize_activation_int8_rowwise", fake_quantize_activation_int8_rowwise)
    monkeypatch.setattr(eager_mod, "runtime_int8_matmul_qkv", fake_int8_matmul_qkv)

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
    assert calls["quantize"] == 3
    assert calls["int8_attention"] == 1


def test_eager_attention_int8_wrappers_use_int8_attention_core_with_mask(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()

    calls = {"quantize": 0, "int8_attention": 0, "mask_shape": None, "is_causal": None}

    def fail_runtime_attention(*args, **kwargs):
        del args, kwargs
        raise AssertionError("float runtime attention path should not run for masked int8 attention-core path")

    def fake_quantize_activation_int8_rowwise(x, *, scale=None, method="absmax", percentile=0.999, eps=1e-8):
        del scale, method, percentile, eps
        calls["quantize"] += 1
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        return torch.ones_like(x, dtype=torch.int8), torch.full((rows,), 0.25, dtype=torch.float32, device=x.device)

    def fake_int8_matmul_qkv(q, k, v, q_scales, k_scales, v_scales, attn_mask=None, *, is_causal=False, scale=None, out_dtype=None):
        del q_scales, k_scales, v_scales, scale
        calls["int8_attention"] += 1
        calls["mask_shape"] = None if attn_mask is None else tuple(attn_mask.shape)
        calls["is_causal"] = is_causal
        assert q.dtype == torch.int8
        assert k.dtype == torch.int8
        assert v.dtype == torch.int8
        assert attn_mask is not None
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.zeros(q.shape, dtype=dtype, device=q.device)

    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime_attention)
    monkeypatch.setattr(eager_mod, "runtime_quantize_activation_int8_rowwise", fake_quantize_activation_int8_rowwise)
    monkeypatch.setattr(eager_mod, "runtime_int8_matmul_qkv", fake_int8_matmul_qkv)

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
    assert calls["quantize"] == 3
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
