from __future__ import annotations

import pytest
import torch

import runtime.ops as runtime_ops_mod
import runtime.quant as runtime_quant_mod
from runtime.ops import pack_bitnet_weight


def test_int8_matmul_qkv_uses_runtime_attention(monkeypatch):
    seen = {}

    def fake_runtime_attention(q, k, v, attn_mask=None, *, is_causal=False, scale=None):
        seen["q_dtype"] = q.dtype
        seen["k_dtype"] = k.dtype
        seen["v_dtype"] = v.dtype
        seen["attn_mask"] = attn_mask
        seen["is_causal"] = is_causal
        seen["scale"] = scale
        return torch.ones_like(v)

    monkeypatch.setattr(runtime_quant_mod, "runtime_attention", fake_runtime_attention)

    q = torch.ones(2, 3, 4, 5, dtype=torch.int8)
    k = torch.ones(2, 3, 4, 5, dtype=torch.int8)
    v = torch.ones(2, 3, 4, 5, dtype=torch.int8)
    scales = torch.full((2, 3, 4, 1), 0.25, dtype=torch.float32)

    out = runtime_quant_mod.int8_matmul_qkv(q, k, v, scales, scales, scales)

    assert out.shape == (2, 3, 4, 5)
    assert seen["q_dtype"] == torch.float32
    assert seen["k_dtype"] == torch.float32
    assert seen["v_dtype"] == torch.float32
    assert seen["attn_mask"] is None
    assert seen["is_causal"] is False
    assert seen["scale"] is None


def test_int8_matmul_qkv_prefers_library_backend_on_hopper(monkeypatch):
    seen = {}

    def fake_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, backend=None, is_causal=None, scale=None):
        seen["backend"] = backend
        seen["dropout_p"] = dropout_p
        seen["is_causal"] = is_causal
        seen["scale"] = scale
        seen["dtype"] = q.dtype
        seen["attn_mask"] = attn_mask
        del k, v
        return torch.ones_like(q)

    monkeypatch.setattr(runtime_quant_mod, "prefer_hopper_library_attention", lambda **kwargs: True)
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: False)
    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: None)
    monkeypatch.setattr(runtime_quant_mod, "select_attention_backend", lambda **kwargs: "torch")
    monkeypatch.setattr(runtime_quant_mod, "runtime_scaled_dot_product_attention", fake_sdpa)

    q = torch.ones(2, 3, 4, 5, dtype=torch.int8)
    k = torch.ones(2, 3, 4, 5, dtype=torch.int8)
    v = torch.ones(2, 3, 4, 5, dtype=torch.int8)
    scales = torch.full((2, 3, 4, 1), 0.25, dtype=torch.float32)

    out = runtime_quant_mod.int8_matmul_qkv(q, k, v, scales, scales, scales, is_causal=True, scale=0.125)

    assert out.shape == (2, 3, 4, 5)
    assert seen["backend"] == "torch"
    assert seen["dropout_p"] == 0.0
    assert seen["is_causal"] is True
    assert seen["scale"] == 0.125
    assert seen["dtype"] == torch.float32
    assert seen["attn_mask"] is None


def test_int8_matmul_qkv_accepts_flattened_row_scales(monkeypatch):
    seen = {}

    def fake_runtime_attention(q, k, v, attn_mask=None, *, is_causal=False, scale=None):
        seen["q"] = q.clone()
        seen["k"] = k.clone()
        seen["v"] = v.clone()
        seen["attn_mask"] = attn_mask
        seen["is_causal"] = is_causal
        seen["scale"] = scale
        return torch.zeros_like(v)

    monkeypatch.setattr(runtime_quant_mod, "runtime_attention", fake_runtime_attention)
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: False)
    monkeypatch.setattr(runtime_quant_mod, "prefer_hopper_library_attention", lambda **kwargs: False)

    q = torch.arange(2 * 3 * 4 * 5, dtype=torch.int8).reshape(2, 3, 4, 5)
    k = (q + 1).to(dtype=torch.int8)
    v = (q + 2).to(dtype=torch.int8)
    q_scales = torch.linspace(0.01, 0.24, steps=24, dtype=torch.float32)
    k_scales = torch.linspace(0.02, 0.48, steps=24, dtype=torch.float32)
    v_scales = torch.linspace(0.03, 0.72, steps=24, dtype=torch.float32)

    out = runtime_quant_mod.int8_matmul_qkv(q, k, v, q_scales, k_scales, v_scales, is_causal=True, scale=0.5)

    assert out.shape == (2, 3, 4, 5)
    assert seen["attn_mask"] is None
    assert seen["is_causal"] is True
    assert seen["scale"] == 0.5
    assert torch.allclose(
        seen["q"],
        q.to(dtype=torch.float32) * q_scales.view(2, 3, 4, 1),
    )
    assert torch.allclose(
        seen["k"],
        k.to(dtype=torch.float32) * k_scales.view(2, 3, 4, 1),
    )
    assert torch.allclose(
        seen["v"],
        v.to(dtype=torch.float32) * v_scales.view(2, 3, 4, 1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native int8 attention dispatch")
def test_int8_matmul_qkv_uses_native_module_when_available(monkeypatch):
    seen = {}

    class FakeModule:
        def int8_attention_forward(
            self,
            q,
            q_scale,
            k,
            k_scale,
            v,
            v_scale,
            attn_mask,
            is_causal,
            scale,
            out_dtype,
        ):
            seen["q_dtype"] = q.dtype
            seen["q_scale_shape"] = tuple(q_scale.shape)
            seen["k_scale_shape"] = tuple(k_scale.shape)
            seen["v_scale_shape"] = tuple(v_scale.shape)
            seen["attn_mask"] = attn_mask
            seen["is_causal"] = is_causal
            seen["scale"] = scale
            seen["out_dtype"] = out_dtype
            return torch.full((*q.shape[:-1], v.shape[-1]), 3.0, dtype=torch.float16, device=q.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "int8_attention")
    monkeypatch.setattr(runtime_quant_mod, "prefer_hopper_library_attention", lambda **kwargs: True)

    q = torch.randint(-4, 4, (2, 3, 4, 5), dtype=torch.int8, device="cuda")
    k = torch.randint(-4, 4, (2, 3, 4, 5), dtype=torch.int8, device="cuda")
    v = torch.randint(-4, 4, (2, 3, 4, 5), dtype=torch.int8, device="cuda")
    q_scales = torch.full((2, 3, 4, 1), 0.25, dtype=torch.float32, device="cuda")
    k_scales = torch.full((2, 3, 4, 1), 0.5, dtype=torch.float32, device="cuda")
    v_scales = torch.full((2, 3, 4, 1), 0.75, dtype=torch.float32, device="cuda")

    out = runtime_quant_mod.int8_matmul_qkv(
        q,
        k,
        v,
        q_scales,
        k_scales,
        v_scales,
        is_causal=True,
        scale=0.125,
        out_dtype=torch.float16,
    )

    assert out.shape == (2, 3, 4, 5)
    assert torch.all(out == 3.0)
    assert seen["q_dtype"] == torch.int8
    assert seen["q_scale_shape"] == (24,)
    assert seen["k_scale_shape"] == (24,)
    assert seen["v_scale_shape"] == (24,)
    assert seen["attn_mask"] is None
    assert seen["is_causal"] is True
    assert seen["scale"] == 0.125
    assert seen["out_dtype"] == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native int8 attention dispatch")
def test_int8_matmul_qkv_uses_native_module_with_mask_when_supported(monkeypatch):
    seen = {}

    class FakeModule:
        def int8_attention_forward(
            self,
            q,
            q_scale,
            k,
            k_scale,
            v,
            v_scale,
            attn_mask,
            is_causal,
            scale,
            out_dtype,
        ):
            del q_scale, k_scale, v_scale, is_causal, scale, out_dtype
            seen["q_dtype"] = q.dtype
            seen["attn_mask_dtype"] = attn_mask.dtype
            seen["attn_mask_shape"] = tuple(attn_mask.shape)
            return torch.full((*q.shape[:-1], v.shape[-1]), 7.0, dtype=torch.float16, device=q.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "int8_attention")

    q = torch.randint(-4, 4, (2, 3, 4, 5), dtype=torch.int8, device="cuda")
    k = torch.randint(-4, 4, (2, 3, 4, 5), dtype=torch.int8, device="cuda")
    v = torch.randint(-4, 4, (2, 3, 4, 5), dtype=torch.int8, device="cuda")
    q_scales = torch.full((2, 3, 4, 1), 0.25, dtype=torch.float32, device="cuda")
    k_scales = torch.full((2, 3, 4, 1), 0.5, dtype=torch.float32, device="cuda")
    v_scales = torch.full((2, 3, 4, 1), 0.75, dtype=torch.float32, device="cuda")
    attn_mask = torch.zeros((2, 3, 4, 4), dtype=torch.float16, device="cuda")
    attn_mask[..., -1] = float("-inf")

    out = runtime_quant_mod.int8_matmul_qkv(
        q,
        k,
        v,
        q_scales,
        k_scales,
        v_scales,
        attn_mask=attn_mask,
        out_dtype=torch.float16,
    )

    assert out.shape == (2, 3, 4, 5)
    assert torch.all(out == 7.0)
    assert seen["q_dtype"] == torch.int8
    assert seen["attn_mask_dtype"] == torch.float32
    assert seen["attn_mask_shape"] == (2, 3, 4, 4)


def test_int8_linear_fallback_uses_runtime_linear(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x"] = x.clone()
        seen["weight"] = weight.clone()
        seen["bias_dtype"] = None if bias is None else bias.dtype
        return torch.full((*x.shape[:-1], weight.shape[0]), 5.0, dtype=x.dtype)

    monkeypatch.setattr(runtime_quant_mod, "runtime_linear", fake_runtime_linear)
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: False)

    x = torch.tensor([[0.5, -0.75, 0.25, 1.0], [1.25, -0.5, 0.75, -1.0]], dtype=torch.float32)
    qweight = torch.tensor([[1, -2, 3, -4], [-4, 3, -2, 1]], dtype=torch.int8)
    inv_scale = torch.tensor([0.25, 0.5], dtype=torch.float32)
    bias = torch.randn(2, dtype=torch.float32)

    out = runtime_quant_mod.int8_linear(
        x,
        qweight,
        inv_scale,
        bias=bias,
        act_method="absmax",
    )

    row_scale = x.abs().amax(dim=-1) / 127.0
    qx = torch.round(x / row_scale.unsqueeze(-1)).clamp_(-127.0, 127.0).to(torch.int8)
    expected_x = qx.to(torch.float32) * row_scale.unsqueeze(-1)
    expected_w = qweight.to(torch.float32) * inv_scale.unsqueeze(-1)

    assert out.shape == (2, 2)
    assert torch.all(out == 5.0)
    assert torch.allclose(seen["x"], expected_x)
    assert torch.allclose(seen["weight"], expected_w)
    assert seen["bias_dtype"] == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA int8 linear fallback test")
def test_int8_linear_falls_back_to_runtime_linear_when_native_unavailable(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x_dtype"] = x.dtype
        seen["weight_dtype"] = weight.dtype
        seen["bias_dtype"] = None if bias is None else bias.dtype
        return torch.full((*x.shape[:-1], weight.shape[0]), 12.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: None)
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: False)
    monkeypatch.setattr(runtime_quant_mod, "runtime_linear", fake_runtime_linear)

    qx = torch.randint(-8, 8, (3, 4), dtype=torch.int8, device="cuda")
    x_scale = torch.full((3,), 0.125, dtype=torch.float32, device="cuda")
    qweight = torch.randint(-8, 8, (5, 4), dtype=torch.int8, device="cuda")
    inv_scale = torch.full((5,), 0.25, dtype=torch.float32, device="cuda")
    bias = torch.randn(5, dtype=torch.float16, device="cuda")

    out = runtime_quant_mod.int8_linear_from_quantized_activation(
        qx,
        x_scale,
        qweight,
        inv_scale,
        bias=bias,
        out_dtype=torch.float16,
    )

    assert out.shape == (3, 5)
    assert torch.all(out == 12.0)
    assert seen["x_dtype"] == torch.float16
    assert seen["weight_dtype"] == torch.float16
    assert seen["bias_dtype"] == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native int8 linear dispatch")
def test_int8_linear_uses_native_module_when_available(monkeypatch):
    seen = {}

    class FakeModule:
        def int8_linear_from_float_forward(self, x, qweight, inv_scale, bias, provided_scale, out_dtype):
            seen["x_dtype"] = x.dtype
            seen["qweight_dtype"] = qweight.dtype
            seen["inv_scale_dtype"] = inv_scale.dtype
            seen["bias_dtype"] = None if bias is None else bias.dtype
            seen["provided_scale_shape"] = None if provided_scale is None else tuple(provided_scale.shape)
            seen["out_dtype"] = out_dtype
            return torch.full((*x.shape[:-1], qweight.shape[0]), 6.0, dtype=torch.float16, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "int8_linear_from_float")

    x = torch.randn(3, 4, dtype=torch.float16, device="cuda")
    qweight = torch.randint(-8, 8, (5, 4), dtype=torch.int8, device="cuda")
    inv_scale = torch.full((5,), 0.25, dtype=torch.float32, device="cuda")

    out = runtime_quant_mod.int8_linear(x, qweight, inv_scale, act_scale=0.125)

    assert out.shape == (3, 5)
    assert torch.all(out == 6.0)
    assert seen["x_dtype"] == torch.float16
    assert seen["qweight_dtype"] == torch.int8
    assert seen["inv_scale_dtype"] == torch.float32
    assert seen["bias_dtype"] is None
    assert seen["provided_scale_shape"] == (1,)
    assert seen["out_dtype"] == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native int8 linear numeric parity")
def test_int8_linear_from_quantized_activation_matches_dequantized_reference_for_single_decode_row():
    torch.manual_seed(0)
    qx = torch.randint(-31, 32, (1, 1, 512), dtype=torch.int8, device="cuda")
    x_scale = torch.rand(1, dtype=torch.float32, device="cuda").mul_(0.02).add_(0.001)
    qweight = torch.randint(-31, 32, (512, 512), dtype=torch.int8, device="cuda")
    inv_scale = torch.rand(512, dtype=torch.float32, device="cuda").mul_(0.02).add_(0.001)
    bias = torch.randn(512, dtype=torch.float16, device="cuda")

    out = runtime_quant_mod.int8_linear_from_quantized_activation(
        qx,
        x_scale,
        qweight,
        inv_scale,
        bias=bias,
        out_dtype=torch.float16,
    )

    x_ref = qx.to(dtype=torch.float32) * x_scale.view(1, 1, 1)
    w_ref = qweight.to(dtype=torch.float32) * inv_scale.view(-1, 1)
    ref = torch.nn.functional.linear(x_ref.to(dtype=torch.float16), w_ref.to(dtype=torch.float16), bias)

    assert out.shape == (1, 1, 512)
    assert torch.allclose(out, ref, atol=2e-1, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native int8 attention frontend dispatch")
def test_int8_attention_uses_native_float_frontend_when_available(monkeypatch):
    seen = {}

    class FakeModule:
        def int8_attention_from_float_forward(
            self,
            q,
            k,
            v,
            attn_mask,
            is_causal,
            scale,
            out_dtype,
            q_provided_scale,
            k_provided_scale,
            v_provided_scale,
        ):
            seen["q_dtype"] = q.dtype
            seen["k_dtype"] = k.dtype
            seen["v_dtype"] = v.dtype
            seen["mask_dtype"] = None if attn_mask is None else attn_mask.dtype
            seen["is_causal"] = is_causal
            seen["scale"] = scale
            seen["out_dtype"] = out_dtype
            seen["q_scale_shape"] = None if q_provided_scale is None else tuple(q_provided_scale.shape)
            seen["k_scale_shape"] = None if k_provided_scale is None else tuple(k_provided_scale.shape)
            seen["v_scale_shape"] = None if v_provided_scale is None else tuple(v_provided_scale.shape)
            return torch.full(q.shape, 4.0, dtype=out_dtype, device=q.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "int8_attention_from_float")

    q = torch.randn(2, 3, 4, 5, dtype=torch.float16, device="cuda")
    k = torch.randn(2, 3, 4, 5, dtype=torch.float16, device="cuda")
    v = torch.randn(2, 3, 4, 5, dtype=torch.float16, device="cuda")
    attn_mask = torch.zeros((2, 3, 4, 4), dtype=torch.float32, device="cuda")

    out = runtime_quant_mod.int8_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=False,
        scale=0.125,
        out_dtype=torch.float16,
        q_scale=0.25,
    )

    assert out.shape == q.shape
    assert torch.all(out == 4.0)
    assert seen["q_dtype"] == torch.float16
    assert seen["k_dtype"] == torch.float16
    assert seen["v_dtype"] == torch.float16
    assert seen["mask_dtype"] == torch.float32
    assert seen["is_causal"] is False
    assert seen["scale"] == 0.125
    assert seen["out_dtype"] == torch.float16
    assert seen["q_scale_shape"] == (1,)
    assert seen["k_scale_shape"] is None
    assert seen["v_scale_shape"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet linear frontend dispatch")
def test_bitnet_linear_from_float_uses_native_module_when_available(monkeypatch):
    seen = {}

    class FakeModule:
        def bitnet_linear_from_float_forward(
            self,
            x,
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
            bias,
            spin_enabled,
            spin_signs,
            pre_scale,
            act_quant_mode,
            act_quant_method,
            act_quant_bits,
            act_quant_percentile,
            act_scale,
            out_dtype,
        ):
            seen["x_dtype"] = x.dtype
            seen["packed_weight_dtype"] = packed_weight.dtype
            seen["scale_values_dtype"] = scale_values.dtype
            seen["layout_header_dtype"] = layout_header.dtype
            seen["segment_offsets_dtype"] = segment_offsets.dtype
            seen["bias_dtype"] = None if bias is None else bias.dtype
            seen["spin_enabled"] = bool(spin_enabled)
            seen["spin_signs_shape"] = None if spin_signs is None else tuple(spin_signs.shape)
            seen["pre_scale_shape"] = None if pre_scale is None else tuple(pre_scale.shape)
            seen["act_quant_mode"] = act_quant_mode
            seen["act_quant_method"] = act_quant_method
            seen["act_quant_bits"] = int(act_quant_bits)
            seen["act_quant_percentile"] = float(act_quant_percentile)
            seen["act_scale_shape"] = None if act_scale is None else tuple(act_scale.shape)
            seen["out_dtype"] = out_dtype
            dtype = x.dtype if out_dtype is None else out_dtype
            return torch.full((*x.shape[:-1], int(layout_header[3].item())), 13.0, dtype=dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "bitnet_linear_from_float")

    x = torch.randn(3, 4, dtype=torch.float16, device="cuda")
    packed_weight = torch.tensor([[0x64], [0x4A]], dtype=torch.uint8, device="cuda")
    scale_values = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    layout_header = torch.tensor([1, 16, 32, 2, 4, 2, 4, 0, 2, 1, 80, 1, 0], dtype=torch.int32, device="cuda")
    segment_offsets = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    bias = torch.randn(2, dtype=torch.float16, device="cuda")

    out = runtime_quant_mod.bitnet_linear_from_float(
        x,
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias=bias,
        spin_enabled=True,
        spin_signs=torch.ones(4, dtype=torch.float32, device="cuda"),
        pre_scale=torch.full((4,), 0.5, dtype=torch.float32, device="cuda"),
        act_quant_mode="static_int8",
        act_scale=torch.tensor(0.25, dtype=torch.float32, device="cuda"),
        act_quant_bits=6,
        act_quant_method="percentile",
        act_quant_percentile=0.95,
    )

    assert out.shape == (3, 2)
    assert torch.all(out == 13.0)
    assert seen["x_dtype"] == torch.float16
    assert seen["packed_weight_dtype"] == torch.uint8
    assert seen["scale_values_dtype"] == torch.float32
    assert seen["layout_header_dtype"] == torch.int32
    assert seen["segment_offsets_dtype"] == torch.int32
    assert seen["bias_dtype"] == torch.float16
    assert seen["spin_enabled"] is True
    assert seen["spin_signs_shape"] == (4,)
    assert seen["pre_scale_shape"] == (4,)
    assert seen["act_quant_mode"] == "static_int8"
    assert seen["act_quant_method"] == "percentile"
    assert seen["act_quant_bits"] == 6
    assert seen["act_quant_percentile"] == 0.95
    assert seen["act_scale_shape"] == (1,)
    assert seen["out_dtype"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet int8 linear frontend dispatch")
def test_bitnet_int8_linear_from_float_uses_native_module_when_available(monkeypatch):
    seen = {}

    class FakeModule:
        def bitnet_int8_linear_from_float_forward(
            self,
            x,
            qweight,
            inv_scale,
            bias,
            pre_scale,
            act_quant_mode,
            act_quant_method,
            act_quant_bits,
            act_quant_percentile,
            act_scale,
            out_dtype,
        ):
            seen["x_dtype"] = x.dtype
            seen["qweight_dtype"] = qweight.dtype
            seen["inv_scale_dtype"] = inv_scale.dtype
            seen["bias_dtype"] = None if bias is None else bias.dtype
            seen["pre_scale_shape"] = None if pre_scale is None else tuple(pre_scale.shape)
            seen["act_quant_mode"] = act_quant_mode
            seen["act_quant_method"] = act_quant_method
            seen["act_quant_bits"] = int(act_quant_bits)
            seen["act_quant_percentile"] = float(act_quant_percentile)
            seen["act_scale_shape"] = None if act_scale is None else tuple(act_scale.shape)
            seen["out_dtype"] = out_dtype
            return torch.full((*x.shape[:-1], qweight.shape[0]), 9.0, dtype=out_dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "bitnet_int8_linear_from_float")

    x = torch.randn(2, 3, 4, dtype=torch.float16, device="cuda")
    qweight = torch.randint(-8, 8, (5, 4), dtype=torch.int8, device="cuda")
    inv_scale = torch.full((5,), 0.25, dtype=torch.float32, device="cuda")
    bias = torch.randn(5, dtype=torch.float16, device="cuda")

    out = runtime_quant_mod.bitnet_int8_linear_from_float(
        x,
        qweight,
        inv_scale,
        bias=bias,
        pre_scale=torch.full((4,), 0.5, dtype=torch.float32, device="cuda"),
        act_quant_mode="static_int8",
        act_scale=torch.tensor(0.125, dtype=torch.float32, device="cuda"),
        act_quant_bits=6,
        act_quant_method="percentile",
        act_quant_percentile=0.95,
    )

    assert out.shape == (2, 3, 5)
    assert torch.all(out == 9.0)
    assert seen["x_dtype"] == torch.float16
    assert seen["qweight_dtype"] == torch.int8
    assert seen["inv_scale_dtype"] == torch.float32
    assert seen["bias_dtype"] == torch.float16
    assert seen["pre_scale_shape"] == (4,)
    assert seen["act_quant_mode"] == "static_int8"
    assert seen["act_quant_method"] == "percentile"
    assert seen["act_quant_bits"] == 6
    assert seen["act_quant_percentile"] == 0.95
    assert seen["act_scale_shape"] == (1,)
    assert seen["out_dtype"] == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet int8 linear frontend parity")
def test_bitnet_int8_linear_from_float_dynamic_matches_split_quantized_path_for_single_decode_row():
    if not runtime_quant_mod.has_native_op("bitnet_int8_linear_from_float"):
        pytest.skip("native BitNet int8 linear frontend not available")

    torch.manual_seed(9_517)
    x = torch.randn(1, 1, 512, dtype=torch.float16, device="cuda")
    qweight = torch.randint(-31, 32, (512, 512), dtype=torch.int8, device="cuda")
    inv_scale = torch.rand(512, dtype=torch.float32, device="cuda").mul_(0.02).add_(0.001)
    bias = torch.randn(512, dtype=torch.float16, device="cuda")
    pre_scale = torch.linspace(0.75, 1.25, 512, dtype=torch.float32, device="cuda")
    qmax = 31.0

    direct = runtime_quant_mod.bitnet_int8_linear_from_float(
        x,
        qweight,
        inv_scale,
        bias=bias,
        pre_scale=pre_scale,
        act_quant_mode="dynamic_int8",
        act_scale=None,
        act_quant_bits=6,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )

    x_local = (x / pre_scale.view(1, 1, -1).to(dtype=x.dtype)).to(dtype=x.dtype)
    row_scale = x_local.float().abs().amax().reshape(1).clamp_min_(1.0e-8).div_(qmax)
    qx = torch.round(x_local.float() / row_scale.view(1, 1, 1)).clamp_(-qmax, qmax).to(dtype=torch.int8)
    split = runtime_quant_mod.int8_linear_from_quantized_activation(
        qx,
        row_scale,
        qweight,
        inv_scale,
        bias,
        out_dtype=torch.float16,
    )

    diff = (direct - split).abs()
    assert float(diff.max().item()) <= 0.03125
    assert float(diff.mean().item()) <= 0.005


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet linear frontend parity")
@pytest.mark.parametrize(
    ("batch", "seq", "in_features", "out_features", "per_row_scale"),
    (
        (1, 1, 256, 192, False),
        (1, 16, 1024, 256, True),
        (1, 9, 4096, 256, True),
    ),
)
def test_bitnet_linear_from_float_matches_split_native_path_without_spin(
    batch: int,
    seq: int,
    in_features: int,
    out_features: int,
    per_row_scale: bool,
):
    if not runtime_quant_mod.has_native_op("bitnet_linear_from_float"):
        pytest.skip("native BitNet linear frontend not available")
    if not runtime_quant_mod.has_native_op("bitnet_linear"):
        pytest.skip("native BitNet linear op not available")
    if not runtime_ops_mod.has_native_op("bitnet_transform_input"):
        pytest.skip("native BitNet input transform op not available")

    torch.manual_seed(batch * 10_000 + seq * 1_000 + in_features + out_features)
    weight = torch.randn(out_features, in_features, dtype=torch.float32, device="cuda")
    packed_weight, scale_values, layout_header, segment_offsets = pack_bitnet_weight(weight)
    x = torch.randn(batch, seq, in_features, dtype=torch.float16, device="cuda")
    bias = torch.randn(out_features, dtype=torch.float16, device="cuda")
    pre_scale = torch.linspace(0.75, 1.25, in_features, dtype=torch.float32, device="cuda")
    rows = batch * seq
    act_scale = (
        torch.linspace(0.0625, 0.1875, rows, dtype=torch.float32, device="cuda")
        if per_row_scale
        else torch.tensor(0.125, dtype=torch.float32, device="cuda")
    )

    direct = runtime_quant_mod.bitnet_linear_from_float(
        x,
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias=bias,
        spin_enabled=False,
        spin_signs=None,
        pre_scale=pre_scale,
        act_quant_mode="static_int8",
        act_scale=act_scale,
        act_quant_bits=6,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )
    split = runtime_quant_mod.bitnet_linear(
        runtime_ops_mod.bitnet_transform_input(
            x,
            spin_enabled=False,
            pre_scale=pre_scale,
            act_quant_mode="static_int8",
            act_scale=act_scale,
            act_quant_bits=6,
            act_quant_method="absmax",
            act_quant_percentile=0.999,
        ),
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias=bias,
    )

    diff = (direct - split).abs()
    assert float(diff.max().item()) <= 0.03125
    assert float(diff.mean().item()) <= 0.005


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet linear frontend parity")
def test_bitnet_linear_from_float_dynamic_matches_split_native_path_without_spin():
    if not runtime_quant_mod.has_native_op("bitnet_linear_from_float"):
        pytest.skip("native BitNet linear frontend not available")
    if not runtime_quant_mod.has_native_op("bitnet_linear"):
        pytest.skip("native BitNet linear op not available")
    if not runtime_ops_mod.has_native_op("bitnet_transform_input"):
        pytest.skip("native BitNet input transform op not available")

    torch.manual_seed(11_024)
    weight = torch.randn(256, 1024, dtype=torch.float32, device="cuda")
    packed_weight, scale_values, layout_header, segment_offsets = pack_bitnet_weight(weight)
    x = torch.randn(2, 9, 1024, dtype=torch.float16, device="cuda")
    bias = torch.randn(256, dtype=torch.float16, device="cuda")
    pre_scale = torch.linspace(0.75, 1.25, 1024, dtype=torch.float32, device="cuda")

    direct = runtime_quant_mod.bitnet_linear_from_float(
        x,
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias=bias,
        spin_enabled=False,
        spin_signs=None,
        pre_scale=pre_scale,
        act_quant_mode="dynamic_int8",
        act_scale=None,
        act_quant_bits=6,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )
    split = runtime_quant_mod.bitnet_linear(
        runtime_ops_mod.bitnet_transform_input(
            x,
            spin_enabled=False,
            pre_scale=pre_scale,
            act_quant_mode="dynamic_int8",
            act_scale=None,
            act_quant_bits=6,
            act_quant_method="absmax",
            act_quant_percentile=0.999,
        ),
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias=bias,
    )

    diff = (direct - split).abs()
    assert float(diff.max().item()) <= 0.03125
    assert float(diff.mean().item()) <= 0.005


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet linear frontend parity")
def test_bitnet_linear_from_float_dynamic_matches_split_native_path_for_single_decode_row():
    if not runtime_quant_mod.has_native_op("bitnet_linear_from_float"):
        pytest.skip("native BitNet linear frontend not available")
    if not runtime_quant_mod.has_native_op("bitnet_linear"):
        pytest.skip("native BitNet linear op not available")
    if not runtime_ops_mod.has_native_op("bitnet_transform_input"):
        pytest.skip("native BitNet input transform op not available")

    torch.manual_seed(1_257)
    weight = torch.randn(192, 256, dtype=torch.float32, device="cuda")
    packed_weight, scale_values, layout_header, segment_offsets = pack_bitnet_weight(weight)
    x = torch.randn(1, 1, 256, dtype=torch.float16, device="cuda")
    bias = torch.randn(192, dtype=torch.float16, device="cuda")
    pre_scale = torch.linspace(0.75, 1.25, 256, dtype=torch.float32, device="cuda")

    direct = runtime_quant_mod.bitnet_linear_from_float(
        x,
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias=bias,
        spin_enabled=False,
        spin_signs=None,
        pre_scale=pre_scale,
        act_quant_mode="dynamic_int8",
        act_scale=None,
        act_quant_bits=6,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )
    split = runtime_quant_mod.bitnet_linear(
        runtime_ops_mod.bitnet_transform_input(
            x,
            spin_enabled=False,
            pre_scale=pre_scale,
            act_quant_mode="dynamic_int8",
            act_scale=None,
            act_quant_bits=6,
            act_quant_method="absmax",
            act_quant_percentile=0.999,
        ),
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias=bias,
    )

    diff = (direct - split).abs()
    assert float(diff.max().item()) <= 0.03125
    assert float(diff.mean().item()) <= 0.005


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet linear parity")
def test_bitnet_linear_matches_dequantized_reference_across_native_plans():
    if not runtime_quant_mod.has_native_op("bitnet_linear"):
        pytest.skip("native BitNet linear not available")
    module = runtime_quant_mod.native_module()
    if module is None or not hasattr(module, "bitnet_linear_forward"):
        pytest.skip("native BitNet linear module entry point not available")

    for batch, seq, in_features, out_features in (
        (1, 1, 256, 192),
        (1, 16, 1024, 256),
        (1, 9, 4096, 256),
    ):
        torch.manual_seed(batch * 10_000 + seq * 1_000 + in_features + out_features)
        x = torch.randn(batch, seq, in_features, device="cuda", dtype=torch.float32)
        weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32) * 0.125
        bias = torch.randn(out_features, device="cuda", dtype=torch.float32) * 0.01
        packed_weight, scale_values, layout_header, segment_offsets = pack_bitnet_weight(weight)
        ref_weight = runtime_quant_mod._dequantize_packed_bitnet_weight(
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
            dtype=torch.float32,
        )
        expected = torch.nn.functional.linear(x, ref_weight, bias)
        actual = module.bitnet_linear_forward(
            x,
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
            bias,
            torch.float32,
        )
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_fp8_linear_uses_runtime_linear(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x_dtype"] = x.dtype
        seen["weight_dtype"] = weight.dtype
        seen["bias_dtype"] = None if bias is None else bias.dtype
        return torch.full((*x.shape[:-1], weight.shape[0]), 7.0, dtype=x.dtype)

    class FakeTracker:
        def __init__(self):
            self.values = []

        def update(self, x):
            self.values.append(tuple(x.shape))

    monkeypatch.setattr(runtime_quant_mod, "runtime_linear", fake_runtime_linear)

    tracker = FakeTracker()
    x = torch.randn(3, 4, dtype=torch.float16)
    weight_fp8 = torch.randn(5, 4, dtype=torch.float32)
    bias = torch.randn(5, dtype=torch.float32)

    out = runtime_quant_mod.fp8_linear(x, weight_fp8, tracker, 0.5, bias=bias)

    assert out.shape == (3, 5)
    assert torch.all(out == 7.0)
    assert tracker.values == [(5, 4)]
    assert seen["x_dtype"] == torch.float16
    assert seen["weight_dtype"] == torch.float16
    assert seen["bias_dtype"] == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native fp8 linear dispatch")
def test_fp8_linear_uses_native_module_when_available(monkeypatch):
    seen = {}

    class FakeTracker:
        def update(self, x):
            seen["tracker_shape"] = tuple(x.shape)

    class FakeModule:
        def fp8_linear_forward(self, x, weight_fp8, weight_scale, bias, out_dtype):
            seen["x_dtype"] = x.dtype
            seen["weight_dtype"] = weight_fp8.dtype
            seen["weight_scale"] = weight_scale
            seen["bias_dtype"] = None if bias is None else bias.dtype
            seen["out_dtype"] = out_dtype
            return torch.full((*x.shape[:-1], weight_fp8.shape[0]), 8.0, dtype=out_dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "fp8_linear")

    tracker = FakeTracker()
    x = torch.randn(3, 4, dtype=torch.float16, device="cuda")
    weight_fp8 = torch.randn(5, 4, dtype=torch.float16, device="cuda")
    bias = torch.randn(5, dtype=torch.float16, device="cuda")

    out = runtime_quant_mod.fp8_linear(x, weight_fp8, tracker, 0.5, bias=bias)

    assert out.shape == (3, 5)
    assert torch.all(out == 8.0)
    assert seen["tracker_shape"] == (5, 4)
    assert seen["x_dtype"] == torch.float16
    assert seen["weight_dtype"] == torch.float16
    assert seen["weight_scale"] == 0.5
    assert seen["bias_dtype"] == torch.float16
    assert seen["out_dtype"] == torch.float16


def test_nf4_quantize_and_dequantize_round_trip_codebook_values():
    qcodes = torch.arange(16, dtype=torch.uint8).view(4, 4)

    decoded = runtime_quant_mod.nf4_dequantize(qcodes, torch.tensor(1.0, dtype=torch.float32))
    requantized, meta = runtime_quant_mod.nf4_quantize(decoded, torch.tensor(1.0, dtype=torch.float32))

    assert decoded.shape == (4, 4)
    assert decoded.dtype == torch.float32
    assert torch.equal(requantized, qcodes)
    assert meta.dtype == torch.float32


def test_nf4_linear_fallback_uses_runtime_linear(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x_dtype"] = x.dtype
        seen["weight"] = weight.clone()
        seen["bias_dtype"] = None if bias is None else bias.dtype
        return torch.full((*x.shape[:-1], weight.shape[0]), 11.0, dtype=x.dtype)

    monkeypatch.setattr(runtime_quant_mod, "runtime_linear", fake_runtime_linear)

    x = torch.ones(2, 4, dtype=torch.float32)
    qcodes = torch.tensor([[0, 1, 14, 15], [7, 8, 9, 10]], dtype=torch.uint8)
    weight_packed = torch.tensor([[0x10, 0xFE], [0x87, 0xA9]], dtype=torch.uint8)
    weight_scale = torch.tensor([2.0, 0.5], dtype=torch.float32)
    bias = torch.randn(2, dtype=torch.float32)

    out = runtime_quant_mod.nf4_linear(x, weight_packed, weight_scale, bias=bias)

    expected_weight = runtime_quant_mod.nf4_dequantize(qcodes, weight_scale.unsqueeze(-1))

    assert out.shape == (2, 2)
    assert torch.all(out == 11.0)
    assert seen["x_dtype"] == torch.float32
    assert seen["bias_dtype"] == torch.float32
    assert torch.allclose(seen["weight"], expected_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native nf4 linear dispatch")
def test_nf4_linear_uses_native_module_when_available(monkeypatch):
    seen = {}

    class FakeModule:
        def nf4_linear_forward(self, x, packed_weight, weight_scale, bias):
            seen["x_dtype"] = x.dtype
            seen["packed_weight_dtype"] = packed_weight.dtype
            seen["weight_scale_dtype"] = weight_scale.dtype
            seen["bias_dtype"] = None if bias is None else bias.dtype
            return torch.full((*x.shape[:-1], packed_weight.shape[0]), 10.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "nf4_linear")

    x = torch.randn(3, 4, dtype=torch.float16, device="cuda")
    weight_packed = torch.randint(0, 255, (5, 2), dtype=torch.uint8, device="cuda")
    weight_scale = torch.full((5,), 0.5, dtype=torch.float32, device="cuda")

    out = runtime_quant_mod.nf4_linear(x, weight_packed, weight_scale)

    assert out.shape == (3, 5)
    assert torch.all(out == 10.0)
    assert seen["x_dtype"] == torch.float16
    assert seen["packed_weight_dtype"] == torch.uint8
    assert seen["weight_scale_dtype"] == torch.float32
    assert seen["bias_dtype"] is None


def test_int4_linear_fallback_uses_runtime_linear(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x_dtype"] = x.dtype
        seen["weight"] = weight.clone()
        seen["bias_dtype"] = None if bias is None else bias.dtype
        return torch.full((*x.shape[:-1], weight.shape[0]), 9.0, dtype=x.dtype)

    monkeypatch.setattr(runtime_quant_mod, "runtime_linear", fake_runtime_linear)
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: False)

    x = torch.ones(2, 4, dtype=torch.float32)
    weight_packed = torch.tensor([[0x98, 0xBA], [0x87, 0x65]], dtype=torch.uint8)
    inv_scale = torch.tensor([0.5, 0.25], dtype=torch.float32)
    bias = torch.randn(2, dtype=torch.float32)

    out = runtime_quant_mod.int4_linear(x, weight_packed, inv_scale, bias=bias)

    assert out.shape == (2, 2)
    assert torch.all(out == 9.0)
    assert seen["x_dtype"] == torch.float32
    assert seen["bias_dtype"] == torch.float32
    assert torch.allclose(
        seen["weight"],
        torch.tensor(
            [
                [0.0, 0.5, 1.0, 1.5],
                [-0.25, 0.0, -0.75, -0.5],
            ],
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA int4 linear fallback test")
def test_int4_linear_falls_back_to_runtime_linear_when_native_unavailable(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x_dtype"] = x.dtype
        seen["weight_dtype"] = weight.dtype
        seen["bias_dtype"] = None if bias is None else bias.dtype
        return torch.full((*x.shape[:-1], weight.shape[0]), 13.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: None)
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: False)
    monkeypatch.setattr(runtime_quant_mod, "runtime_linear", fake_runtime_linear)

    x = torch.randn(3, 4, dtype=torch.float16, device="cuda")
    weight_packed = torch.randint(0, 255, (5, 2), dtype=torch.uint8, device="cuda")
    inv_scale = torch.full((5,), 0.25, dtype=torch.float32, device="cuda")
    bias = torch.randn(5, dtype=torch.float16, device="cuda")

    out = runtime_quant_mod.int4_linear(x, weight_packed, inv_scale, bias=bias)

    assert out.shape == (3, 5)
    assert torch.all(out == 13.0)
    assert seen["x_dtype"] == torch.float16
    assert seen["weight_dtype"] == torch.float16
    assert seen["bias_dtype"] == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native int4 linear dispatch")
def test_int4_linear_uses_native_module_when_available(monkeypatch):
    seen = {}

    class FakeModule:
        def int4_linear_forward(self, x, packed_weight, inv_scale, bias):
            seen["x_dtype"] = x.dtype
            seen["packed_weight_dtype"] = packed_weight.dtype
            seen["inv_scale_dtype"] = inv_scale.dtype
            seen["bias_dtype"] = None if bias is None else bias.dtype
            return torch.full((*x.shape[:-1], inv_scale.shape[0]), 14.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "int4_linear")

    x = torch.randn(3, 4, dtype=torch.float16, device="cuda")
    weight_packed = torch.randint(0, 255, (5, 2), dtype=torch.uint8, device="cuda")
    inv_scale = torch.full((5,), 0.25, dtype=torch.float32, device="cuda")

    out = runtime_quant_mod.int4_linear(x, weight_packed, inv_scale)

    assert out.shape == (3, 5)
    assert torch.all(out == 14.0)
    assert seen["x_dtype"] == torch.float16
    assert seen["packed_weight_dtype"] == torch.uint8
    assert seen["inv_scale_dtype"] == torch.float32
    assert seen["bias_dtype"] is None


def test_bitnet_linear_fallback_uses_runtime_linear(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x_dtype"] = x.dtype
        seen["weight"] = weight.clone()
        seen["bias_dtype"] = None if bias is None else bias.dtype
        return torch.full((*x.shape[:-1], weight.shape[0]), 11.0, dtype=x.dtype)

    monkeypatch.setattr(runtime_quant_mod, "runtime_linear", fake_runtime_linear)
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: False)

    x = torch.ones(2, 4, dtype=torch.float32)
    packed_weight = torch.tensor([[0x64], [0x4A]], dtype=torch.uint8)
    scale_values = torch.tensor([0.5], dtype=torch.float32)
    layout_header = torch.tensor([1, 16, 32, 2, 4, 2, 4, 0, 2, 1, 80, 1, 0], dtype=torch.int32)
    segment_offsets = torch.tensor([0, 2], dtype=torch.int32)
    bias = torch.randn(2, dtype=torch.float32)

    out = runtime_quant_mod.bitnet_linear(
        x,
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias=bias,
    )

    assert out.shape == (2, 2)
    assert torch.all(out == 11.0)
    assert seen["x_dtype"] == torch.float32
    assert seen["bias_dtype"] == torch.float32
    assert torch.allclose(
        seen["weight"],
        torch.tensor(
            [
                [-0.5, 0.0, 0.5, 0.0],
                [0.5, 0.5, -0.5, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_bitnet_linear_rejects_invalid_layout_header():
    x = torch.ones(1, 4, dtype=torch.float32)
    packed_weight = torch.tensor([[0x64]], dtype=torch.uint8)
    scale_values = torch.tensor([0.5], dtype=torch.float32)
    layout_header = torch.tensor([1, 16, 32], dtype=torch.int32)
    segment_offsets = torch.tensor([0, 1], dtype=torch.int32)

    try:
        runtime_quant_mod.bitnet_linear(
            x,
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
        )
    except ValueError as exc:
        assert "layout_header" in str(exc)
    else:
        raise AssertionError("expected invalid BitNet layout_header to raise ValueError")


def test_bitnet_linear_rejects_unsupported_interleave_mode():
    x = torch.ones(1, 4, dtype=torch.float32)
    packed_weight = torch.tensor([[0x64]], dtype=torch.uint8)
    scale_values = torch.tensor([0.5], dtype=torch.float32)
    layout_header = torch.tensor([1, 16, 32, 1, 4, 1, 4, 0, 1, 99, 80, 1, 0], dtype=torch.int32)
    segment_offsets = torch.tensor([0, 1], dtype=torch.int32)

    try:
        runtime_quant_mod.bitnet_linear(
            x,
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
        )
    except ValueError as exc:
        assert "interleave_mode" in str(exc)
    else:
        raise AssertionError("expected unsupported BitNet interleave_mode to raise ValueError")
