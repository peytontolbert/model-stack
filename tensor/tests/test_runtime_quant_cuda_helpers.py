from __future__ import annotations

import pytest
import torch

import runtime.quant as runtime_quant_mod


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
        def int8_linear_forward(self, qx, x_scale, qweight, inv_scale, bias, out_dtype):
            seen["qx_dtype"] = qx.dtype
            seen["x_scale_shape"] = tuple(x_scale.shape)
            seen["qweight_dtype"] = qweight.dtype
            seen["inv_scale_dtype"] = inv_scale.dtype
            seen["bias_dtype"] = None if bias is None else bias.dtype
            seen["out_dtype"] = out_dtype
            return torch.full((*qx.shape[:-1], qweight.shape[0]), 6.0, dtype=torch.float16, device=qx.device)

    monkeypatch.setattr(runtime_quant_mod, "native_module", lambda: FakeModule())
    monkeypatch.setattr(runtime_quant_mod, "has_native_op", lambda name: name == "int8_linear")

    x = torch.randn(3, 4, dtype=torch.float16, device="cuda")
    qweight = torch.randint(-8, 8, (5, 4), dtype=torch.int8, device="cuda")
    inv_scale = torch.full((5,), 0.25, dtype=torch.float32, device="cuda")

    out = runtime_quant_mod.int8_linear(x, qweight, inv_scale, act_scale=0.125)

    assert out.shape == (3, 5)
    assert torch.all(out == 6.0)
    assert seen["qx_dtype"] == torch.int8
    assert seen["x_scale_shape"] == (3,)
    assert seen["qweight_dtype"] == torch.int8
    assert seen["inv_scale_dtype"] == torch.float32
    assert seen["bias_dtype"] is None
    assert seen["out_dtype"] == torch.float16


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
