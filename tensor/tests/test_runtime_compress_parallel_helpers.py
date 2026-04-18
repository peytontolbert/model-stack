from __future__ import annotations

from pathlib import Path

import torch

import compress.export as delta_mod
import compress.lora as lora_mod
import compress.quantization as quant_mod
import interpret.features.sae_ops as sae_ops_mod
import runtime.ops as runtime_ops_mod
import runtime.quant as runtime_quant_mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_lora_linear_uses_runtime_linear_for_base_and_adapter_paths(monkeypatch):
    calls = []

    def fake_runtime_linear(x, weight, bias):
        calls.append((tuple(x.shape), tuple(weight.shape), bias is not None))
        return torch.ones(*x.shape[:-1], weight.shape[0], dtype=x.dtype)

    monkeypatch.setattr(lora_mod, "runtime_linear", fake_runtime_linear)

    layer = lora_mod.LoRALinear(4, 3, bias=True, lora_rank=2)
    x = torch.randn(5, 4)
    out = layer(x)

    assert out.shape == (5, 3)
    assert calls[0] == ((5, 4), (3, 4), True)
    assert calls[1] == ((5, 4), (2, 4), False)
    assert calls[2] == ((5, 2), (3, 2), False)


def test_quantized_linear_uses_runtime_linear(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x"] = tuple(x.shape)
        seen["weight"] = tuple(weight.shape)
        seen["bias"] = bias is not None
        return torch.full((*x.shape[:-1], weight.shape[0]), 5.0, dtype=x.dtype)

    monkeypatch.setattr(quant_mod, "runtime_linear", fake_runtime_linear)

    layer = quant_mod.QuantizedLinearInt8(4, 3, bias=True)
    x = torch.randn(2, 4)
    out = layer(x)

    assert out.shape == (2, 3)
    assert torch.all(out == 5.0)
    assert seen == {"x": (2, 4), "weight": (3, 4), "bias": True}


def test_quantized_linear_reuses_cached_dequantized_weight(monkeypatch):
    seen = []

    def fake_runtime_linear(x, weight, bias):
        seen.append(id(weight))
        return torch.full((*x.shape[:-1], weight.shape[0]), 3.0, dtype=x.dtype)

    monkeypatch.setattr(quant_mod, "runtime_linear", fake_runtime_linear)

    layer = quant_mod.QuantizedLinearInt8(4, 3, bias=False)
    x = torch.randn(2, 4)
    layer(x)
    layer(x)

    assert len(seen) == 2
    assert seen[0] == seen[1]


def test_collect_linear_calibration_inputs_records_per_module_inputs():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    x = torch.randn(5, 4)

    captured = quant_mod.collect_linear_calibration_inputs(model, x, max_samples=3)

    assert set(captured) == {"0", "2"}
    assert captured["0"].shape == (3, 4)
    assert captured["2"].shape[1] == 3


def test_quantized_int8_gptq_calls_row_optimizer(monkeypatch):
    seen = {}

    def fake_gptq(weight, calibration_inputs, *, bits, calibration, percentile, damp=0.01):
        seen["weight"] = tuple(weight.shape)
        seen["inputs"] = tuple(calibration_inputs.shape)
        seen["bits"] = bits
        seen["calibration"] = calibration
        seen["percentile"] = percentile
        return weight.clone(), torch.ones(weight.shape[0], dtype=torch.float32, device=weight.device)

    monkeypatch.setattr(quant_mod, "_gptq_dequantize_rows", fake_gptq)

    layer = quant_mod.QuantizedLinearInt8(4, 3, bias=True)
    linear = torch.nn.Linear(4, 3, bias=True)
    layer.from_float(linear, calibration_inputs=torch.randn(8, 4), weight_opt="gptq")

    assert seen == {
        "weight": (3, 4),
        "inputs": (8, 4),
        "bits": 8,
        "calibration": "absmax",
        "percentile": 0.999,
    }


def test_quantized_int4_linear_uses_runtime_int4_linear(monkeypatch):
    seen = {}

    def fake_runtime_int4_linear(x, weight_packed, inv_scale, bias=None):
        seen["x"] = tuple(x.shape)
        seen["packed_weight"] = tuple(weight_packed.shape)
        seen["inv_scale"] = tuple(inv_scale.shape)
        seen["bias"] = bias is not None
        return torch.full((*x.shape[:-1], inv_scale.shape[0]), 4.0, dtype=x.dtype)

    monkeypatch.setattr(quant_mod, "runtime_int4_linear", fake_runtime_int4_linear)

    layer = quant_mod.QuantizedLinearInt4(4, 3, bias=True)
    x = torch.randn(2, 4)
    out = layer(x)

    assert out.shape == (2, 3)
    assert torch.all(out == 4.0)
    assert seen == {"x": (2, 4), "packed_weight": (3, 2), "inv_scale": (3,), "bias": True}
    assert layer.runtime_supports_packed_backend("cublaslt") is False


def test_quantized_int8_awq_and_static_activation_quant_precondition_runtime_input(monkeypatch):
    seen = {}

    def fake_runtime_quantize_activation_int8_rowwise(x, *, scale=None, method="absmax", percentile=0.999, eps=1e-8):
        del method, percentile, eps
        seen["x"] = x.clone()
        seen["act_scale"] = scale.clone() if isinstance(scale, torch.Tensor) else scale
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        return torch.ones_like(x, dtype=torch.int8), torch.full((rows,), 0.25, dtype=torch.float32, device=x.device)

    def fake_runtime_int8_linear_from_quantized_activation(qx, row_scale, qweight, inv_scale, bias=None, *, out_dtype=None):
        del row_scale, qweight, inv_scale, bias
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.full((*qx.shape[:-1], 3), 8.0, dtype=dtype, device=qx.device)

    monkeypatch.setattr(quant_mod, "runtime_quantize_activation_int8_rowwise", fake_runtime_quantize_activation_int8_rowwise)
    monkeypatch.setattr(quant_mod, "runtime_int8_linear_from_quantized_activation", fake_runtime_int8_linear_from_quantized_activation)

    layer = quant_mod.QuantizedLinearInt8(4, 3, bias=False)
    linear = torch.nn.Linear(4, 3, bias=False)
    calibration_inputs = torch.randn(16, 4).abs() + 0.1
    layer.from_float(
        linear,
        calibration_inputs=calibration_inputs,
        weight_opt="awq",
        activation_quant="static_int8",
        activation_quant_bits=8,
        activation_quant_method="absmax",
    )
    x = torch.randn(2, 4)
    out = layer(x)

    assert out.shape == (2, 3)
    assert torch.all(out == 8.0)
    assert torch.allclose(seen["x"], quant_mod._apply_pre_scale_to_input(x, layer.pre_scale))
    assert torch.allclose(seen["act_scale"], layer.act_scale)
    assert layer.weight_opt == "awq"
    assert layer.act_quant_mode == "static_int8"
    assert float(layer.act_scale.item()) > 0.0


def test_quantized_int4_spin_transforms_input_before_runtime_kernel(monkeypatch):
    seen = {}

    def fake_runtime_int4_linear(x, weight_packed, inv_scale, bias=None):
        seen["x"] = x.clone()
        return torch.full((*x.shape[:-1], inv_scale.shape[0]), 4.0, dtype=x.dtype)

    monkeypatch.setattr(quant_mod, "runtime_int4_linear", fake_runtime_int4_linear)

    layer = quant_mod.QuantizedLinearInt4(4, 3, bias=False)
    layer.spin_enabled_flag.fill_(1)
    layer.spin_signs.copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
    x = torch.arange(8, dtype=torch.float32).view(2, 4)
    out = layer(x)

    assert out.shape == (2, 3)
    assert torch.all(out == 4.0)
    assert torch.allclose(seen["x"], quant_mod.apply_spin_transform(x, layer.spin_signs))


def test_quantized_bitnet_linear_uses_runtime_bitnet_linear(monkeypatch):
    seen = {}

    def fake_runtime_bitnet_linear(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None):
        seen["x"] = tuple(x.shape)
        seen["packed_weight"] = tuple(packed_weight.shape)
        seen["scale_values"] = tuple(scale_values.shape)
        seen["layout_header"] = tuple(layout_header.shape)
        seen["segment_offsets"] = tuple(segment_offsets.shape)
        seen["bias"] = bias is not None
        return torch.full((*x.shape[:-1], int(layout_header[3].item())), 6.0, dtype=x.dtype)

    monkeypatch.setattr(quant_mod, "runtime_bitnet_linear", fake_runtime_bitnet_linear)

    layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=True)
    linear = torch.nn.Linear(4, 3, bias=True)
    layer.from_float(linear)
    x = torch.randn(2, 4)
    out = layer(x)

    assert out.shape == (2, 3)
    assert torch.all(out == 6.0)
    assert seen["x"] == (2, 4)
    assert seen["scale_values"] == (1,)
    assert seen["layout_header"] == (13,)
    assert seen["segment_offsets"] == (2,)
    assert seen["bias"] is True
    assert layer.runtime_supports_packed_backend("bitnet") is True
    assert layer.runtime_supports_packed_backend("cublaslt") is False


def test_quantized_bitnet_linear_exposes_packed_linear_spec():
    layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=True)
    layer.from_float(torch.nn.Linear(4, 3, bias=True))
    layer.spin_enabled_flag.fill_(1)
    layer.spin_signs.copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
    layer.pre_scale.copy_(torch.tensor([1.0, 0.5, 1.5, 1.0]))
    layer.act_quant_mode = "static_int8"
    layer.act_quant_method = "percentile"
    layer.act_quant_bits = 6
    layer.act_quant_percentile = 0.95
    layer.act_scale.fill_(0.25)

    spec = layer.runtime_packed_linear_spec(backend="bitnet", dtype=torch.float16, device="cpu")

    assert spec["format"] == "bitnet_w2a8"
    assert spec["backend"] == "bitnet"
    assert tuple(spec["packed_weight"].shape) == tuple(layer.packed_weight.shape)
    assert tuple(spec["scale_values"].shape) == tuple(layer.scale_values.shape)
    assert tuple(spec["layout_header"].shape) == (13,)
    assert tuple(spec["segment_offsets"].shape) == tuple(layer.segment_offsets.shape)
    assert spec["bias"] is not None and spec["bias"].dtype == torch.float16
    assert spec["spin_enabled"] is True
    assert torch.equal(spec["spin_signs"], layer.spin_signs.cpu())
    assert torch.equal(spec["pre_scale"], layer.pre_scale.cpu())
    assert spec["act_quant_mode"] == "static_int8"
    assert spec["act_quant_method"] == "percentile"
    assert spec["act_quant_bits"] == 6
    assert spec["act_quant_percentile"] == 0.95
    assert torch.equal(spec["act_scale"], layer.act_scale.cpu())


def test_quantized_bitnet_packed_linear_signature_tracks_activation_percentile():
    layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=True)
    layer.from_float(torch.nn.Linear(4, 3, bias=True))
    layer.act_quant_mode = "dynamic_int8"
    layer.act_quant_method = "percentile"
    layer.act_quant_bits = 6
    layer.act_quant_percentile = 0.9

    sig_a = runtime_ops_mod.packed_linear_module_signature(layer, backend="bitnet")

    layer.act_quant_percentile = 0.97
    sig_b = runtime_ops_mod.packed_linear_module_signature(layer, backend="bitnet")

    assert sig_a != sig_b


def test_resolve_packed_linear_module_spec_uses_backend_specific_payload():
    bitnet = quant_mod.QuantizedLinearBitNet(4, 3, bias=True)
    bitnet.from_float(torch.nn.Linear(4, 3, bias=True))
    bitnet.act_quant_mode = "dynamic_int8"
    bitnet.act_quant_method = "percentile"
    bitnet.act_quant_bits = 5
    bitnet.act_quant_percentile = 0.93
    bitnet_spec = runtime_ops_mod.resolve_packed_linear_module_spec(
        bitnet,
        backend="bitnet",
        reference=torch.randn(2, 4, dtype=torch.float16),
    )
    assert bitnet_spec["format"] == "bitnet_w2a8"
    assert bitnet_spec["backend"] == "bitnet"
    assert bitnet_spec["bias"] is not None and bitnet_spec["bias"].dtype == torch.float16
    assert bitnet_spec["act_quant_method"] == "percentile"
    assert bitnet_spec["act_quant_bits"] == 5
    assert bitnet_spec["act_quant_percentile"] == 0.93

    dense = torch.nn.Linear(4, 3, bias=True)
    dense_spec = runtime_ops_mod.resolve_packed_linear_module_spec(
        dense,
        backend="cublaslt",
        reference=torch.randn(2, 4, dtype=torch.float32),
    )
    assert dense_spec["format"] == "dense_packed"
    assert dense_spec["backend"] == "cublaslt"
    assert tuple(dense_spec["packed_weight"].shape) == tuple(dense.weight.shape)
    assert dense_spec["bias"] is not None and tuple(dense_spec["bias"].shape) == tuple(dense.bias.shape)


def test_resolve_packed_qkv_module_spec_uses_dense_and_bitnet_payloads():
    reference = torch.randn(2, 4, dtype=torch.float16)

    q_dense = torch.nn.Linear(4, 3, bias=True)
    k_dense = torch.nn.Linear(4, 5, bias=True)
    v_dense = torch.nn.Linear(4, 7, bias=True)
    dense_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_dense,
        k_dense,
        v_dense,
        backend="cublaslt",
        reference=reference,
    )
    assert dense_spec["format"] == "dense_qkv_packed"
    assert dense_spec["backend"] == "cublaslt"
    assert dense_spec["q_size"] == 3
    assert dense_spec["k_size"] == 5
    assert dense_spec["v_size"] == 7
    assert tuple(dense_spec["packed_weight"].shape) == (15, 4)

    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 5, bias=True).from_float(torch.nn.Linear(4, 5, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 7, bias=True).from_float(torch.nn.Linear(4, 7, bias=True))
    bitnet_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=reference,
    )
    assert bitnet_spec["format"] == "bitnet_qkv"
    assert bitnet_spec["backend"] == "bitnet"
    assert bitnet_spec["q_size"] == 3
    assert bitnet_spec["k_size"] == 5
    assert bitnet_spec["v_size"] == 7
    assert bitnet_spec["q_spec"]["format"] == "bitnet_w2a8"
    assert bitnet_spec["k_spec"]["format"] == "bitnet_w2a8"
    assert bitnet_spec["v_spec"]["format"] == "bitnet_w2a8"


def test_packed_bitnet_projection_helpers_execute_from_spec(monkeypatch):
    calls = []

    def fake_runtime_bitnet_linear(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None):
        del packed_weight, scale_values, segment_offsets, bias
        out_features = int(layout_header[3].item())
        calls.append((tuple(x.shape), out_features))
        return torch.full(x.shape[:-1] + (out_features,), float(len(calls)), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "bitnet_linear", fake_runtime_bitnet_linear)

    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 5, bias=True).from_float(torch.nn.Linear(4, 5, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 7, bias=True).from_float(torch.nn.Linear(4, 7, bias=True))
    qkv_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=torch.randn(2, 4, dtype=torch.float16),
    )

    qh, kh, vh = runtime_ops_mod.qkv_packed_spec_heads_projection(
        torch.randn(2, 3, 4, dtype=torch.float16),
        qkv_spec,
        q_heads=1,
        kv_heads=1,
        backend="bitnet",
    )

    assert qh.shape == (2, 1, 3, 3)
    assert kh.shape == (2, 1, 3, 5)
    assert vh.shape == (2, 1, 3, 7)
    assert calls[:3] == [((2, 3, 4), 3), ((2, 3, 4), 5), ((2, 3, 4), 7)]

    out_spec = runtime_ops_mod.resolve_packed_linear_module_spec(
        quant_mod.QuantizedLinearBitNet(7, 4, bias=True).from_float(torch.nn.Linear(7, 4, bias=True)),
        backend="bitnet",
        reference=torch.randn(2, 7, dtype=torch.float16),
    )
    out = runtime_ops_mod.head_output_packed_projection(
        torch.randn(2, 1, 3, 7, dtype=torch.float16),
        out_spec,
        backend="bitnet",
    )
    assert out.shape == (2, 3, 4)
    assert calls[3] == ((2, 3, 7), 4)


def test_bitnet_qkv_packed_projection_uses_native_helper_when_available(monkeypatch):
    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    qkv_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=torch.randn(2, 4, dtype=torch.float16),
    )

    calls = {}

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
            calls["q_x"] = tuple(q_x.shape)
            calls["k_x"] = tuple(k_x.shape)
            calls["v_x"] = tuple(v_x.shape)
            calls["q_heads"] = int(q_heads)
            calls["kv_heads"] = int(kv_heads)
            q_width = int(q_layout_header[3].item()) // int(q_heads)
            k_width = int(k_layout_header[3].item()) // int(kv_heads)
            v_width = int(v_layout_header[3].item()) // int(kv_heads)
            return (
                torch.full((q_x.shape[0], int(q_heads), q_x.shape[1], q_width), 1.0, dtype=q_x.dtype),
                torch.full((k_x.shape[0], int(kv_heads), k_x.shape[1], k_width), 2.0, dtype=k_x.dtype),
                torch.full((v_x.shape[0], int(kv_heads), v_x.shape[1], v_width), 3.0, dtype=v_x.dtype),
            )

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "bitnet_qkv_packed_heads_projection")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeNativeModule())
    monkeypatch.setattr(
        runtime_quant_mod,
        "bitnet_linear",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bitnet_linear fallback should not run")),
    )

    with torch.no_grad():
        qh, kh, vh = runtime_ops_mod.qkv_packed_spec_heads_projection(
            torch.randn(2, 3, 4, dtype=torch.float16),
            qkv_spec,
            q_heads=2,
            kv_heads=2,
            backend="bitnet",
        )

    assert qh.shape == (2, 2, 3, 2)
    assert kh.shape == (2, 2, 3, 2)
    assert vh.shape == (2, 2, 3, 2)
    assert calls == {
        "q_x": (2, 3, 4),
        "k_x": (2, 3, 4),
        "v_x": (2, 3, 4),
        "q_heads": 2,
        "kv_heads": 2,
    }


def test_bitnet_delta_roundtrip_restores_packed_buffers():
    model_src = torch.nn.Sequential(quant_mod.QuantizedLinearBitNet(4, 3, bias=True))
    model_dst = torch.nn.Sequential(quant_mod.QuantizedLinearBitNet(4, 3, bias=True))
    model_src[0].from_float(torch.nn.Linear(4, 3, bias=True))
    model_dst[0].from_float(torch.nn.Linear(4, 3, bias=True))
    model_src[0].spin_enabled_flag.fill_(1)
    model_src[0].spin_signs.copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
    model_dst[0].spin_enabled_flag.zero_()
    model_dst[0].spin_signs.fill_(1.0)
    model_dst[0]._cached_weight = torch.randn(3, 4)
    model_dst[0]._cached_weight_key = ("stale",)

    delta = delta_mod.build_delta(model=model_src)
    delta_mod.apply_delta(model_dst, delta)

    assert "0" in delta["quant"]
    info = delta["quant"]["0"]
    assert info["type"] == "bitnet_w2a8"
    assert "layout_header" in info
    assert "segment_offsets" in info
    assert torch.equal(model_dst[0].packed_weight.cpu(), model_src[0].packed_weight.cpu())
    assert torch.equal(model_dst[0].layout_header.cpu(), model_src[0].layout_header.cpu())
    assert torch.equal(model_dst[0].segment_offsets.cpu(), model_src[0].segment_offsets.cpu())
    assert bool(model_dst[0].spin_enabled_flag.item()) is True
    assert torch.equal(model_dst[0].spin_signs.cpu(), model_src[0].spin_signs.cpu())
    assert model_dst[0]._cached_weight is None
    assert model_dst[0]._cached_weight_key is None


def test_int8_delta_roundtrip_restores_quant_optimization_state():
    model_src = torch.nn.Sequential(quant_mod.QuantizedLinearInt8(4, 3, bias=False))
    model_dst = torch.nn.Sequential(quant_mod.QuantizedLinearInt8(4, 3, bias=False))
    linear = torch.nn.Linear(4, 3, bias=False)
    calibration_inputs = torch.randn(12, 4).abs() + 0.1
    model_src[0].from_float(
        linear,
        calibration_inputs=calibration_inputs,
        weight_opt="awq",
        activation_quant="static_int8",
        activation_quant_method="absmax",
    )
    model_dst[0].from_float(torch.nn.Linear(4, 3, bias=False))

    delta = delta_mod.build_delta(model=model_src)
    delta_mod.apply_delta(model_dst, delta)

    assert torch.equal(model_dst[0].pre_scale.cpu(), model_src[0].pre_scale.cpu())
    assert torch.equal(model_dst[0].act_scale.cpu(), model_src[0].act_scale.cpu())
    assert model_dst[0].weight_opt == model_src[0].weight_opt
    assert model_dst[0].act_quant_mode == model_src[0].act_quant_mode
    assert model_dst[0].act_quant_method == model_src[0].act_quant_method


def test_spin_transform_preserves_linear_algebra():
    torch.manual_seed(0)
    x = torch.randn(3, 6)
    w = torch.randn(5, 6)
    spin_signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    x_spin = quant_mod.apply_spin_transform(x, spin_signs)
    w_spin = quant_mod.apply_spin_transform(w, spin_signs)
    w_recovered = quant_mod.undo_spin_transform(w_spin, spin_signs)

    assert torch.allclose(x @ w.t(), x_spin @ w_spin.t(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(w, w_recovered, atol=1e-6, rtol=1e-6)


def test_sae_decode_uses_runtime_helpers(monkeypatch):
    calls = []

    def fake_runtime_activation(x, activation):
        calls.append(("activation", activation, tuple(x.shape)))
        return x + 2

    def fake_runtime_linear(x, weight, bias):
        calls.append(("linear", tuple(x.shape), tuple(weight.shape), bias is not None))
        return torch.ones(*x.shape[:-1], weight.shape[0], dtype=x.dtype)

    monkeypatch.setattr(sae_ops_mod, "runtime_activation", fake_runtime_activation)
    monkeypatch.setattr(sae_ops_mod, "runtime_linear", fake_runtime_linear)

    sae = sae_ops_mod.SparseAutoencoder(4, 2, bias=True)
    z = torch.randn(3, 2)
    x_hat = sae_ops_mod.sae_decode(sae, z)

    assert x_hat.shape == (3, 4)
    assert calls[0] == ("activation", "relu", (3, 2))
    assert calls[1] == ("linear", (3, 2), tuple(sae.decoder.weight.shape), True)


def test_tensor_parallel_uses_runtime_linear_in_source() -> None:
    source = _read("dist/parallel/tensor_parallel.py")
    assert "from runtime.ops import linear as runtime_linear" in source
    assert "runtime_linear(x, self.weight, self.bias)" in source
    assert "runtime_linear(x_local, self.weight, None)" in source
