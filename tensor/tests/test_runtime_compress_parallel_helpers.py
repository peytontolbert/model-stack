from __future__ import annotations

from pathlib import Path

import pytest
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


def test_quantized_nf4_linear_uses_runtime_nf4_linear(monkeypatch):
    seen = {}

    def fake_runtime_nf4_linear(x, weight_packed, weight_scale, bias=None):
        seen["x"] = tuple(x.shape)
        seen["packed_weight"] = tuple(weight_packed.shape)
        seen["weight_scale"] = tuple(weight_scale.shape)
        seen["bias"] = bias is not None
        return torch.full((*x.shape[:-1], weight_scale.shape[0]), 4.5, dtype=x.dtype)

    monkeypatch.setattr(quant_mod, "runtime_nf4_linear", fake_runtime_nf4_linear)

    layer = quant_mod.QuantizedLinearNF4(4, 3, bias=True)
    x = torch.randn(2, 4)
    out = layer(x)

    assert out.shape == (2, 3)
    assert torch.all(out == 4.5)
    assert seen == {"x": (2, 4), "packed_weight": (3, 2), "weight_scale": (3,), "bias": True}
    assert layer.runtime_supports_packed_backend("cublaslt") is False


def test_quantized_int8_awq_and_static_activation_quant_precondition_runtime_input(monkeypatch):
    seen = {}

    def fake_runtime_int8_linear(x, qweight, inv_scale, bias=None, *, act_scale=None, act_method="absmax", act_percentile=0.999):
        del qweight, inv_scale, bias
        seen["x"] = x.clone()
        seen["act_scale"] = act_scale.clone() if isinstance(act_scale, torch.Tensor) else act_scale
        seen["act_method"] = act_method
        seen["act_percentile"] = act_percentile
        return torch.full((*x.shape[:-1], 3), 8.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(quant_mod, "runtime_int8_linear", fake_runtime_int8_linear)

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
    assert seen["act_method"] == "absmax"
    assert seen["act_percentile"] == 0.999
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


def test_quantized_bitnet_compute_backend_weight_packs_words_and_reuses_cache():
    layer = quant_mod.QuantizedLinearBitNet(20, 5, bias=True).from_float(torch.nn.Linear(20, 5, bias=True))

    packed_words_a, row_scales_a = layer._compute_backend_weight(device="cpu")
    packed_words_b, row_scales_b = layer._compute_backend_weight(device="cpu")

    assert packed_words_a.data_ptr() == packed_words_b.data_ptr()
    assert row_scales_a.data_ptr() == row_scales_b.data_ptr()
    assert packed_words_a.dtype == torch.int32
    assert row_scales_a.dtype == torch.float32
    assert tuple(packed_words_a.shape) == (1, 2, 128)
    assert tuple(row_scales_a.shape) == (1, 128)

    expected_word0_raw = (
        int(layer.packed_weight[0, 0].item())
        | (int(layer.packed_weight[0, 1].item()) << 8)
        | (int(layer.packed_weight[0, 2].item()) << 16)
        | (int(layer.packed_weight[0, 3].item()) << 24)
    )
    expected_word1_raw = int(layer.packed_weight[0, 4].item()) | (0x55 << 8) | (0x55 << 16) | (0x55 << 24)
    expected_word0 = torch.tensor(expected_word0_raw, dtype=torch.int64).to(dtype=torch.int32).item()
    expected_word1 = torch.tensor(expected_word1_raw, dtype=torch.int64).to(dtype=torch.int32).item()
    assert int(packed_words_a[0, 0, 0].item()) == expected_word0
    assert int(packed_words_a[0, 1, 0].item()) == expected_word1

    expected_row_scales = quant_mod._bitnet_row_scales(
        layer.scale_values,
        layer.layout_header,
        layer.segment_offsets,
    )
    assert torch.allclose(row_scales_a[0, : expected_row_scales.numel()], expected_row_scales)
    assert torch.count_nonzero(row_scales_a[0, expected_row_scales.numel() :]) == 0


def test_quantized_bitnet_decode_backend_weight_packs_masks_and_reuses_cache():
    layer = quant_mod.QuantizedLinearBitNet(20, 5, bias=True).from_float(torch.nn.Linear(20, 5, bias=True))

    nz_masks_a, sign_masks_a, row_scales_a = layer._decode_backend_weight(device="cpu")
    nz_masks_b, sign_masks_b, row_scales_b = layer._decode_backend_weight(device="cpu")

    assert nz_masks_a.data_ptr() == nz_masks_b.data_ptr()
    assert sign_masks_a.data_ptr() == sign_masks_b.data_ptr()
    assert row_scales_a.data_ptr() == row_scales_b.data_ptr()
    assert nz_masks_a.dtype == torch.int32
    assert sign_masks_a.dtype == torch.int32
    assert row_scales_a.dtype == torch.float32
    assert tuple(nz_masks_a.shape) == (1, 1, 128)
    assert tuple(sign_masks_a.shape) == (1, 1, 128)
    assert tuple(row_scales_a.shape) == (1, 128)

    signed = quant_mod._unpack_bitnet_signed(
        layer.packed_weight[: int(layer.layout_header[3].item())],
        original_last_dim=int(layer.layout_header[4].item()),
    ).to(dtype=torch.int8)
    bit_positions = (1 << torch.arange(signed.shape[1], dtype=torch.int64)).view(1, -1)
    expected_nz_mask = int((((signed[0] != 0).to(dtype=torch.int64) * bit_positions).sum()).item())
    expected_sign_mask = int((((signed[0] > 0).to(dtype=torch.int64) * bit_positions).sum()).item())
    assert int(nz_masks_a[0, 0, 0].item()) == expected_nz_mask
    assert int(sign_masks_a[0, 0, 0].item()) == expected_sign_mask

    expected_row_scales = quant_mod._bitnet_row_scales(
        layer.scale_values,
        layer.layout_header,
        layer.segment_offsets,
    )
    assert torch.allclose(row_scales_a[0, : expected_row_scales.numel()], expected_row_scales)
    assert torch.count_nonzero(row_scales_a[0, expected_row_scales.numel() :]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet input transform dispatch")
def test_bitnet_transform_input_uses_native_helper_when_available(monkeypatch):
    seen = {}

    class FakeNativeModule:
        def bitnet_transform_input_forward(
            self,
            x,
            spin_enabled,
            spin_signs,
            pre_scale,
            act_quant_mode,
            act_quant_method,
            act_quant_bits,
            act_quant_percentile,
            act_scale,
        ):
            seen["x_shape"] = tuple(x.shape)
            seen["spin_enabled"] = bool(spin_enabled)
            seen["spin_signs_shape"] = None if spin_signs is None else tuple(spin_signs.shape)
            seen["pre_scale_shape"] = None if pre_scale is None else tuple(pre_scale.shape)
            seen["act_quant_mode"] = act_quant_mode
            seen["act_quant_method"] = act_quant_method
            seen["act_quant_bits"] = int(act_quant_bits)
            seen["act_quant_percentile"] = float(act_quant_percentile)
            seen["act_scale_shape"] = None if act_scale is None else tuple(act_scale.shape)
            return x + 3.0

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "bitnet_transform_input")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeNativeModule())

    x = torch.randn(2, 4, device="cuda", dtype=torch.float16)
    out = runtime_ops_mod.bitnet_transform_input(
        x,
        spin_enabled=True,
        spin_signs=torch.ones(4),
        pre_scale=torch.full((4,), 0.5),
        act_quant_mode="static_int8",
        act_scale=torch.full((1,), 0.25),
        act_quant_bits=6,
        act_quant_method="percentile",
        act_quant_percentile=0.95,
    )

    assert torch.allclose(out, x + 3.0)
    assert seen == {
        "x_shape": (2, 4),
        "spin_enabled": True,
        "spin_signs_shape": (4,),
        "pre_scale_shape": (4,),
        "act_quant_mode": "static_int8",
        "act_quant_method": "percentile",
        "act_quant_bits": 6,
        "act_quant_percentile": 0.95,
        "act_scale_shape": (1,),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet input transform dispatch")
def test_bitnet_transform_input_native_matches_python_fallback(monkeypatch):
    if not runtime_ops_mod.has_native_op("bitnet_transform_input"):
        pytest.skip("native BitNet input transform op unavailable")

    x = torch.randn(4, 16, 64, device="cuda", dtype=torch.float16)
    spin_signs = torch.where(torch.arange(64, device="cuda") % 2 == 0, 1.0, -1.0).to(dtype=torch.float32)
    pre_scale = torch.linspace(0.75, 1.25, 64, device="cuda", dtype=torch.float32)
    act_scale = torch.tensor(0.125, device="cuda", dtype=torch.float32)

    native = runtime_ops_mod.bitnet_transform_input(
        x,
        spin_enabled=True,
        spin_signs=spin_signs,
        pre_scale=pre_scale,
        act_quant_mode="static_int8",
        act_scale=act_scale,
        act_quant_bits=8,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: False)
    fallback = runtime_ops_mod.bitnet_transform_input(
        x,
        spin_enabled=True,
        spin_signs=spin_signs,
        pre_scale=pre_scale,
        act_quant_mode="static_int8",
        act_scale=act_scale,
        act_quant_bits=8,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )

    assert torch.equal(native, fallback)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet input transform dispatch")
def test_bitnet_transform_input_native_nospin_matches_python_fallback(monkeypatch):
    if not runtime_ops_mod.has_native_op("bitnet_transform_input"):
        pytest.skip("native BitNet input transform op unavailable")

    x = torch.randn(4, 16, 64, device="cuda", dtype=torch.float16)
    pre_scale = torch.linspace(0.75, 1.25, 64, device="cuda", dtype=torch.float32)
    act_scale = torch.tensor(0.125, device="cuda", dtype=torch.float32)

    native = runtime_ops_mod.bitnet_transform_input(
        x,
        spin_enabled=False,
        pre_scale=pre_scale,
        act_quant_mode="static_int8",
        act_scale=act_scale,
        act_quant_bits=6,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: False)
    fallback = runtime_ops_mod.bitnet_transform_input(
        x,
        spin_enabled=False,
        pre_scale=pre_scale,
        act_quant_mode="static_int8",
        act_scale=act_scale,
        act_quant_bits=6,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )

    assert torch.equal(native, fallback)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet input transform dispatch")
def test_bitnet_transform_input_dynamic_native_matches_python_fallback(monkeypatch):
    if not runtime_ops_mod.has_native_op("bitnet_transform_input"):
        pytest.skip("native BitNet input transform op unavailable")

    x = torch.randn(4, 16, 64, device="cuda", dtype=torch.float16)
    pre_scale = torch.linspace(0.75, 1.25, 64, device="cuda", dtype=torch.float32)

    native = runtime_ops_mod.bitnet_transform_input(
        x,
        spin_enabled=False,
        pre_scale=pre_scale,
        act_quant_mode="dynamic_int8",
        act_scale=None,
        act_quant_bits=6,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: False)
    fallback = runtime_ops_mod.bitnet_transform_input(
        x,
        spin_enabled=False,
        pre_scale=pre_scale,
        act_quant_mode="dynamic_int8",
        act_scale=None,
        act_quant_bits=6,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    )

    assert torch.equal(native, fallback)


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


def test_packed_bitnet_linear_spec_supports_percentile_and_mse_activation_quantization(monkeypatch):
    seen = {}

    def fake_runtime_int8_linear_from_quantized_activation(qx, row_scale, qweight, inv_scale, bias=None, *, out_dtype=None):
        del inv_scale, bias
        seen.setdefault("qxs", []).append(qx.clone())
        seen.setdefault("row_scales", []).append(row_scale.clone())
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.full((*qx.shape[:-1], qweight.shape[0]), 2.0, dtype=dtype, device=qx.device)

    monkeypatch.setattr(runtime_quant_mod, "int8_linear_from_quantized_activation", fake_runtime_int8_linear_from_quantized_activation)
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: None)

    x = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, -0.75, 1.5, -1.0]], dtype=torch.float32)
    layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))

    layer.act_quant_mode = "dynamic_int8"
    layer.act_quant_method = "percentile"
    layer.act_quant_bits = 6
    layer.act_quant_percentile = 0.9
    percentile_spec = layer.runtime_packed_linear_spec(backend="bitnet", dtype=torch.float32, device="cpu")
    runtime_ops_mod.linear_from_packed_spec(x, percentile_spec, backend="bitnet")

    expected_percentile_scale = quant_mod.calibrate_activation_scale(x, method="percentile", bits=6, p=0.9)
    expected_percentile_qx = torch.round(x / expected_percentile_scale).clamp_(-31.0, 31.0).to(torch.int8)
    assert torch.equal(seen["qxs"][0], expected_percentile_qx)
    assert torch.allclose(seen["row_scales"][0], expected_percentile_scale.reshape(1).expand(x.shape[0]))

    layer.act_quant_mode = "static_int8"
    layer.act_quant_method = "mse"
    layer.act_quant_bits = 5
    layer.act_scale.fill_(0.125)
    mse_spec = layer.runtime_packed_linear_spec(backend="bitnet", dtype=torch.float32, device="cpu")
    runtime_ops_mod.linear_from_packed_spec(x, mse_spec, backend="bitnet")

    expected_mse_qx = torch.round(x / layer.act_scale).clamp_(-15.0, 15.0).to(torch.int8)
    assert torch.equal(seen["qxs"][1], expected_mse_qx)
    assert torch.allclose(seen["row_scales"][1], layer.act_scale.reshape(1).expand(x.shape[0]))


def test_packed_bitnet_int8_linear_spec_prefers_native_from_float_helper(monkeypatch):
    seen = {}

    class FakeNativeModule:
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
            del inv_scale, bias, act_scale
            seen["x"] = x.clone()
            seen["qweight_shape"] = tuple(qweight.shape)
            seen["pre_scale"] = None if pre_scale is None else pre_scale.clone()
            seen["act_quant_mode"] = act_quant_mode
            seen["act_quant_method"] = act_quant_method
            seen["act_quant_bits"] = act_quant_bits
            seen["act_quant_percentile"] = act_quant_percentile
            seen["out_dtype"] = out_dtype
            dtype = torch.float32 if out_dtype is None else out_dtype
            return torch.full((*x.shape[:-1], qweight.shape[0]), 4.0, dtype=dtype, device=x.device)

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "bitnet_int8_linear_from_float")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeNativeModule())
    monkeypatch.setattr(
        runtime_quant_mod,
        "int8_linear_from_quantized_activation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("native packed BitNet int8 helper should bypass Python quantized linear")),
    )

    layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(
        torch.nn.Linear(4, 3, bias=True),
        activation_quant="dynamic_int8",
        activation_quant_bits=6,
        activation_quant_method="percentile",
        activation_quant_percentile=0.9,
    )
    spec = layer.runtime_packed_linear_spec(backend="bitnet", dtype=torch.float32, device="cpu")
    x = torch.randn(2, 4, dtype=torch.float32)
    out = runtime_ops_mod.linear_from_packed_spec(x, spec, backend="bitnet")

    assert out.shape == (2, 3)
    assert torch.all(out == 4.0)
    assert seen["qweight_shape"] == (3, 4)
    assert torch.equal(seen["x"], x)
    assert seen["pre_scale"] is None
    assert seen["act_quant_mode"] == "dynamic_int8"
    assert seen["act_quant_method"] == "percentile"
    assert seen["act_quant_bits"] == 6
    assert seen["act_quant_percentile"] == 0.9
    assert seen["out_dtype"] == torch.float32


def test_quantized_bitnet_int8_packed_linear_spec_omits_inactive_transform_tensors():
    layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    layer.act_quant_mode = "dynamic_int8"
    layer.act_quant_bits = 6

    spec = layer.runtime_packed_linear_spec(backend="bitnet", dtype=torch.float16, device="cpu")

    assert spec["format"] == "bitnet_w2a8_int8"
    assert spec["spin_enabled"] is False
    assert spec["spin_signs"] is None
    assert spec["pre_scale"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BitNet input transform dispatch")
def test_packed_bitnet_linear_spec_uses_native_bitnet_transform_input_when_available(monkeypatch):
    seen = {}

    class FakeNativeModule:
        def bitnet_transform_input_forward(
            self,
            x,
            spin_enabled,
            spin_signs,
            pre_scale,
            act_quant_mode,
            act_quant_method,
            act_quant_bits,
            act_quant_percentile,
            act_scale,
        ):
            del spin_enabled, spin_signs, pre_scale, act_quant_mode, act_quant_method, act_quant_bits, act_quant_percentile, act_scale
            return x + 1.5

    def fake_runtime_bitnet_linear(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None):
        del packed_weight, scale_values, layout_header, segment_offsets, bias
        seen["x"] = x.clone()
        return torch.full((*x.shape[:-1], 3), 2.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "bitnet_transform_input")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeNativeModule())
    monkeypatch.setattr(runtime_quant_mod, "bitnet_linear", fake_runtime_bitnet_linear)

    layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    spec = layer.runtime_packed_linear_spec(backend="bitnet", dtype=torch.float16, device="cuda")
    x = torch.randn(2, 4, device="cuda", dtype=torch.float16)
    out = runtime_ops_mod.linear_from_packed_spec(x, spec, backend="bitnet")

    assert out.shape == (2, 3)
    assert torch.all(out == 2.0)
    assert torch.allclose(seen["x"], x + 1.5)


def test_quantized_bitnet_runtime_linear_prefers_bitnet_int8_from_float_helper_for_nospin_w2a8(monkeypatch):
    seen = {}

    def fake_runtime_bitnet_int8_linear_from_float(
        x,
        qweight,
        inv_scale,
        bias=None,
        *,
        pre_scale=None,
        act_quant_mode="dynamic_int8",
        act_scale=None,
        act_quant_bits=8,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    ):
        seen["x"] = x.clone()
        seen["qweight_shape"] = tuple(qweight.shape)
        seen["inv_scale_shape"] = tuple(inv_scale.shape)
        seen["bias"] = None if bias is None else bias.clone()
        seen["pre_scale"] = None if pre_scale is None else pre_scale.clone()
        seen["act_quant_mode"] = act_quant_mode
        seen["act_scale"] = act_scale
        seen["act_quant_bits"] = act_quant_bits
        seen["act_quant_method"] = act_quant_method
        seen["act_quant_percentile"] = act_quant_percentile
        return torch.full((*x.shape[:-1], 3), 7.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(quant_mod, "runtime_bitnet_int8_linear_from_float", fake_runtime_bitnet_int8_linear_from_float)
    monkeypatch.setattr(
        quant_mod,
        "runtime_bitnet_transform_input",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("split BitNet path should not run for no-spin W2A8")),
    )
    monkeypatch.setattr(
        quant_mod,
        "runtime_bitnet_linear_from_float",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("BitNet from-float path should not run for no-spin W2A8")),
    )
    monkeypatch.setattr(
        quant_mod,
        "runtime_int8_linear_from_quantized_activation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("split quantized int8 linear path should not run for no-spin W2A8")),
    )

    layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    layer.act_quant_mode = "dynamic_int8"
    layer.act_quant_bits = 6
    layer.act_quant_method = "mse"
    layer.act_quant_percentile = 0.95
    x = torch.randn(2, 4)

    out = layer.runtime_linear(x)

    assert out.shape == (2, 3)
    assert torch.all(out == 7.0)
    assert torch.allclose(seen["x"], x)
    assert seen["qweight_shape"] == (3, 4)
    assert seen["inv_scale_shape"] == (3,)
    assert seen["bias"].shape == (3,)
    assert seen["pre_scale"] is None
    assert seen["act_quant_mode"] == "dynamic_int8"
    assert seen["act_scale"] is None
    assert seen["act_quant_bits"] == 6
    assert seen["act_quant_method"] == "mse"
    assert seen["act_quant_percentile"] == 0.95


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


def test_quantized_int8_shared_input_signature_ignores_irrelevant_activation_metadata():
    layer = quant_mod.QuantizedLinearInt8(4, 3, bias=True)
    layer.from_float(torch.nn.Linear(4, 3, bias=True))
    layer.act_quant_bits = 8

    layer.act_quant_mode = "dynamic_int8"
    layer.act_quant_method = "absmax"
    layer.act_quant_percentile = 0.999
    layer.act_scale.fill_(0.25)
    dynamic_sig_a = layer.runtime_shared_int8_input_signature()

    layer.act_scale.fill_(0.5)
    dynamic_sig_b = layer.runtime_shared_int8_input_signature()
    assert dynamic_sig_a == dynamic_sig_b

    layer.act_quant_method = "percentile"
    dynamic_sig_c = layer.runtime_shared_int8_input_signature()
    assert dynamic_sig_a != dynamic_sig_c

    layer.act_quant_mode = "static_int8"
    layer.act_quant_method = "absmax"
    layer.act_quant_percentile = 0.999
    layer.act_scale.fill_(0.125)
    static_sig_a = layer.runtime_shared_int8_input_signature()

    layer.act_quant_method = "mse"
    layer.act_quant_percentile = 0.95
    static_sig_b = layer.runtime_shared_int8_input_signature()
    assert static_sig_a == static_sig_b

    layer.act_scale.fill_(0.25)
    static_sig_c = layer.runtime_shared_int8_input_signature()
    assert static_sig_a != static_sig_c


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
    assert bitnet_spec["format"] == "bitnet_w2a8_int8"
    assert bitnet_spec["backend"] == "bitnet"
    assert tuple(bitnet_spec["qweight"].shape) == (3, 4)
    assert tuple(bitnet_spec["inv_scale"].shape) == (3,)
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


def test_quantized_bitnet_percentile_weight_calibration_changes_packed_layout():
    linear = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [
                    [100.0, 0.2, 0.2, 0.2],
                    [0.5, 0.4, 0.3, 0.2],
                    [0.1, 0.1, 0.1, 0.1],
                ],
                dtype=torch.float32,
            )
        )

    absmax_layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=False).from_float(linear, calibration="absmax")
    percentile_layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=False).from_float(
        linear,
        calibration="percentile",
        percentile=0.5,
    )

    assert int(absmax_layer.layout_header[7].item()) == 0
    assert tuple(absmax_layer.scale_values.shape) == (1,)
    assert int(percentile_layer.layout_header[7].item()) == 2
    assert int(percentile_layer.layout_header[8].item()) == 1
    assert tuple(percentile_layer.scale_values.shape) == (3,)
    assert not torch.equal(absmax_layer.scale_values.cpu(), percentile_layer.scale_values.cpu())


def test_public_pack_bitnet_weight_accepts_explicit_metadata_overrides():
    weight = torch.tensor(
        [
            [0.8, -0.2, 0.1, -0.4],
            [0.3, 1.6, -0.9, 0.2],
        ],
        dtype=torch.float32,
    )
    scale_values = torch.tensor([0.5, 1.0], dtype=torch.float32)
    layout_header = torch.tensor([1, 16, 32, 2, 4, 16, 32, 2, 1, 1, 80, 1, 0], dtype=torch.int32)
    segment_offsets = torch.tensor([0, 2], dtype=torch.int32)

    packed_weight, packed_scales, packed_header, packed_offsets = runtime_ops_mod.pack_bitnet_weight(
        weight,
        scale_values=scale_values,
        layout_header=layout_header,
        segment_offsets=segment_offsets,
    )

    assert torch.equal(packed_scales.cpu(), scale_values)
    assert torch.equal(packed_header.cpu(), layout_header)
    assert torch.equal(packed_offsets.cpu(), segment_offsets)
    dequant = quant_mod._dequantize_bitnet_weight(
        packed_weight,
        packed_scales,
        packed_header,
        packed_offsets,
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [
            [0.5, 0.0, 0.0, -0.5],
            [0.0, 1.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(dequant, expected)


def test_public_pack_bitnet_weight_supports_percentile_calibration():
    weight = torch.tensor(
        [
            [100.0, 0.2, 0.2, 0.2],
            [0.5, 0.4, 0.3, 0.2],
            [0.1, 0.1, 0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    packed_weight, scale_values, layout_header, segment_offsets = runtime_ops_mod.pack_bitnet_weight(
        weight,
        calibration="percentile",
        percentile=0.5,
    )

    assert tuple(packed_weight.shape) == (16, 8)
    assert tuple(scale_values.shape) == (3,)
    assert int(layout_header[7].item()) == 2
    assert int(layout_header[8].item()) == 1
    assert torch.equal(segment_offsets.cpu(), torch.tensor([0, 3], dtype=torch.int32))


def test_quantized_bitnet_gptq_delegates_to_public_pack_helper(monkeypatch):
    seen = {}

    def fake_pack_bitnet_weight(
        weight,
        scale_values=None,
        layout_header=None,
        segment_offsets=None,
        *,
        calibration="absmax",
        percentile=0.999,
        weight_opt="none",
        calibration_inputs=None,
    ):
        del scale_values, layout_header, segment_offsets
        seen["weight"] = tuple(weight.shape)
        seen["calibration"] = calibration
        seen["percentile"] = percentile
        seen["weight_opt"] = weight_opt
        seen["inputs"] = None if calibration_inputs is None else tuple(calibration_inputs.shape)
        return (
            torch.ones((16, 8), dtype=torch.uint8),
            torch.ones((3,), dtype=torch.float32),
            torch.tensor([1, 16, 32, 3, 4, 16, 32, 2, 1, 1, 80, 1, 0], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32),
        )

    monkeypatch.setattr(quant_mod, "runtime_pack_bitnet_weight", fake_pack_bitnet_weight)

    layer = quant_mod.QuantizedLinearBitNet(4, 3, bias=True)
    linear = torch.nn.Linear(4, 3, bias=True)
    layer.from_float(
        linear,
        calibration_inputs=torch.randn(8, 4),
        weight_opt="gptq",
        calibration="percentile",
        percentile=0.9,
    )

    assert seen == {
        "weight": (3, 4),
        "calibration": "percentile",
        "percentile": 0.9,
        "weight_opt": "gptq",
        "inputs": (8, 4),
    }
    assert int(layer.layout_header[7].item()) == 2
    assert int(layer.layout_header[8].item()) == 1
    assert tuple(layer.scale_values.shape) == (3,)


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
    assert bitnet_spec["format"] == "bitnet_qkv_fused"
    assert bitnet_spec["backend"] == "bitnet"
    assert bitnet_spec["q_size"] == 3
    assert bitnet_spec["k_size"] == 5
    assert bitnet_spec["v_size"] == 7
    assert tuple(bitnet_spec["packed_weight"].shape) == (16, 8)
    assert tuple(bitnet_spec["scale_values"].shape) == (15,)
    assert tuple(bitnet_spec["layout_header"].shape) == (13,)
    assert tuple(bitnet_spec["segment_offsets"].shape) == (2,)
    assert bitnet_spec["packed_bias"] is not None and tuple(bitnet_spec["packed_bias"].shape) == (15,)


def test_resolve_packed_qkv_module_spec_falls_back_to_unfused_bitnet_when_input_transforms_differ():
    reference = torch.randn(2, 4, dtype=torch.float16)

    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 5, bias=True).from_float(torch.nn.Linear(4, 5, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 7, bias=True).from_float(torch.nn.Linear(4, 7, bias=True))
    k_bitnet.pre_scale.fill_(2.0)

    bitnet_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=reference,
    )

    assert bitnet_spec["format"] == "bitnet_qkv"
    assert bitnet_spec["q_spec"]["format"] == "bitnet_w2a8"
    assert bitnet_spec["k_spec"]["format"] == "bitnet_w2a8"
    assert bitnet_spec["v_spec"]["format"] == "bitnet_w2a8"


def test_resolve_packed_qkv_module_spec_keeps_fused_bitnet_when_none_mode_metadata_differs():
    reference = torch.randn(2, 4, dtype=torch.float16)

    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 5, bias=True).from_float(torch.nn.Linear(4, 5, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 7, bias=True).from_float(torch.nn.Linear(4, 7, bias=True))
    k_bitnet.act_quant_method = "mse"
    v_bitnet.act_quant_percentile = 0.75

    bitnet_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=reference,
    )

    assert bitnet_spec["format"] == "bitnet_qkv_fused"


def test_resolve_packed_qkv_module_spec_keeps_fused_bitnet_when_dynamic_mode_act_scale_differs():
    reference = torch.randn(2, 4, dtype=torch.float16)

    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 5, bias=True).from_float(torch.nn.Linear(4, 5, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 7, bias=True).from_float(torch.nn.Linear(4, 7, bias=True))
    for module in (q_bitnet, k_bitnet, v_bitnet):
        module.act_quant_mode = "dynamic_int8"
        module.act_quant_method = "percentile"
        module.act_quant_bits = 6
        module.act_quant_percentile = 0.9
    k_bitnet.act_scale.fill_(0.25)
    v_bitnet.act_scale.fill_(0.5)

    bitnet_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=reference,
    )

    assert bitnet_spec["format"] == "bitnet_qkv_fused_int8"
    assert tuple(bitnet_spec["qweight"].shape) == (15, 4)
    assert tuple(bitnet_spec["inv_scale"].shape) == (15,)


def test_resolve_packed_qkv_module_spec_keeps_fused_bitnet_when_spin_disabled_buffers_differ():
    reference = torch.randn(2, 4, dtype=torch.float16)

    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 5, bias=True).from_float(torch.nn.Linear(4, 5, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 7, bias=True).from_float(torch.nn.Linear(4, 7, bias=True))
    for module in (q_bitnet, k_bitnet, v_bitnet):
        module.act_quant_mode = "dynamic_int8"
        module.act_quant_method = "absmax"
        module.act_quant_bits = 6
    k_bitnet.spin_signs.copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
    v_bitnet.spin_signs.copy_(torch.tensor([-1.0, -1.0, 1.0, 1.0]))

    bitnet_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=reference,
    )

    assert bitnet_spec["format"] == "bitnet_qkv_fused_int8"
    assert tuple(bitnet_spec["qweight"].shape) == (15, 4)
    assert tuple(bitnet_spec["inv_scale"].shape) == (15,)


def test_resolve_packed_qkv_module_spec_keeps_fused_bitnet_when_static_mode_calibration_metadata_differs():
    reference = torch.randn(2, 4, dtype=torch.float16)

    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 3, bias=True).from_float(torch.nn.Linear(4, 3, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 5, bias=True).from_float(torch.nn.Linear(4, 5, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 7, bias=True).from_float(torch.nn.Linear(4, 7, bias=True))
    for module in (q_bitnet, k_bitnet, v_bitnet):
        module.act_quant_mode = "static_int8"
        module.act_quant_bits = 5
        module.act_scale.fill_(0.125)
    k_bitnet.act_quant_method = "mse"
    v_bitnet.act_quant_percentile = 0.8

    bitnet_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=reference,
    )

    assert bitnet_spec["format"] == "bitnet_qkv_fused_int8"
    assert tuple(bitnet_spec["qweight"].shape) == (15, 4)
    assert tuple(bitnet_spec["inv_scale"].shape) == (15,)


def test_bitnet_int8_qkv_packed_projection_shares_quantized_input(monkeypatch):
    reference = torch.randn(2, 4, dtype=torch.float16)
    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    for module in (q_bitnet, k_bitnet, v_bitnet):
        module.act_quant_mode = "dynamic_int8"
        module.act_quant_method = "percentile"
        module.act_quant_bits = 6
        module.act_quant_percentile = 0.9

    qkv_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=reference,
    )

    calls = []

    def fake_runtime_int8_linear_from_quantized_activation(qx, row_scale, qweight, inv_scale, bias=None, *, out_dtype=None):
        del row_scale, inv_scale, bias
        calls.append((tuple(qx.shape), int(qweight.shape[0]), out_dtype))
        dtype = torch.float32 if out_dtype is None else out_dtype
        return torch.full((*qx.shape[:-1], qweight.shape[0]), 3.0, dtype=dtype, device=qx.device)

    monkeypatch.setattr(runtime_quant_mod, "int8_linear_from_quantized_activation", fake_runtime_int8_linear_from_quantized_activation)
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: None)

    qh, kh, vh = runtime_ops_mod.qkv_packed_spec_heads_projection(
        torch.randn(2, 3, 4, dtype=torch.float16),
        qkv_spec,
        q_heads=2,
        kv_heads=2,
        backend="bitnet",
    )

    assert qkv_spec["format"] == "bitnet_qkv_fused_int8"
    assert qh.shape == (2, 2, 3, 2)
    assert kh.shape == (2, 2, 3, 2)
    assert vh.shape == (2, 2, 3, 2)
    assert calls == [((2, 3, 4), 12, torch.float16)]


def test_bitnet_int8_qkv_packed_projection_prefers_native_from_float_helper(monkeypatch):
    reference = torch.randn(2, 4, dtype=torch.float16)
    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    for module in (q_bitnet, k_bitnet, v_bitnet):
        module.act_quant_mode = "dynamic_int8"
        module.act_quant_method = "percentile"
        module.act_quant_bits = 6
        module.act_quant_percentile = 0.9

    qkv_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=reference,
    )

    seen = {}

    class FakeNativeModule:
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
            del inv_scale, bias, pre_scale, act_scale
            seen["x_shape"] = tuple(x.shape)
            seen["qweight_shape"] = tuple(qweight.shape)
            seen["act_quant_mode"] = act_quant_mode
            seen["act_quant_method"] = act_quant_method
            seen["act_quant_bits"] = act_quant_bits
            seen["act_quant_percentile"] = act_quant_percentile
            dtype = torch.float32 if out_dtype is None else out_dtype
            return torch.full((*x.shape[:-1], qweight.shape[0]), 5.0, dtype=dtype, device=x.device)

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "bitnet_int8_linear_from_float")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeNativeModule())
    monkeypatch.setattr(
        runtime_quant_mod,
        "int8_linear_from_quantized_activation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("native fused BitNet int8 helper should bypass Python quantized linear")),
    )

    qh, kh, vh = runtime_ops_mod.qkv_packed_spec_heads_projection(
        torch.randn(2, 3, 4, dtype=torch.float16),
        qkv_spec,
        q_heads=2,
        kv_heads=2,
        backend="bitnet",
    )

    assert qkv_spec["format"] == "bitnet_qkv_fused_int8"
    assert qh.shape == (2, 2, 3, 2)
    assert kh.shape == (2, 2, 3, 2)
    assert vh.shape == (2, 2, 3, 2)
    assert seen["x_shape"] == (2, 3, 4)
    assert seen["qweight_shape"] == (12, 4)
    assert seen["act_quant_mode"] == "dynamic_int8"
    assert seen["act_quant_method"] == "percentile"
    assert seen["act_quant_bits"] == 6
    assert seen["act_quant_percentile"] == 0.9


def test_bitnet_int8_qkv_packed_projection_prefers_native_fused_helper(monkeypatch):
    reference = torch.randn(2, 4, dtype=torch.float16)
    q_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    k_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    v_bitnet = quant_mod.QuantizedLinearBitNet(4, 4, bias=True).from_float(torch.nn.Linear(4, 4, bias=True))
    for module in (q_bitnet, k_bitnet, v_bitnet):
        module.act_quant_mode = "dynamic_int8"
        module.act_quant_method = "percentile"
        module.act_quant_bits = 6
        module.act_quant_percentile = 0.9

    qkv_spec = runtime_ops_mod.resolve_packed_qkv_module_spec(
        q_bitnet,
        k_bitnet,
        v_bitnet,
        backend="bitnet",
        reference=reference,
    )

    seen = {}

    class FakeNativeModule:
        def bitnet_int8_fused_qkv_packed_heads_projection_forward(
            self,
            x,
            qweight,
            inv_scale,
            packed_bias,
            pre_scale,
            act_quant_mode,
            act_quant_method,
            act_quant_bits,
            act_quant_percentile,
            act_scale,
            q_size,
            k_size,
            v_size,
            q_heads,
            kv_heads,
            out_dtype,
        ):
            del inv_scale, packed_bias, pre_scale, act_scale
            seen["x_shape"] = tuple(x.shape)
            seen["qweight_shape"] = tuple(qweight.shape)
            seen["act_quant_mode"] = act_quant_mode
            seen["act_quant_method"] = act_quant_method
            seen["act_quant_bits"] = act_quant_bits
            seen["act_quant_percentile"] = act_quant_percentile
            seen["q_size"] = q_size
            seen["k_size"] = k_size
            seen["v_size"] = v_size
            seen["q_heads"] = q_heads
            seen["kv_heads"] = kv_heads
            dtype = torch.float32 if out_dtype is None else out_dtype
            return (
                torch.full((x.shape[0], q_heads, x.shape[1], q_size // q_heads), 7.0, dtype=dtype, device=x.device),
                torch.full((x.shape[0], kv_heads, x.shape[1], k_size // kv_heads), 8.0, dtype=dtype, device=x.device),
                torch.full((x.shape[0], kv_heads, x.shape[1], v_size // kv_heads), 9.0, dtype=dtype, device=x.device),
            )

    monkeypatch.setattr(
        runtime_ops_mod,
        "has_native_op",
        lambda name: name == "bitnet_int8_fused_qkv_packed_heads_projection",
    )
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeNativeModule())
    monkeypatch.setattr(
        runtime_ops_mod,
        "_bitnet_int8_linear_from_float_input",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("native fused BitNet int8 QKV helper should bypass fused Python linear helper")
        ),
    )

    qh, kh, vh = runtime_ops_mod.qkv_packed_spec_heads_projection(
        torch.randn(2, 3, 4, dtype=torch.float16),
        qkv_spec,
        q_heads=2,
        kv_heads=2,
        backend="bitnet",
    )

    assert qkv_spec["format"] == "bitnet_qkv_fused_int8"
    assert qh.shape == (2, 2, 3, 2)
    assert kh.shape == (2, 2, 3, 2)
    assert vh.shape == (2, 2, 3, 2)
    assert torch.all(qh == 7.0)
    assert torch.all(kh == 8.0)
    assert torch.all(vh == 9.0)
    assert seen["x_shape"] == (2, 3, 4)
    assert seen["qweight_shape"] == (12, 4)
    assert seen["act_quant_mode"] == "dynamic_int8"
    assert seen["act_quant_method"] == "percentile"
    assert seen["act_quant_bits"] == 6
    assert seen["act_quant_percentile"] == 0.9
    assert seen["q_size"] == 4
    assert seen["k_size"] == 4
    assert seen["v_size"] == 4
    assert seen["q_heads"] == 2
    assert seen["kv_heads"] == 2


def test_packed_bitnet_projection_helpers_execute_from_spec(monkeypatch):
    calls = []

    def fake_runtime_bitnet_linear(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None):
        del packed_weight, scale_values, segment_offsets, bias
        out_features = int(layout_header[3].item())
        calls.append((tuple(x.shape), out_features))
        return torch.full(x.shape[:-1] + (out_features,), float(len(calls)), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(runtime_quant_mod, "bitnet_linear", fake_runtime_bitnet_linear)
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: None)

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
    assert calls[0] == ((2, 3, 4), 15)

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
    assert calls[1] == ((2, 3, 7), 4)


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
            del packed_weight, scale_values, segment_offsets, packed_bias
            calls["x"] = tuple(x.shape)
            calls["q_size"] = int(q_size)
            calls["k_size"] = int(k_size)
            calls["v_size"] = int(v_size)
            calls["q_heads"] = int(q_heads)
            calls["kv_heads"] = int(kv_heads)
            q_width = int(q_size) // int(q_heads)
            k_width = int(k_size) // int(kv_heads)
            v_width = int(v_size) // int(kv_heads)
            return (
                torch.full((x.shape[0], int(q_heads), x.shape[1], q_width), 1.0, dtype=x.dtype),
                torch.full((x.shape[0], int(kv_heads), x.shape[1], k_width), 2.0, dtype=x.dtype),
                torch.full((x.shape[0], int(kv_heads), x.shape[1], v_width), 3.0, dtype=x.dtype),
            )

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "bitnet_fused_qkv_packed_heads_projection")
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
        "x": (2, 3, 4),
        "q_size": 4,
        "k_size": 4,
        "v_size": 4,
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
