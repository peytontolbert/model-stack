from __future__ import annotations

import torch

from compress.quantization import QuantizedLinearBitNet
from runtime.ops import bitnet_transform_input
from runtime.quant import bitnet_int8_linear_from_float


def test_bitnet_dynamic_transform_uses_row_local_scales_on_python_fallback() -> None:
    x = torch.tensor([[1.0, 2.0], [100.0, 200.0]], dtype=torch.float32)

    out = bitnet_transform_input(
        x,
        act_quant_mode="dynamic_int8",
        act_quant_bits=2,
        act_quant_method="absmax",
    )

    expected = torch.tensor([[0.0, 2.0], [0.0, 200.0]], dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_bitnet_dynamic_int8_from_float_uses_row_local_scales_on_python_fallback() -> None:
    x = torch.tensor([[1.0, 2.0], [100.0, 200.0]], dtype=torch.float32)
    qweight = torch.eye(2, dtype=torch.int8)
    inv_scale = torch.ones(2, dtype=torch.float32)

    out = bitnet_int8_linear_from_float(
        x,
        qweight,
        inv_scale,
        act_quant_mode="dynamic_int8",
        act_quant_bits=2,
        act_quant_method="absmax",
    )

    expected = torch.tensor([[0.0, 2.0], [0.0, 200.0]], dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_quantized_linear_bitnet_shared_int8_input_uses_row_local_dynamic_scales() -> None:
    layer = QuantizedLinearBitNet(2, 2, bias=False)
    layer.act_quant_mode = "dynamic_int8"
    layer.act_quant_bits = 2
    layer.act_quant_method = "absmax"
    x = torch.tensor([[1.0, 2.0], [100.0, 200.0]], dtype=torch.float32)

    qx, row_scale, out_dtype = layer.runtime_quantize_int8_input(x)

    assert out_dtype is torch.float32
    torch.testing.assert_close(row_scale, torch.tensor([2.0, 200.0]))
    torch.testing.assert_close(qx, torch.tensor([[0, 1], [0, 1]], dtype=torch.int8))


def test_quantized_linear_bitnet_accepts_dynamic_int4_alias() -> None:
    layer = QuantizedLinearBitNet(2, 2, bias=False)
    layer.act_quant_mode = "dynamic_int4"
    layer.act_quant_bits = 8
    layer.act_quant_method = "absmax"
    x = torch.tensor([[1.0, 2.0], [100.0, 200.0]], dtype=torch.float32)

    qx, row_scale, out_dtype = layer.runtime_quantize_int8_input(x)

    assert out_dtype is torch.float32
    torch.testing.assert_close(row_scale, torch.tensor([2.0 / 7.0, 200.0 / 7.0]))
    assert int(qx.abs().max().item()) <= 7
    torch.testing.assert_close(qx[:, 1], torch.tensor([7, 7], dtype=torch.int8))
