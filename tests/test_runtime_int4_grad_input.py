import pytest
import torch

from compress.quantization import QuantizedLinearInt4
from runtime.native import has_native_op, native_module
from runtime.quant import _dequantize_packed_int4_weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native int4 grad-input kernel")
def test_native_int4_grad_input_matches_dequantized_reference():
    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    dense = torch.nn.Linear(32, 48, bias=False).to(device=device, dtype=dtype)
    int4 = QuantizedLinearInt4(32, 48, bias=False).to(device=device, dtype=dtype)
    int4.from_float(dense)
    grad_out = torch.randn(17, 48, device=device, dtype=dtype)
    scale = int4.inv_scale.to(dtype=torch.float32)

    module = native_module()
    assert has_native_op("int4_linear_grad_input")
    assert hasattr(module, "int4_linear_grad_input_forward")

    actual = module.int4_linear_grad_input_forward(grad_out, int4.qweight_packed, scale, 32)
    weight = _dequantize_packed_int4_weight(
        int4.qweight_packed,
        scale,
        original_last_dim=32,
        dtype=dtype,
    )
    expected = grad_out.matmul(weight)
    torch.cuda.synchronize()

    assert actual.shape == (17, 32)
    assert torch.isfinite(actual).all()
    assert float((actual.float() - expected.float()).abs().max()) <= 0.04
