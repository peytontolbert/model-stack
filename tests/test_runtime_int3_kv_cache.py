import os

import pytest
import torch

from runtime.kv_cache import quantize_pack_int3_lastdim, unpack_dequantize_int3_lastdim
from runtime.native import has_native_op, native_module, runtime_info


def test_int3_kv_native_metadata_exposed():
    module = native_module()
    assert has_native_op("int3_kv_pack")
    assert has_native_op("int3_kv_dequantize")
    assert hasattr(module, "int3_kv_pack_forward")
    assert hasattr(module, "int3_kv_dequantize_forward")
    backend_ops = set(runtime_info().get("cuda_backend_ops", ()))
    if torch.cuda.is_available():
        assert "int3_kv_pack" in backend_ops
        assert "int3_kv_dequantize" in backend_ops


def test_int3_kv_native_cpu_fallback_quantizes_last_dim():
    module = native_module()
    x = torch.randn(2, 3, 17, dtype=torch.float32)
    packed, scale = module.int3_kv_pack_forward(x)
    y = module.int3_kv_dequantize_forward(packed, scale, x.shape[-1], torch.float32)

    assert packed.shape == (2, 3, 9)
    assert scale.shape == (2, 3)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    max_steps = ((x - y).abs() / scale.unsqueeze(-1)).max()
    assert float(max_steps) <= 0.51


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native int3 KV kernel")
def test_int3_kv_cuda_route_matches_reference_error_bound():
    torch.manual_seed(123)
    x = torch.randn(4, 8, 64, device="cuda", dtype=torch.bfloat16)

    os.environ["MODEL_STACK_DISABLE_NATIVE_INT3_KV"] = "1"
    try:
        packed_ref, scale_ref, dim_ref = quantize_pack_int3_lastdim(x)
        y_ref = unpack_dequantize_int3_lastdim(packed_ref, scale_ref, dim_ref, dtype=torch.bfloat16)
    finally:
        os.environ.pop("MODEL_STACK_DISABLE_NATIVE_INT3_KV", None)

    packed, scale, dim = quantize_pack_int3_lastdim(x)
    y = unpack_dequantize_int3_lastdim(packed, scale, dim, dtype=torch.bfloat16)
    torch.cuda.synchronize()

    assert packed.shape == packed_ref.shape
    assert scale.shape == scale_ref.shape
    assert dim == dim_ref == x.shape[-1]
    assert float((scale - scale_ref).abs().max()) <= 1e-5
    assert float(((x.float() - y.float()).abs() / scale.unsqueeze(-1)).max()) <= 0.51
    assert float(((x.float() - y_ref.float()).abs() / scale_ref.unsqueeze(-1)).max()) <= 0.51
