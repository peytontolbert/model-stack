from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.native import native_module


def _module():
    module = native_module()
    assert module is not None
    assert hasattr(module, "attention_partitioned_reference_forward")
    return module


def _cuda_device() -> torch.device:
    return torch.device(os.getenv("MODEL_STACK_TEST_DEVICE", "cuda:0"))


def test_partitioned_attention_reference_matches_torch_on_cpu_causal() -> None:
    module = _module()

    torch.manual_seed(0)
    q = torch.randn(2, 4, 129, 32)
    k = torch.randn(2, 4, 129, 32)
    v = torch.randn(2, 4, 129, 32)

    out = module.attention_partitioned_reference_forward(q, k, v, None, True, None, 32)
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_partitioned_attention_reference_matches_torch_on_cpu_with_gqa_and_mask() -> None:
    module = _module()

    torch.manual_seed(1)
    q = torch.randn(1, 4, 33, 16)
    k = torch.randn(1, 2, 33, 16)
    v = torch.randn(1, 2, 33, 16)
    mask = torch.randn(1, 4, 33, 33, dtype=torch.float32) * 0.01

    out = module.attention_partitioned_reference_forward(q, k, v, mask, False, None, 8)
    k_all = k.repeat_interleave(2, dim=1)
    v_all = v.repeat_interleave(2, dim=1)
    ref = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask, dropout_p=0.0, is_causal=False)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for partitioned attention reference parity")
def test_partitioned_attention_reference_matches_torch_on_cuda_long_context() -> None:
    module = _module()
    device = _cuda_device()

    torch.manual_seed(1024)
    q = torch.randn(1, 8, 1024, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 1024, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 8, 1024, 64, device=device, dtype=torch.bfloat16)
    scale = q.size(-1) ** -0.5

    out = module.attention_partitioned_reference_forward(q, k, v, None, True, scale, 256)
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=scale)
    torch.cuda.synchronize(device)

    max_abs_diff = (out.float() - ref.float()).abs().max().item()
    assert max_abs_diff <= 0.02
