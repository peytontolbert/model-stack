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

import runtime.ops as runtime_ops


def _test_device() -> torch.device:
    return torch.device(os.getenv("MODEL_STACK_TEST_DEVICE", "cuda:0"))


def _skip_if_native_activation_unavailable(device: torch.device) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native activation parity")
    module = runtime_ops.native_module()
    if module is None or not hasattr(module, "activation_forward"):
        pytest.skip("native activation_forward is unavailable")
    if not runtime_ops.has_native_op("activation"):
        pytest.skip("native activation op is unavailable")
    if not runtime_ops.has_native_op("gated_activation"):
        pytest.skip("native gated_activation op is unavailable")
    if device.type != "cuda":
        pytest.skip("CUDA device required for native activation parity")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native activation parity")
def test_native_activation_leaky_relu_half_squared_matches_torch() -> None:
    device = _test_device()
    _skip_if_native_activation_unavailable(device)

    torch.manual_seed(0)
    x = torch.randn(23, 41, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        native_out = runtime_ops.activation(x, "leaky_relu_0p5_squared")
        torch_out = F.leaky_relu(x, negative_slope=0.5).square()
        torch.cuda.synchronize(device)

    max_abs_diff = (native_out.float() - torch_out.float()).abs().max().item()
    assert max_abs_diff <= 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native gated activation parity")
def test_native_gated_activation_leaky_relu_half_squared_matches_torch() -> None:
    device = _test_device()
    _skip_if_native_activation_unavailable(device)

    torch.manual_seed(1)
    x = torch.randn(19, 37, device=device, dtype=torch.bfloat16)
    gate = torch.randn(19, 37, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        native_out = runtime_ops.gated_activation(x, gate, "leaky_relu_0p5_squared")
        torch_out = F.leaky_relu(x, negative_slope=0.5).square() * gate
        torch.cuda.synchronize(device)

    max_abs_diff = (native_out.float() - torch_out.float()).abs().max().item()
    assert max_abs_diff <= 0.02
