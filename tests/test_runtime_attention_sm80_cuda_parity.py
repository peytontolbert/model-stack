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


def _skip_if_sm80_native_attention_unavailable(device: torch.device) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native SM80 attention parity")
    major, _minor = torch.cuda.get_device_capability(device)
    if major < 8 or major >= 9:
        pytest.skip("SM80/Ada-class CUDA device required for native SM80 attention parity")
    module = runtime_ops.native_module()
    if module is None or not hasattr(module, "attention_forward"):
        pytest.skip("native attention_forward is unavailable")
    if not runtime_ops.has_native_op("attention_prefill"):
        pytest.skip("native attention_prefill op is unavailable")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native SM80 attention parity")
@pytest.mark.parametrize("seq_len", [1024, 1537, 2048, 4096, 8192])
def test_native_sm80_attention_matches_torch_on_long_context_buckets(monkeypatch, seq_len: int) -> None:
    device = _test_device()
    _skip_if_sm80_native_attention_unavailable(device)

    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_PYTORCH_MEMEFF", "1")
    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_CUTLASS", "1")
    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_TENSORCORE", "1")
    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_FLASH", "1")
    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_INFERENCE", "0")
    monkeypatch.setenv("MODEL_STACK_SM80_INFERENCE_PREFILL_KERNEL", "64x64_rf")
    monkeypatch.setenv("MODEL_STACK_PREFER_NATIVE_SM80_INFERENCE_ATTENTION", "1")
    monkeypatch.setenv("MODEL_STACK_SM80_NATIVE_ATTENTION_MIN_SMS", "1")

    torch.manual_seed(seq_len)
    dtype = torch.bfloat16
    batch = 1
    heads = 8
    head_dim = 64
    scale = head_dim ** -0.5
    module = runtime_ops.native_module()
    assert module is not None

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    assert runtime_ops._prefer_native_sm80_inference_attention(
        q,
        k,
        v,
        None,
        is_causal=True,
        op_name="attention_prefill",
    )

    with torch.no_grad():
        _ = module.attention_forward(q, k, v, None, True, scale)
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        torch.cuda.synchronize(device)

        native_out = module.attention_forward(q, k, v, None, True, scale)
        torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        torch.cuda.synchronize(device)

    max_abs_diff = (native_out.float() - torch_out.float()).abs().max().item()
    assert max_abs_diff <= 0.02, (
        f"native SM80 attention drifted from torch at seq_len={seq_len}: "
        f"max_abs_diff={max_abs_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native SM80 split-K attention parity")
def test_native_sm80_split_kv_attention_matches_torch(monkeypatch) -> None:
    device = _test_device()
    _skip_if_sm80_native_attention_unavailable(device)

    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_PYTORCH_MEMEFF", "1")
    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_CUTLASS", "1")
    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_TENSORCORE", "1")
    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_FLASH", "1")
    monkeypatch.setenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_INFERENCE", "0")
    monkeypatch.setenv("MODEL_STACK_ENABLE_ATTENTION_PREFILL_SM80_SPLIT_KV", "1")
    monkeypatch.setenv("MODEL_STACK_SM80_SPLIT_KV_CHUNK", "512")
    monkeypatch.setenv("MODEL_STACK_SM80_INFERENCE_PREFILL_KERNEL", "64x64_rf")

    torch.manual_seed(4096)
    dtype = torch.float16
    batch = 1
    heads = 8
    seq_len = 4096
    head_dim = 64
    scale = head_dim ** -0.5
    module = runtime_ops.native_module()
    assert module is not None

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    with torch.no_grad():
        native_out = module.attention_forward(q, k, v, None, True, scale)
        torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        torch.cuda.synchronize(device)

    max_abs_diff = (native_out.float() - torch_out.float()).abs().max().item()
    assert max_abs_diff <= 0.02
