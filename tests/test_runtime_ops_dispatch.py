from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runtime.ops as runtime_ops


class _UnexpectedNativeModule:
    def __getattr__(self, name: str):
        raise AssertionError(f"native dispatch should not be used in this test: {name}")


class _AttentionNativeModule:
    def __init__(self, result: torch.Tensor):
        self.result = result
        self.calls = 0

    def attention_forward(self, *args, **kwargs) -> torch.Tensor:
        self.calls += 1
        return self.result


def _force_eager_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(runtime_ops, "has_native_op", lambda name: True)
    monkeypatch.setattr(runtime_ops, "native_module", lambda: _UnexpectedNativeModule())


def test_linear_prefers_eager_reference_when_heuristic_trips(monkeypatch) -> None:
    _force_eager_dispatch(monkeypatch)
    monkeypatch.setattr(runtime_ops, "_prefer_eager_cuda_linear", lambda *args, **kwargs: True)

    x = torch.randn(3, 5, 7)
    weight = torch.randn(11, 7)
    bias = torch.randn(11)

    out = runtime_ops.linear(x, weight, bias)

    assert torch.allclose(out, F.linear(x, weight, bias))


def test_qkv_projection_prefers_eager_reference_when_heuristic_trips(monkeypatch) -> None:
    _force_eager_dispatch(monkeypatch)
    monkeypatch.setattr(runtime_ops, "_prefer_eager_cuda_qkv_projection", lambda *args, **kwargs: True)
    monkeypatch.setattr(runtime_ops, "_prefer_eager_cuda_linear", lambda *args, **kwargs: True)

    x = torch.randn(2, 4, 8)
    q_weight = torch.randn(8, 8)
    k_weight = torch.randn(8, 8)
    v_weight = torch.randn(8, 8)
    q_bias = torch.randn(8)
    k_bias = torch.randn(8)
    v_bias = torch.randn(8)

    q, k, v = runtime_ops.qkv_projection(
        x,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
    )

    assert torch.allclose(q, F.linear(x, q_weight, q_bias))
    assert torch.allclose(k, F.linear(x, k_weight, k_bias))
    assert torch.allclose(v, F.linear(x, v_weight, v_bias))


def test_mlp_prefers_eager_reference_when_heuristic_trips(monkeypatch) -> None:
    _force_eager_dispatch(monkeypatch)
    monkeypatch.setattr(runtime_ops, "_prefer_eager_cuda_mlp", lambda *args, **kwargs: True)
    monkeypatch.setattr(runtime_ops, "_prefer_eager_cuda_linear", lambda *args, **kwargs: True)

    x = torch.randn(2, 3, 6)
    w_in_weight = torch.randn(10, 6)
    w_in_bias = torch.randn(10)
    w_out_weight = torch.randn(4, 10)
    w_out_bias = torch.randn(4)

    out = runtime_ops.mlp(
        x,
        w_in_weight,
        w_in_bias,
        w_out_weight,
        w_out_bias,
        activation="relu",
        gated=False,
    )

    hidden = F.relu(F.linear(x, w_in_weight, w_in_bias))
    ref = F.linear(hidden, w_out_weight, w_out_bias)
    assert torch.allclose(out, ref)


def test_add_layer_norm_prefers_eager_reference_when_heuristic_trips(monkeypatch) -> None:
    _force_eager_dispatch(monkeypatch)
    monkeypatch.setattr(runtime_ops, "_prefer_eager_cuda_add_layer_norm", lambda *args, **kwargs: True)

    x = torch.randn(2, 3, 5)
    update = torch.randn(2, 3, 5)
    weight = torch.randn(5)
    bias = torch.randn(5)

    combined, normalized = runtime_ops.add_layer_norm(x, update, weight, bias, residual_scale=0.5, eps=1e-5)
    ref_combined = x + (update * 0.5)
    ref_normalized = F.layer_norm(ref_combined, (5,), weight, bias, 1e-5)

    assert torch.allclose(combined, ref_combined)
    assert torch.allclose(normalized, ref_normalized)


def test_embedding_prefers_eager_reference_when_heuristic_trips(monkeypatch) -> None:
    _force_eager_dispatch(monkeypatch)
    monkeypatch.setattr(runtime_ops, "_prefer_eager_cuda_embedding", lambda *args, **kwargs: True)

    weight = torch.randn(13, 7)
    indices = torch.tensor([[1, 2, 3], [4, 0, 5]], dtype=torch.long)

    out = runtime_ops.embedding(weight, indices, padding_idx=0)

    assert torch.allclose(out, F.embedding(indices, weight, padding_idx=0))


def test_attention_prefers_native_sm80_prefill_before_torch_library(monkeypatch) -> None:
    q = torch.randn(2, 4, 8, 64)
    expected = torch.randn_like(q)
    native = _AttentionNativeModule(expected)

    monkeypatch.setattr(runtime_ops, "_prefer_native_sm80_inference_attention", lambda *args, **kwargs: True)
    monkeypatch.setattr(runtime_ops, "prefer_torch_library_attention", lambda **kwargs: True)
    monkeypatch.setattr(runtime_ops, "has_native_op", lambda name: name == "attention_prefill")
    monkeypatch.setattr(runtime_ops, "native_module", lambda: native)
    monkeypatch.setattr(
        runtime_ops.F,
        "scaled_dot_product_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("torch library attention should be bypassed")
        ),
    )

    out = runtime_ops.attention(q, q, q, is_causal=True)

    assert native.calls == 1
    assert torch.equal(out, expected)


def test_attention_uses_torch_library_when_native_sm80_prefill_is_not_preferred(monkeypatch) -> None:
    q = torch.randn(2, 4, 8, 64)
    expected = torch.randn_like(q)

    monkeypatch.setattr(runtime_ops, "_prefer_native_sm80_inference_attention", lambda *args, **kwargs: False)
    monkeypatch.setattr(runtime_ops, "prefer_torch_library_attention", lambda **kwargs: True)
    monkeypatch.setattr(runtime_ops, "has_native_op", lambda name: True)
    monkeypatch.setattr(runtime_ops, "native_module", lambda: _UnexpectedNativeModule())
    monkeypatch.setattr(runtime_ops.F, "scaled_dot_product_attention", lambda *args, **kwargs: expected)

    out = runtime_ops.attention(q, q, q, is_causal=True)

    assert torch.equal(out, expected)
