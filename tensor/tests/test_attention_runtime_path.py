from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import attn.eager as eager_mod
import runtime.ops as runtime_ops_mod
from attn.eager import EagerAttention
from specs.config import ModelConfig


def _build_cfg(attn_dropout: float = 0.0) -> ModelConfig:
    return ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        vocab_size=64,
        dtype="float32",
        attn_dropout=attn_dropout,
    )


def _stub_output_projection(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, backend: str | None = None) -> torch.Tensor:
    del weight, bias, backend
    return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)


def test_eager_attention_prefers_runtime_attention_in_eval(monkeypatch):
    monkeypatch.delenv("ATTN_BACKEND", raising=False)
    monkeypatch.delenv("ATTN_BACKEND_FILE", raising=False)
    attn = EagerAttention(_build_cfg(), use_rope=False)
    attn.eval()
    calls = {"runtime": 0, "legacy": 0}

    def fake_runtime(q, k, v, attn_mask=None, is_causal=False, scale=None):
        del k, v, attn_mask, is_causal, scale
        calls["runtime"] += 1
        return torch.zeros_like(q)

    def fail_legacy(*args, **kwargs):
        del args, kwargs
        calls["legacy"] += 1
        raise AssertionError("legacy attention path should not be used on the default eval path")

    monkeypatch.setattr(eager_mod, "runtime_attention", fake_runtime)
    monkeypatch.setattr(eager_mod, "scaled_dot_product_attention", fail_legacy)
    monkeypatch.setattr(eager_mod, "runtime_head_output_projection", _stub_output_projection)

    x = torch.randn(2, 3, 16)
    y = attn(x, None, None, None)

    assert calls["runtime"] == 1
    assert calls["legacy"] == 0
    assert y.shape == x.shape


def test_eager_attention_respects_explicit_backend_override(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=False, backend_override="torch")
    attn.eval()
    calls = {"runtime": 0, "legacy": 0}

    def fail_runtime(*args, **kwargs):
        del args, kwargs
        calls["runtime"] += 1
        raise AssertionError("runtime attention path should not run when a backend is explicitly forced")

    def fake_legacy(q, k, v, attn_mask=None, dropout_p=0.0, backend=None, is_causal=None, scale=None):
        del k, v, attn_mask, dropout_p, backend, is_causal, scale
        calls["legacy"] += 1
        return torch.zeros_like(q)

    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime)
    monkeypatch.setattr(eager_mod, "scaled_dot_product_attention", fake_legacy)
    monkeypatch.setattr(eager_mod, "runtime_head_output_projection", _stub_output_projection)

    x = torch.randn(2, 3, 16)
    y = attn(x, None, None, None)

    assert calls["runtime"] == 0
    assert calls["legacy"] == 1
    assert y.shape == x.shape


def test_eager_attention_training_dropout_uses_legacy_path(monkeypatch):
    monkeypatch.delenv("ATTN_BACKEND", raising=False)
    monkeypatch.delenv("ATTN_BACKEND_FILE", raising=False)
    attn = EagerAttention(_build_cfg(attn_dropout=0.1), use_rope=False)
    attn.train()
    calls = {"runtime": 0, "legacy": 0}

    def fail_runtime(*args, **kwargs):
        del args, kwargs
        calls["runtime"] += 1
        raise AssertionError("runtime attention path should not run when training-time dropout is active")

    def fake_legacy(q, k, v, attn_mask=None, dropout_p=0.0, backend=None, is_causal=None, scale=None):
        del k, v, attn_mask, backend, is_causal, scale
        calls["legacy"] += 1
        assert float(dropout_p) == 0.1
        return torch.zeros_like(q)

    monkeypatch.setattr(eager_mod, "runtime_attention", fail_runtime)
    monkeypatch.setattr(eager_mod, "scaled_dot_product_attention", fake_legacy)
    monkeypatch.setattr(eager_mod, "select_attention_backend", lambda **kwargs: "torch")
    monkeypatch.setattr(eager_mod, "runtime_head_output_projection", _stub_output_projection)

    x = torch.randn(2, 3, 16)
    y = attn(x, None, None, None)

    assert calls["runtime"] == 0
    assert calls["legacy"] == 1
    assert y.shape == x.shape


def test_eager_attention_internal_rope_cache_uses_runtime_resolution(monkeypatch):
    attn = EagerAttention(_build_cfg(), use_rope=True)
    seen = {}

    def fake_resolve(*, reference, head_dim, base_theta, attention_scaling=1.0, position_ids=None):
        seen["reference_shape"] = tuple(reference.shape)
        seen["head_dim"] = head_dim
        seen["base_theta"] = base_theta
        seen["attention_scaling"] = attention_scaling
        seen["position_ids"] = position_ids
        cos = torch.zeros(reference.shape[1], head_dim, dtype=reference.dtype, device=reference.device)
        sin = torch.ones(reference.shape[1], head_dim, dtype=reference.dtype, device=reference.device)
        return cos, sin

    monkeypatch.setattr(eager_mod, "runtime_resolve_rotary_embedding", fake_resolve)

    x = torch.randn(2, 3, 16)
    attn._ensure_rope_cache(3, x)

    assert seen["reference_shape"] == (2, 3, 16)
    assert seen["head_dim"] == 4
    assert seen["base_theta"] == attn.rope_theta
    assert seen["attention_scaling"] == attn.rope_attention_scaling
    assert seen["position_ids"] is None
    assert attn._rope_cos.shape == (3, 4)
    assert attn._rope_sin.shape == (3, 4)


def test_eager_attention_grad_enabled_training_path_stays_eager_safe(monkeypatch):
    monkeypatch.delenv("ATTN_BACKEND", raising=False)
    monkeypatch.delenv("ATTN_BACKEND_FILE", raising=False)
    attn = EagerAttention(_build_cfg(attn_dropout=0.0), use_rope=True)
    attn.train()

    class FakeModule:
        def __getattr__(self, name):
            if name.endswith("_forward"):
                raise AssertionError(f"native {name} should not run on grad-enabled attention path")
            raise AttributeError(name)

    monkeypatch.setattr(
        runtime_ops_mod,
        "has_native_op",
        lambda name: name in {
            "linear",
            "split_heads",
            "merge_heads",
            "qkv_projection",
            "qkv_heads_projection",
            "head_output_projection",
            "rope",
            "attention_prefill",
            "attention_decode",
        },
    )
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeModule())

    x = torch.randn(2, 3, 16, requires_grad=True)
    y = attn(x, None, None, None)

    assert y.shape == x.shape

    y.sum().backward()
    assert x.grad is not None
    assert attn.w_q.weight.grad is not None
    assert attn.w_k.weight.grad is not None
    assert attn.w_v.weight.grad is not None
    assert attn.w_o.weight.grad is not None
