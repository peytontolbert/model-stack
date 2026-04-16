from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runtime.ops as runtime_ops_mod


class _FailingNativeModule:
    def __getattr__(self, name):
        if name.endswith("_forward"):
            raise AssertionError(f"native {name} should not run on grad-enabled path")
        raise AttributeError(name)


def _rms_norm_reference(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    mean_sq = (x.float() * x.float()).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(mean_sq + float(eps))
    if weight is not None:
        out = out * weight
    return out.to(dtype=x.dtype)


def _apply_rotary_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_b = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sin_b = sin.view(1, 1, sin.shape[0], sin.shape[1])

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos_b) + (rotate_half(q) * sin_b), (k * cos_b) + (rotate_half(k) * sin_b)


def test_runtime_norm_ops_use_eager_reference_when_grad_enabled(monkeypatch):
    x = torch.randn(2, 3, 4, requires_grad=True)
    update = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(4, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)

    monkeypatch.setattr(
        runtime_ops_mod,
        "has_native_op",
        lambda name: name in {"rms_norm", "layer_norm", "add_rms_norm", "add_layer_norm", "residual_add"},
    )
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: _FailingNativeModule())

    rms = runtime_ops_mod.rms_norm(x, weight=weight, eps=1e-6)
    rms_ref = _rms_norm_reference(x, weight, 1e-6)
    assert torch.allclose(rms, rms_ref)

    layer = runtime_ops_mod.layer_norm(x, weight=weight, bias=bias, eps=1e-5)
    layer_ref = F.layer_norm(x, (x.shape[-1],), weight, bias, 1e-5)
    assert torch.allclose(layer, layer_ref)

    combined_rms, normalized_rms = runtime_ops_mod.add_rms_norm(
        x,
        update,
        weight=weight,
        residual_scale=0.5,
        eps=1e-6,
    )
    combined_rms_ref = x + (update * 0.5)
    normalized_rms_ref = _rms_norm_reference(combined_rms_ref, weight, 1e-6)
    assert torch.allclose(combined_rms, combined_rms_ref)
    assert torch.allclose(normalized_rms, normalized_rms_ref)

    combined_layer, normalized_layer = runtime_ops_mod.add_layer_norm(
        x,
        update,
        weight=weight,
        bias=bias,
        residual_scale=0.5,
        eps=1e-5,
    )
    combined_layer_ref = x + (update * 0.5)
    normalized_layer_ref = F.layer_norm(combined_layer_ref, (x.shape[-1],), weight, bias, 1e-5)
    assert torch.allclose(combined_layer, combined_layer_ref)
    assert torch.allclose(normalized_layer, normalized_layer_ref, atol=1e-5, rtol=1e-4)

    residual = runtime_ops_mod.residual_add(x, update, residual_scale=0.5)
    assert torch.allclose(residual, x + (update * 0.5))

    total = rms.sum() + layer.sum() + combined_rms.sum() + normalized_rms.sum()
    total = total + combined_layer.sum() + normalized_layer.sum() + residual.sum()
    total.backward()
    assert x.grad is not None
    assert update.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None


def test_runtime_apply_rotary_uses_eager_reference_when_grad_enabled(monkeypatch):
    q = torch.randn(2, 4, 3, 6, requires_grad=True)
    k = torch.randn(2, 4, 3, 6, requires_grad=True)
    cos = torch.randn(3, 6, requires_grad=True)
    sin = torch.randn(3, 6, requires_grad=True)

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "rope")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: _FailingNativeModule())

    q_out, k_out = runtime_ops_mod.apply_rotary(q, k, cos, sin)
    q_ref, k_ref = _apply_rotary_reference(q, k, cos, sin)

    assert torch.allclose(q_out, q_ref)
    assert torch.allclose(k_out, k_ref)

    (q_out.sum() + k_out.sum()).backward()
    assert q.grad is not None
    assert k.grad is not None
    assert cos.grad is not None
    assert sin.grad is not None


def test_runtime_attention_uses_eager_reference_when_grad_enabled(monkeypatch):
    q = torch.randn(2, 4, 3, 5, requires_grad=True)
    k = torch.randn(2, 4, 3, 5, requires_grad=True)
    v = torch.randn(2, 4, 3, 5, requires_grad=True)

    monkeypatch.setattr(
        runtime_ops_mod,
        "has_native_op",
        lambda name: name in {"attention_prefill", "attention_decode"},
    )
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: _FailingNativeModule())

    out = runtime_ops_mod.attention(q, k, v, is_causal=True)

    scores = torch.matmul(q, k.transpose(2, 3)) * (q.shape[-1] ** -0.5)
    causal = torch.triu(
        torch.ones(q.shape[2], k.shape[2], device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal.view(1, 1, q.shape[2], k.shape[2]), float("-inf"))
    ref = torch.matmul(torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype), v)

    assert torch.allclose(out, ref)

    out.sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_runtime_projection_and_embedding_ops_use_eager_reference_when_grad_enabled(monkeypatch):
    x = torch.randn(2, 3, 8, requires_grad=True)
    q_weight = torch.randn(8, 8, requires_grad=True)
    q_bias = torch.randn(8, requires_grad=True)
    k_weight = torch.randn(8, 8, requires_grad=True)
    k_bias = torch.randn(8, requires_grad=True)
    v_weight = torch.randn(8, 8, requires_grad=True)
    v_bias = torch.randn(8, requires_grad=True)
    out_weight = torch.randn(8, 8, requires_grad=True)
    out_bias = torch.randn(8, requires_grad=True)
    embed_weight = torch.randn(32, 8, requires_grad=True)
    indices = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)

    monkeypatch.setattr(
        runtime_ops_mod,
        "has_native_op",
        lambda name: name in {
            "linear",
            "split_heads",
            "merge_heads",
            "qkv_projection",
            "qkv_heads_projection",
            "qkv_packed_heads_projection",
            "head_output_projection",
            "embedding",
        },
    )
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: _FailingNativeModule())

    q, k, v = runtime_ops_mod.qkv_heads_projection(
        x,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        q_heads=2,
        kv_heads=2,
    )
    q_ref = F.linear(x, q_weight, q_bias).view(2, 3, 2, 4).permute(0, 2, 1, 3).contiguous()
    k_ref = F.linear(x, k_weight, k_bias).view(2, 3, 2, 4).permute(0, 2, 1, 3).contiguous()
    v_ref = F.linear(x, v_weight, v_bias).view(2, 3, 2, 4).permute(0, 2, 1, 3).contiguous()
    assert torch.allclose(q, q_ref)
    assert torch.allclose(k, k_ref)
    assert torch.allclose(v, v_ref)

    projected = runtime_ops_mod.head_output_projection(q, out_weight, out_bias)
    projected_ref = F.linear(q.permute(0, 2, 1, 3).contiguous().view(2, 3, 8), out_weight, out_bias)
    assert torch.allclose(projected, projected_ref)

    embedded = runtime_ops_mod.embedding(embed_weight, indices)
    embedded_ref = F.embedding(indices, embed_weight)
    assert torch.allclose(embedded, embedded_ref)

    (q.sum() + k.sum() + v.sum() + projected.sum() + embedded.sum()).backward()
    assert x.grad is not None
    assert q_weight.grad is not None
    assert k_weight.grad is not None
    assert v_weight.grad is not None
    assert out_weight.grad is not None
    assert embed_weight.grad is not None
