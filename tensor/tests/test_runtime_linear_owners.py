from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import blocks.adapters as adapters_mod
import blocks.examples.example_lm as example_lm_mod
import blocks.moe_block as moe_block_mod
import model.heads as heads_mod
from specs.config import ModelConfig


def test_sequence_classification_head_uses_runtime_linear(monkeypatch):
    head = heads_mod.SequenceClassificationHead(hidden_size=4, num_labels=3)
    hidden = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    seen = {}

    def fake_runtime_linear(x, weight, bias=None, *, backend=None):
        seen["x"] = x
        seen["weight"] = weight
        seen["bias"] = bias
        seen["backend"] = backend
        return torch.full((x.shape[0], weight.shape[0]), 7.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(heads_mod, "runtime_linear", fake_runtime_linear)

    out = head(hidden, attention_mask)

    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    expected_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    assert torch.equal(seen["x"], expected_pooled)
    assert seen["weight"] is head.proj.weight
    assert seen["bias"] is head.proj.bias
    assert seen["backend"] is None
    assert torch.equal(out, torch.full((2, 3), 7.0))


def test_token_classification_head_uses_runtime_linear(monkeypatch):
    head = heads_mod.TokenClassificationHead(hidden_size=4, num_labels=5)
    hidden = torch.randn(2, 3, 4)
    seen = {}

    def fake_runtime_linear(x, weight, bias=None, *, backend=None):
        seen["x"] = x
        seen["weight"] = weight
        seen["bias"] = bias
        seen["backend"] = backend
        return torch.zeros(x.shape[0], x.shape[1], weight.shape[0], dtype=x.dtype, device=x.device)

    monkeypatch.setattr(heads_mod, "runtime_linear", fake_runtime_linear)

    out = head(hidden)

    assert seen["x"] is hidden
    assert seen["weight"] is head.proj.weight
    assert seen["bias"] is head.proj.bias
    assert seen["backend"] is None
    assert out.shape == (2, 3, 5)


def test_bottleneck_adapter_uses_runtime_linear_for_down_and_up(monkeypatch):
    adapter = adapters_mod.BottleneckAdapter(hidden_size=4, bottleneck=2, dropout_p=0.0, scale=1.5)
    x = torch.randn(2, 3, 4)
    seen = []

    def fake_runtime_linear(x_in, weight, bias=None, *, backend=None):
        seen.append({"x": x_in, "weight": weight, "bias": bias, "backend": backend})
        fill = float(len(seen))
        return torch.full(
            x_in.shape[:-1] + (weight.shape[0],),
            fill,
            dtype=x_in.dtype,
            device=x_in.device,
        )

    monkeypatch.setattr(adapters_mod, "runtime_linear", fake_runtime_linear)

    out = adapter(x)

    assert len(seen) == 2
    assert seen[0]["x"] is x
    assert seen[0]["weight"] is adapter.down.weight
    assert seen[0]["bias"] is adapter.down.bias
    assert seen[1]["weight"] is adapter.up.weight
    assert seen[1]["bias"] is adapter.up.bias
    assert torch.equal(out, torch.full_like(x, 3.0))


def test_moe_router_uses_runtime_linear(monkeypatch):
    moe = moe_block_mod.MoEMLP(hidden_size=4, ff_size=8, num_experts=2, k=1, dropout_p=0.0)
    moe.experts = nn.ModuleList([nn.Identity(), nn.Identity()])
    x = torch.randn(2, 3, 4)
    fake_logits = torch.randn(2, 3, 2)
    seen = {}

    def fake_runtime_linear(x_in, weight, bias=None, *, backend=None):
        seen["linear"] = {"x": x_in, "weight": weight, "bias": bias, "backend": backend}
        return fake_logits

    def fake_topk_router(logits, *, k):
        seen["router"] = {"logits": logits, "k": k}
        return "routes"

    def fake_combine(expert_out, routes):
        seen["combine"] = {"expert_out": expert_out, "routes": routes}
        return expert_out[0]

    def fake_balance(logits, routes, *, num_experts):
        seen["balance"] = {"logits": logits, "routes": routes, "num_experts": num_experts}
        return torch.tensor(0.25)

    monkeypatch.setattr(moe_block_mod, "runtime_linear", fake_runtime_linear)
    monkeypatch.setattr(moe_block_mod, "topk_router", fake_topk_router)
    monkeypatch.setattr(moe_block_mod, "combine_expert_outputs", fake_combine)
    monkeypatch.setattr(moe_block_mod, "load_balance_loss", fake_balance)

    y, l_aux = moe(x)

    assert seen["linear"]["x"] is x
    assert seen["linear"]["weight"] is moe.router.weight
    assert seen["linear"]["bias"] is moe.router.bias
    assert seen["router"] == {"logits": fake_logits, "k": 1}
    assert seen["combine"]["routes"] == "routes"
    assert len(seen["combine"]["expert_out"]) == 2
    assert torch.equal(y, x)
    assert float(l_aux) == 0.25


def test_example_transformer_lm_head_uses_runtime_linear(monkeypatch):
    monkeypatch.setattr(example_lm_mod, "init_transformer_stack", lambda blocks, recipe=None: None)
    cfg = ModelConfig(
        d_model=8,
        n_heads=2,
        n_layers=0,
        d_ff=16,
        vocab_size=32,
        dtype="float32",
    )
    model = example_lm_mod.ExampleTransformerLM(cfg)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    seen = {}

    def fake_runtime_embedding(weight, indices, padding_idx=None):
        seen["embedding"] = {"weight": weight, "indices": indices, "padding_idx": padding_idx}
        return torch.ones(indices.shape[0], indices.shape[1], weight.shape[1], dtype=weight.dtype)

    def fake_apply_native_norm(x, norm):
        seen["norm"] = {"x": x, "norm": norm}
        return x + 2

    def fake_runtime_linear(x, weight, bias=None, *, backend=None):
        seen["x"] = x
        seen["weight"] = weight
        seen["bias"] = bias
        seen["backend"] = backend
        return torch.zeros(x.shape[0], x.shape[1], weight.shape[0], dtype=x.dtype, device=x.device)

    monkeypatch.setattr(example_lm_mod, "runtime_embedding", fake_runtime_embedding)
    monkeypatch.setattr(example_lm_mod, "apply_native_norm", fake_apply_native_norm)
    monkeypatch.setattr(example_lm_mod, "runtime_linear", fake_runtime_linear)

    out = model(input_ids)

    assert seen["embedding"]["weight"] is model.embed.weight
    assert torch.equal(seen["embedding"]["indices"], input_ids)
    assert seen["embedding"]["padding_idx"] == model.embed.padding_idx
    assert seen["norm"]["norm"] is model.norm
    assert seen["weight"] is model.lm_head.weight
    assert seen["bias"] is model.lm_head.bias
    assert seen["backend"] is None
    assert out.shape == (1, 3, 32)
