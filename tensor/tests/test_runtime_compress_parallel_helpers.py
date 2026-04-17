from __future__ import annotations

from pathlib import Path

import torch

import compress.lora as lora_mod
import compress.quantization as quant_mod
import interpret.features.sae_ops as sae_ops_mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_lora_linear_uses_runtime_linear_for_base_and_adapter_paths(monkeypatch):
    calls = []

    def fake_runtime_linear(x, weight, bias):
        calls.append((tuple(x.shape), tuple(weight.shape), bias is not None))
        return torch.ones(*x.shape[:-1], weight.shape[0], dtype=x.dtype)

    monkeypatch.setattr(lora_mod, "runtime_linear", fake_runtime_linear)

    layer = lora_mod.LoRALinear(4, 3, bias=True, lora_rank=2)
    x = torch.randn(5, 4)
    out = layer(x)

    assert out.shape == (5, 3)
    assert calls[0] == ((5, 4), (3, 4), True)
    assert calls[1] == ((5, 4), (2, 4), False)
    assert calls[2] == ((5, 2), (3, 2), False)


def test_quantized_linear_uses_runtime_linear(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x"] = tuple(x.shape)
        seen["weight"] = tuple(weight.shape)
        seen["bias"] = bias is not None
        return torch.full((*x.shape[:-1], weight.shape[0]), 5.0, dtype=x.dtype)

    monkeypatch.setattr(quant_mod, "runtime_linear", fake_runtime_linear)

    layer = quant_mod.QuantizedLinearInt8(4, 3, bias=True)
    x = torch.randn(2, 4)
    out = layer(x)

    assert out.shape == (2, 3)
    assert torch.all(out == 5.0)
    assert seen == {"x": (2, 4), "weight": (3, 4), "bias": True}


def test_sae_decode_uses_runtime_helpers(monkeypatch):
    calls = []

    def fake_runtime_activation(x, activation):
        calls.append(("activation", activation, tuple(x.shape)))
        return x + 2

    def fake_runtime_linear(x, weight, bias):
        calls.append(("linear", tuple(x.shape), tuple(weight.shape), bias is not None))
        return torch.ones(*x.shape[:-1], weight.shape[0], dtype=x.dtype)

    monkeypatch.setattr(sae_ops_mod, "runtime_activation", fake_runtime_activation)
    monkeypatch.setattr(sae_ops_mod, "runtime_linear", fake_runtime_linear)

    sae = sae_ops_mod.SparseAutoencoder(4, 2, bias=True)
    z = torch.randn(3, 2)
    x_hat = sae_ops_mod.sae_decode(sae, z)

    assert x_hat.shape == (3, 4)
    assert calls[0] == ("activation", "relu", (3, 2))
    assert calls[1] == ("linear", (3, 2), tuple(sae.decoder.weight.shape), True)


def test_tensor_parallel_uses_runtime_linear_in_source() -> None:
    source = _read("dist/parallel/tensor_parallel.py")
    assert "from runtime.ops import linear as runtime_linear" in source
    assert "runtime_linear(x, self.weight, self.bias)" in source
    assert "runtime_linear(x_local, self.weight, None)" in source
