from __future__ import annotations

import torch
import torch.nn as nn

import blocks.stack as stack_mod
import interpret.features.sae as sae_mod
import interpret.probes.linear as probe_mod


def test_transformer_stack_uses_apply_native_norm(monkeypatch):
    calls = []

    def fake_execute(blocks, x, attn_mask, cache):
        calls.append(("execute", tuple(x.shape)))
        return x + 1

    def fake_norm(x, norm):
        calls.append(("norm", norm))
        return x + 2

    monkeypatch.setattr(stack_mod, "execute_block_stack", fake_execute)
    monkeypatch.setattr(stack_mod, "apply_native_norm", fake_norm)

    stack = stack_mod.TransformerStack(nn.ModuleList(), nn.LayerNorm(4))
    x = torch.zeros(2, 3, 4)
    out = stack(x)

    assert torch.equal(out, x + 3)
    assert calls[0] == ("execute", (2, 3, 4))
    assert calls[1][0] == "norm"


def test_encoder_decoder_stack_uses_apply_native_norm(monkeypatch):
    calls = []

    def fake_encode(blocks, x, enc_pad_mask):
        calls.append(("encode", tuple(x.shape)))
        return x + 1

    def fake_decode(blocks, x, memory, dec_attn_mask, memory_mask, dec_cache):
        calls.append(("decode", tuple(x.shape), tuple(memory.shape)))
        return x + memory

    def fake_norm(x, norm):
        calls.append(("norm", norm))
        return x + 2

    monkeypatch.setattr(stack_mod, "execute_encoder_stack", fake_encode)
    monkeypatch.setattr(stack_mod, "execute_decoder_stack", fake_decode)
    monkeypatch.setattr(stack_mod, "apply_native_norm", fake_norm)

    stack = stack_mod.EncoderDecoderStack(nn.ModuleList(), nn.ModuleList(), nn.LayerNorm(4))
    enc_x = torch.zeros(2, 3, 4)
    dec_x = torch.ones(2, 3, 4)
    out = stack(enc_x, dec_x)

    assert torch.equal(out, dec_x + (enc_x + 1) + 2)
    assert calls[0] == ("encode", (2, 3, 4))
    assert calls[1] == ("decode", (2, 3, 4), (2, 3, 4))
    assert calls[2][0] == "norm"


def test_linear_probe_uses_runtime_linear(monkeypatch):
    seen = {}

    def fake_runtime_linear(x, weight, bias):
        seen["x"] = x
        seen["weight"] = weight
        seen["bias"] = bias
        return torch.full((x.shape[0], weight.shape[0]), 7.0, dtype=x.dtype)

    monkeypatch.setattr(probe_mod, "runtime_linear", fake_runtime_linear)

    probe = probe_mod.LinearProbe(4, 3, bias=True)
    x = torch.randn(5, 4)
    out = probe(x)

    assert out.shape == (5, 3)
    assert torch.all(out == 7.0)
    assert seen["x"] is x
    assert seen["weight"] is probe.linear.weight
    assert seen["bias"] is probe.linear.bias


def test_sparse_autoencoder_uses_runtime_linear_and_activation(monkeypatch):
    calls = []

    def fake_runtime_linear(x, weight, bias):
        calls.append(("linear", tuple(x.shape), tuple(weight.shape), bias is not None))
        return torch.ones(*x.shape[:-1], weight.shape[0], dtype=x.dtype)

    def fake_runtime_activation(x, activation):
        calls.append(("activation", activation, tuple(x.shape)))
        return x + 3

    monkeypatch.setattr(sae_mod, "runtime_linear", fake_runtime_linear)
    monkeypatch.setattr(sae_mod, "runtime_activation", fake_runtime_activation)

    sae = sae_mod.SparseAutoencoder(4, 2, bias=True)
    x = torch.randn(5, 4)
    x_hat, z = sae(x)

    assert z.shape == (5, 2)
    assert x_hat.shape == (5, 4)
    assert calls[0] == ("linear", (5, 4), tuple(sae.encoder.weight.shape), True)
    assert calls[1] == ("activation", "relu", (5, 2))
    assert calls[2] == ("linear", (5, 2), tuple(sae.decoder.weight.shape), True)
