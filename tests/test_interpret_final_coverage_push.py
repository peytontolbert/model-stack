from __future__ import annotations

from contextlib import contextmanager
import types

import pytest
import torch
import torch.nn as nn

import interpret.attribution.direct as direct_module
import interpret.attn.weights as weights_module
import interpret.model_adapter as model_adapter_module
from interpret.attribution.direct import (
    _final_residual_module,
    component_logit_attribution,
    head_logit_attribution,
    mlp_neuron_logit_attribution,
    sae_feature_logit_attribution,
)
from interpret.attribution.grad_x_input import grad_x_input_tokens
from interpret.attribution.integrated_gradients import integrated_gradients_tokens, predict_argmax
from interpret.attribution.occlusion import _occlude_positions_transform, token_occlusion_importance
from interpret.causal.slice_patching import output_patching_slice
from interpret.causal.steer import steer_residual
from interpret.activation_cache import ActivationCache, CaptureSpec
from interpret.features.sae import SAEConfig, fit_sae
import importlib

logit_lens_module = importlib.import_module("interpret.logit_lens")
from interpret.logit_lens import _get_lm_proj_weight, logit_lens
from interpret.model_adapter import AttentionSnapshot, MLPSnapshot, ModelInputs, eager_attention_forward
from interpret.neuron.ablate import _wrap_mlp_forward_zero_channels
from interpret.neuron.mlp_lens import mlp_lens as neuron_mlp_lens
from interpret.probes.dataset import _normalize_feature, ProbeFeatureSlice, build_probe_dataset
from interpret.search.greedy import greedy_head_recovery
from interpret.tracer import ActivationTracer
from runtime.attention_modules import EagerAttention
from runtime.causal import CausalLM
from runtime.seq2seq import EncoderDecoderLM
from specs.config import ModelConfig
from tensor.mlp import MLP


def _causal_model() -> CausalLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = CausalLM(cfg)
    model.eval()
    return model


def _seq2seq_model() -> EncoderDecoderLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = EncoderDecoderLM(cfg)
    model.eval()
    return model


class _BadHandle:
    def remove(self) -> None:
        raise RuntimeError("remove failed")


class _TupleBlock(nn.Module):
    def forward(self, x: torch.Tensor):
        return (x,)


class _LegacyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = MLP(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _LegacyCausal(nn.Module):
    def __init__(self, with_blocks: bool = True) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([_LegacyBlock()]) if with_blocks else nn.ModuleList([])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return x


class _AppendReadTorchCache:
    def append_and_read(self, k: torch.Tensor, v: torch.Tensor, layer: int):
        return k, v


@contextmanager
def _noop_ctx(value=None, *args, **kwargs):
    yield value


def test_attention_weight_and_model_adapter_last_backend_branches(monkeypatch) -> None:
    monkeypatch.setattr(weights_module, "patched_attention", lambda *args, **kwargs: _noop_ctx())
    with pytest.raises(RuntimeError):
        weights_module.attention_snapshot_for_layer(_causal_model(), torch.randint(0, 8, (1, 4)), layer_index=0)

    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32, vocab_size=32, attn_impl="eager")
    attn = EagerAttention(cfg, n_kv_heads=2, use_rope=False, backend_override="torch")
    monkeypatch.setattr(model_adapter_module, "scaled_dot_product_attention", lambda q, k, v, attn_mask=None, dropout_p=0.0, backend="torch", is_causal=False, scale=1.0: q)
    out = eager_attention_forward(attn, torch.randn(1, 2, cfg.d_model), None, None, torch.zeros(1, 1, 2, 2, dtype=torch.float64), _AppendReadTorchCache())
    assert out.shape == (1, 2, cfg.d_model)


def test_attribution_restore_and_error_branches(monkeypatch) -> None:
    with pytest.raises(RuntimeError):
        _final_residual_module(model_adapter_module.ModelAdapter(_LegacyCausal(with_blocks=False)), stack="causal")

    seq2seq = _seq2seq_model()
    enc = torch.randint(0, seq2seq.cfg.vocab_size, (1, 4))
    dec = torch.randint(0, seq2seq.cfg.vocab_size, (1, 3))
    seq2seq.train()
    scores = component_logit_attribution(seq2seq, None, enc_input_ids=enc, dec_input_ids=dec, stack="decoder")
    assert any(key.endswith("cross_block.attn") or key.endswith("cross_block.mlp") for key in scores)
    assert seq2seq.training

    @contextmanager
    def _bad_attention(*args, **kwargs):
        kwargs["capture"](AttentionSnapshot(head_out=None))
        yield

    monkeypatch.setattr(direct_module, "patched_attention", _bad_attention)
    with pytest.raises(RuntimeError):
        head_logit_attribution(_causal_model(), torch.randint(0, 8, (1, 4)), layer_index=0)

    with pytest.raises(ValueError):
        mlp_neuron_logit_attribution(seq2seq, None, enc_input_ids=enc, dec_input_ids=dec, layer_index=0, stack="decoder")

    @contextmanager
    def _bad_mlp(*args, **kwargs):
        kwargs["capture"](MLPSnapshot(mlp_mid=None))
        yield

    monkeypatch.setattr(direct_module, "patched_mlp", _bad_mlp)
    with pytest.raises(RuntimeError):
        mlp_neuron_logit_attribution(_causal_model(), torch.randint(0, 8, (1, 4)), layer_index=0)

    with pytest.raises(ValueError):
        sae_feature_logit_attribution(seq2seq, None, enc_input_ids=enc, dec_input_ids=dec, layer_index=0, sae=MLP(16, 4), stack="decoder")


def test_grad_integrated_occlusion_and_logit_projection_branches(monkeypatch) -> None:
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 4))
    model.train()
    grad_scores = grad_x_input_tokens(model, input_ids)
    assert grad_scores.shape == (input_ids.shape[1],)
    assert model.training

    assert isinstance(predict_argmax(model, input_ids), int)
    model.train()
    ig_scores = integrated_gradients_tokens(model, input_ids, steps=3)
    assert ig_scores.shape == (input_ids.shape[1],)
    assert model.training

    transform = _occlude_positions_transform([-1, 99])
    assert torch.equal(transform(torch.ones(1)), torch.ones(1))
    model.train()
    occ_scores = token_occlusion_importance(model, input_ids, positions=[-1])
    assert occ_scores.shape == (input_ids.shape[1],)
    assert model.training

    class _NoWeightEmbedCausal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Identity()
            self.blocks = nn.ModuleList([nn.Identity()])

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            return torch.ones(input_ids.shape[0], input_ids.shape[1], 4)

    with pytest.raises(AttributeError):
        _get_lm_proj_weight(_NoWeightEmbedCausal())

    monkeypatch.setattr(logit_lens_module.ActivationTracer, "trace", lambda self: _noop_ctx(self._cache))
    monkeypatch.setattr(logit_lens_module.ActivationTracer, "add_residual_streams", lambda self, stack=None: [])
    import interpret.neuron.mlp_lens as neuron_mlp_lens_module
    monkeypatch.setattr(neuron_mlp_lens_module.ActivationTracer, "trace", lambda self: _noop_ctx(self._cache))
    monkeypatch.setattr(neuron_mlp_lens_module.ActivationTracer, "add_mlp_surfaces", lambda self, stack=None, kind=None, include=None: [])
    assert logit_lens(model, input_ids, layer_ids=[0]) == {}
    assert neuron_mlp_lens(model, input_ids, layer_ids=[0]) == {}


def test_slice_steer_probe_and_tracer_cleanup_branches(monkeypatch) -> None:
    model = nn.Sequential(nn.Linear(2, 2))
    monkeypatch.setattr(model[0], "register_forward_hook", lambda hook: _BadHandle())
    with output_patching_slice(model, {"0": torch.ones(1, 2)}, time_slice=slice(0, 1)):
        _ = model(torch.zeros(1, 2))

    legacy = _LegacyCausal()
    legacy.blocks[0] = _TupleBlock()  # type: ignore[index]
    with steer_residual(legacy, {0: torch.ones(4)}):
        out = legacy(torch.randint(0, 8, (1, 3)))
    assert isinstance(out, tuple)

    block = _LegacyBlock()
    monkeypatch.setattr(block, "register_forward_hook", lambda hook: _BadHandle())
    fake = _LegacyCausal()
    fake.blocks[0] = block  # type: ignore[index]
    with steer_residual(fake, {0: torch.ones(1, 1, 4)}):
        _ = fake(torch.randint(0, 8, (1, 3)))

    with pytest.raises(ValueError):
        _normalize_feature(torch.ones(1))
    cache = ActivationCache()
    cache.store("a", torch.ones(2, 4, 3), CaptureSpec(move_to_cpu=False))
    cache.store("b", torch.ones(2, 3, 2), CaptureSpec(move_to_cpu=False))
    with pytest.raises(ValueError):
        build_probe_dataset(cache, [ProbeFeatureSlice("a"), ProbeFeatureSlice("b")], input_ids=torch.arange(8).view(2, 4))

    tracer = ActivationTracer(_LegacyCausal())
    assert tracer.add_block_residual_streams() == ["blocks.0.attn", "blocks.0.mlp"]
    assert tracer.add_block_outputs() == ["blocks.0"]
    assert ActivationTracer(_seq2seq_model()).add_embedding_output(stack="encoder") == "encoder.embed"


def test_sae_and_ablate_remaining_branches(monkeypatch) -> None:
    features = torch.zeros(8, 4)
    sae, info = fit_sae(features, cfg=SAEConfig(code_dim=2, lr=0.0, epochs=3, batch_size=4, patience=1, device="cpu"))
    assert info["loss"] >= 0.0
    assert sae.encoder.weight.shape[0] == 2

    x = torch.randn(1, 2, 8)
    for activation in ("swiglu", "geglu", "reglu"):
        mlp = MLP(8, 4, activation=activation)
        _, wrapped = _wrap_mlp_forward_zero_channels(mlp, [0, 1])
        assert wrapped(x).shape == (1, 2, 8)


def test_greedy_non_eager_and_empty_selection_branch() -> None:
    model = _LegacyCausal()
    clean = torch.randint(0, 8, (1, 4))
    corrupted = clean.clone()
    corrupted[0, 0] = (corrupted[0, 0] + 1) % 8
    result = greedy_head_recovery(model, clean_input_ids=clean, corrupted_input_ids=corrupted, k=1, score_fn=lambda outputs: outputs[0, -1, 0])
    assert result["selected"] == []
    assert result["curve"] == []
