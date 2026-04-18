from __future__ import annotations

from contextlib import contextmanager
import builtins
import importlib
import sys
import types

import pytest
import torch
import torch.nn as nn

from interpret.activation_cache import ActivationCache, CaptureSpec
from interpret.analysis.mask_effects import logit_change_with_mask
from interpret.analysis.residual import residual_norms
from interpret.attn.ablate import ablate_attention_heads
from interpret.attn.rollout import attention_rollout
from interpret.attn.saliency import head_grad_saliencies
from interpret.attribution.direct import (
    head_logit_attribution,
    mlp_neuron_logit_attribution,
    sae_feature_logit_attribution,
)
from interpret.attribution.integrated_gradients import integrated_gradients_tokens
from interpret.attribution.occlusion import token_occlusion_importance
from interpret.causal.head_patching import _wrap_forward_capture_heads, causal_trace_heads_restore_table
from interpret.causal.slice_patching import output_patching_slice
from interpret.causal.steer import steer_residual
from interpret.causal.sweeps import (
    block_output_patch_sweep,
    head_patch_sweep,
    mlp_neuron_patch_sweep,
    path_patch_effect,
)
from interpret.features.sae import SparseAutoencoder
from interpret.features.stats import channel_stats
from interpret.importance.module_scan import module_importance_scan
from interpret.logit_diff import logit_diff_lens
from interpret.model_adapter import ModelInputs, eager_attention_forward
from interpret.neuron.ablate import _wrap_mlp_forward_zero_channels
from interpret.neuron.mlp_lens import mlp_lens
from interpret.probes.dataset import ProbeFeatureSlice, _normalize_feature, build_probe_dataset
from interpret.search.greedy import _evaluate_selected_heads, greedy_head_recovery
from runtime.attention_modules import EagerAttention
from runtime.causal import CausalLM
from runtime.encoder import EncoderModel
from runtime.seq2seq import EncoderDecoderLM
from specs.config import ModelConfig
from tensor.mlp import MLP

import interpret.analysis.mask_effects as mask_effects_module
import interpret.analysis.residual as residual_module
import interpret.attn.rollout as rollout_module
import interpret.attn.saliency as saliency_module
import interpret.attribution.occlusion as occlusion_module
import interpret.causal.head_patching as head_patching_module
import interpret.causal.sweeps as sweeps_module
import interpret.logit_diff as logit_diff_module
import interpret.model_adapter as model_adapter_module
import interpret.neuron.mlp_lens as neuron_mlp_lens_module
import interpret.tracer as tracer_module


def _causal_model() -> CausalLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = CausalLM(cfg)
    model.eval()
    return model


def _encoder_model() -> EncoderModel:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = EncoderModel(cfg)
    model.eval()
    return model


def _seq2seq_model() -> EncoderDecoderLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = EncoderDecoderLM(cfg)
    model.eval()
    return model


class _TupleModule(nn.Module):
    def forward(self, x: torch.Tensor):
        return (x,)


class _LegacyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _LegacyCausal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([_LegacyBlock()])
        self.norm = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return x


class _NoWeightEmbedCausal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Identity()
        self.blocks = nn.ModuleList([_LegacyBlock()])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return torch.ones(input_ids.shape[0], input_ids.shape[1], 4)


class _FakeGPUHeadOut:
    device = types.SimpleNamespace(type="cuda")

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return torch.ones(1, 4, 2, 4)


class _DummyCache:
    def __init__(self, values: dict[str, torch.Tensor | None]) -> None:
        self.values = values

    def get(self, key: str):
        return self.values.get(key)


class _FakeTrace:
    def __init__(self, cache) -> None:
        self.cache = cache

    def __enter__(self):
        return self.cache

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTracer:
    def __init__(self, cache) -> None:
        self._cache = cache

    def add_residual_streams(self, *args, **kwargs):
        return []

    def trace(self):
        return _FakeTrace(self._cache)


class _SimpleSweepAdapter:
    kind = "causal"

    def __init__(self, modules: list[tuple[str, nn.Module]]) -> None:
        self._targets = [types.SimpleNamespace(name=name, module=mod, layer_index=i, kind=None) for i, (name, mod) in enumerate(modules)]

    def block_targets(self, stack=None):
        return self._targets

    def attention_targets(self, stack=None, kind=None):
        return self._targets

    def mlp_module(self, layer_idx, stack=None, kind=None):
        raise RuntimeError("no mlp")

    def named_modules(self):
        return {name: mod for name, mod in [(target.name, target.module) for target in self._targets]}

    def forward(self, inputs):
        x = torch.ones(1, 2, 4)
        for target in self._targets:
            _ = target.module(x)
        return torch.randn(1, 2, 8)


class _FakeMaskAdapter:
    kind = "encoder_decoder"

    def sequence_tokens(self, inputs, stack=None):
        return None


class _FakeOcclusionAdapter:
    kind = "causal"

    def forward(self, inputs):
        return torch.zeros(1, 2, 4)

    def output_module(self):
        return None

    def sequence_tokens(self, inputs, stack=None):
        return None


class _FakeProbeTarget:
    def __init__(self) -> None:
        self.name = "blocks.0.mlp"
        self.layer_index = 0
        self.kind = None


class _FakeProbeAdapter:
    kind = "causal"

    def forward(self, inputs):
        return torch.zeros(1, 2, 4)

    def mlp_targets(self, stack=None, kind=None):
        return [_FakeProbeTarget()]

    def embedding_module(self, stack=None):
        return nn.Identity()


@contextmanager
def _noop_ctx(*args, **kwargs):
    yield


def test_mask_effects_residual_rollout_and_occlusion_remaining_branches(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(mask_effects_module, "get_model_adapter", lambda model: _FakeMaskAdapter())
        m.setattr(mask_effects_module, "coerce_model_inputs", lambda *args, **kwargs: ModelInputs())
        with pytest.raises(ValueError):
            mask_effects_module.logit_change_with_mask(object(), None, stack="decoder")

    causal = _causal_model()
    causal.train()
    input_ids = torch.randint(0, causal.cfg.vocab_size, (1, 4))
    assert isinstance(logit_change_with_mask(causal, input_ids), float)
    assert causal.training

    encoder = _encoder_model()
    encoder.train()
    enc_ids = torch.randint(0, encoder.cfg.vocab_size, (1, 4))
    out = residual_norms(encoder, enc_ids, attn_mask=torch.ones(1, 4, dtype=torch.long))
    assert out["pre"].shape[0] == encoder.cfg.n_layers
    assert encoder.training

    seq2seq = _seq2seq_model()
    enc = torch.randint(0, seq2seq.cfg.vocab_size, (1, 4))
    dec = torch.randint(0, seq2seq.cfg.vocab_size, (1, 3))
    enc_resid = residual_norms(seq2seq, None, enc_input_ids=enc, dec_input_ids=dec, stack="encoder", enc_padding_mask=torch.ones(1, 4, dtype=torch.long))
    assert enc_resid["pre"].shape[0] == seq2seq.cfg.n_layers

    causal_for_cache = _causal_model()
    adapter = residual_module.get_model_adapter(causal_for_cache)
    d_model = causal_for_cache.cfg.d_model
    values = {}
    for target in adapter.block_targets(stack="causal"):
        if target.layer_index == 0:
            values[f"{target.name}.resid_pre"] = None
            values[f"{target.name}.resid_post"] = None
        else:
            values[f"{target.name}.resid_pre"] = torch.ones(1, 4, d_model)
            values[f"{target.name}.resid_post"] = torch.ones(1, 4, d_model)
    with monkeypatch.context() as m:
        m.setattr(residual_module, "ActivationTracer", lambda model, spec=None: _FakeTracer(_DummyCache(values)))
        reduced = residual_module.residual_norms(_causal_model(), torch.randint(0, 8, (1, 4)))
    assert reduced["pre"].shape[0] == 1

    with monkeypatch.context() as m:
        m.setattr(rollout_module, "get_model_adapter", lambda model: _FakeMaskAdapter())
        m.setattr(rollout_module, "coerce_model_inputs", lambda *args, **kwargs: ModelInputs())
        with pytest.raises(ValueError):
            rollout_module.attention_rollout(object(), None)

    rollout = attention_rollout(_causal_model(), torch.randint(0, 8, (1, 4)), head_agg="max")
    assert rollout.shape == (1, 4, 4)

    with monkeypatch.context() as m:
        m.setattr(rollout_module, "attention_weights_for_layer", lambda *args, **kwargs: torch.ones(1, 2, 4, 3))
        with pytest.raises(ValueError):
            attention_rollout(_causal_model(), torch.randint(0, 8, (1, 4)))

    encoder = _encoder_model()
    ig = integrated_gradients_tokens(encoder, torch.randint(0, encoder.cfg.vocab_size, (1, 4)), steps=2)
    assert ig.shape == (4,)

    occ = token_occlusion_importance(encoder, torch.randint(0, encoder.cfg.vocab_size, (1, 4)))
    assert occ.shape == (4,)
    with pytest.raises(ValueError):
        token_occlusion_importance(encoder, torch.randint(0, encoder.cfg.vocab_size, (1, 4)), mode="prob")

    with monkeypatch.context() as m:
        m.setattr(occlusion_module, "get_model_adapter", lambda model: _FakeOcclusionAdapter())
        m.setattr(occlusion_module, "coerce_model_inputs", lambda *args, **kwargs: ModelInputs())
        m.setattr(occlusion_module, "resolve_model_score", lambda *args, **kwargs: (torch.tensor(0.0), None, 0))
        with pytest.raises(ValueError):
            occlusion_module.token_occlusion_importance(nn.Identity(), None)


def test_direct_attr_training_restores_and_sae_mlp_branch() -> None:
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 4))

    model.train()
    head_scores = head_logit_attribution(model, input_ids, layer_index=0)
    assert head_scores.shape[0] == model.cfg.n_heads
    assert model.training

    model.train()
    neuron_scores = mlp_neuron_logit_attribution(model, input_ids, layer_index=0)
    assert neuron_scores.ndim == 1
    assert model.training

    sae = SparseAutoencoder(model.cfg.d_model, 4)
    model.train()
    feature_scores = sae_feature_logit_attribution(model, input_ids, layer_index=0, sae=sae, module="mlp")
    assert feature_scores.shape[0] == 4
    assert model.training


def test_saliency_head_patching_slice_and_steer_remaining_branches(monkeypatch) -> None:
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 4))

    def _fake_capture(attn, sink, layer_idx, detach=False, move_to_cpu=False):
        orig = attn.forward

        def new(q, k, v, mask, cache=None, *, position_embeddings=None, position_ids=None):
            sink[layer_idx] = torch.ones(1, attn.n_heads, q.shape[1], attn.head_dim)
            return orig(q, k, v, mask, cache, position_embeddings=position_embeddings, position_ids=position_ids)

        return orig, new

    with monkeypatch.context() as m:
        m.setattr(head_patching_module, "_wrap_forward_capture_heads", _fake_capture)
        sal = saliency_module.head_grad_saliencies(model, input_ids)
    assert sal.shape == (model.cfg.n_layers, model.cfg.n_heads)

    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32, vocab_size=32, attn_impl="eager")
    attn = EagerAttention(cfg, use_rope=False)
    sink: dict[int, torch.Tensor] = {}
    orig, wrapped = _wrap_forward_capture_heads(attn, sink, 0, detach=True, move_to_cpu=True)

    with monkeypatch.context() as m:
        m.setattr(
            head_patching_module,
            "eager_attention_forward",
            lambda *args, **kwargs: kwargs["capture"](model_adapter_module.AttentionSnapshot(head_out=_FakeGPUHeadOut())) or args[1],
        )
        _ = wrapped(torch.randn(1, 2, cfg.d_model), None, None, None)
    attn.forward = orig  # type: ignore[assignment]
    assert sink[0].device.type == "cpu"

    def _no_store_capture(attn, sink, layer_idx, **kwargs):
        orig = attn.forward

        def new(q, k, v, mask, cache=None, *, position_embeddings=None, position_ids=None):
            return orig(q, k, v, mask, cache, position_embeddings=position_embeddings, position_ids=position_ids)

        return orig, new

    with monkeypatch.context() as m:
        m.setattr(head_patching_module, "_wrap_forward_capture_heads", _no_store_capture)
        table = causal_trace_heads_restore_table(_causal_model(), clean_input_ids=input_ids, corrupted_input_ids=input_ids.roll(1, 1))
    assert table.shape[0] == model.cfg.n_layers

    tuple_model = nn.Sequential(_TupleModule())
    with output_patching_slice(tuple_model, {"0": torch.ones(1, 1, 2), "missing": torch.ones(1, 1, 2)}, time_slice=slice(0, 1)):
        out = tuple_model(torch.ones(1, 1, 2))
    assert isinstance(out, tuple)

    legacy = _LegacyCausal()
    with steer_residual(legacy, {0: torch.ones(1, 3, 4)}):
        steered = legacy(torch.randint(0, 8, (1, 3)))
    assert isinstance(steered, torch.Tensor)


def test_module_scan_logit_diff_dataset_stats_mlp_lens_and_ablate_remaining(monkeypatch) -> None:
    seq2seq = _seq2seq_model()
    enc = torch.randint(0, seq2seq.cfg.vocab_size, (1, 4))
    dec = torch.randint(0, seq2seq.cfg.vocab_size, (1, 3))
    seq2seq.train()
    scan = module_importance_scan(seq2seq, None, enc_input_ids=enc, dec_input_ids=dec, stack="decoder")
    assert scan
    assert seq2seq.training

    with pytest.raises(AttributeError):
        logit_diff_lens(_NoWeightEmbedCausal(), torch.randint(0, 8, (1, 4)), 0, 1)

    with monkeypatch.context() as m:
        m.setattr(logit_diff_module.ActivationTracer, "trace", lambda self: _FakeTrace(ActivationCache()))
        m.setattr(logit_diff_module.ActivationTracer, "add_residual_streams", lambda self, stack=None: [])
        assert logit_diff_lens(_causal_model(), torch.randint(0, 8, (1, 4)), 0, 1, layer_ids=[0]) == {}

    cache = ActivationCache()
    cache.store("two_d", torch.arange(6, dtype=torch.float32).view(2, 3), CaptureSpec(move_to_cpu=False))
    stats = channel_stats(cache, "two_d")
    assert stats["mean"].shape[0] == 3

    assert _normalize_feature(torch.ones(2, 3)).shape == (2, 1, 3)
    cache = ActivationCache()
    cache.store("a", torch.ones(2, 4, 3), CaptureSpec(move_to_cpu=False))
    ds = build_probe_dataset(cache, [ProbeFeatureSlice("a")], targets=torch.arange(8).view(2, 4))
    assert ds.y.shape[0] == 8

    emb = nn.Embedding(8, 4)
    with monkeypatch.context() as m:
        m.setattr(neuron_mlp_lens_module, "get_model_adapter", lambda model: types.SimpleNamespace(kind="causal", forward=lambda inputs: torch.zeros(1, 2, 4), mlp_targets=lambda stack=None, kind=None: [_FakeProbeTarget()], embedding_module=lambda stack=None: emb))
        m.setattr(neuron_mlp_lens_module, "coerce_model_inputs", lambda *args, **kwargs: ModelInputs())
        m.setattr(neuron_mlp_lens_module, "ActivationTracer", lambda model: types.SimpleNamespace(add_mlp_surfaces=lambda **kwargs: [], trace=lambda: _FakeTrace(_DummyCache({"blocks.0.mlp.mlp_out": torch.ones(1, 2, 4)}))))
        lens = neuron_mlp_lens_module.mlp_lens(nn.Identity(), None, layer_ids=[0], topk=1)
    assert 0 in lens

    with monkeypatch.context() as m:
        m.setattr(neuron_mlp_lens_module, "get_model_adapter", lambda model: _FakeProbeAdapter())
        m.setattr(neuron_mlp_lens_module, "coerce_model_inputs", lambda *args, **kwargs: ModelInputs())
        m.setattr(neuron_mlp_lens_module, "ActivationTracer", lambda model: types.SimpleNamespace(add_mlp_surfaces=lambda **kwargs: [], trace=lambda: _FakeTrace(_DummyCache({"blocks.0.mlp.mlp_out": torch.ones(1, 2, 4)}))))
        with pytest.raises(AttributeError):
            neuron_mlp_lens_module.mlp_lens(nn.Identity(), None, layer_ids=[0], topk=1)

    x = torch.randn(1, 2, 8)
    weird = MLP(8, 4, activation="weird")
    _, wrapped_weird = _wrap_mlp_forward_zero_channels(weird, [0])
    assert wrapped_weird(x).shape == (1, 2, 8)
    plain = MLP(8, 4, activation="silu")
    _, wrapped_plain = _wrap_mlp_forward_zero_channels(plain, [0])
    assert wrapped_plain(x).shape == (1, 2, 8)

    seen: dict[str, torch.dtype] = {}
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32, vocab_size=32, attn_impl="eager")
    attn = EagerAttention(cfg, n_kv_heads=2, use_rope=False, backend_override="torch")
    with monkeypatch.context() as m:
        m.setattr(model_adapter_module, "runtime_prepare_attention_mask", lambda *args, **kwargs: torch.zeros(1, 1, 2, 2, dtype=torch.float64))

        def _fake_sdp(q, k, v, attn_mask=None, **kwargs):
            seen["mask_dtype"] = attn_mask.dtype
            return q

        m.setattr(model_adapter_module, "scaled_dot_product_attention", _fake_sdp)
        out = eager_attention_forward(attn, torch.randn(1, 2, cfg.d_model), None, None, torch.ones(1, 2, dtype=torch.long))
    assert out.shape == (1, 2, cfg.d_model)
    assert seen["mask_dtype"] == out.dtype


def test_sweeps_and_greedy_remaining_branches(monkeypatch) -> None:
    fake_adapter = _SimpleSweepAdapter([("mod", _TupleModule())])
    with monkeypatch.context() as m:
        m.setattr(sweeps_module, "get_model_adapter", lambda model: fake_adapter)
        with pytest.raises(ValueError):
            block_output_patch_sweep(object(), clean_inputs=ModelInputs(), corrupted_inputs=ModelInputs())
        causal_inputs = ModelInputs.causal(torch.tensor([[1, 2]]))
        block = block_output_patch_sweep(object(), clean_inputs=causal_inputs, corrupted_inputs=causal_inputs)
    assert block["scores"].shape == (1, 2)

    attn_adapter = _SimpleSweepAdapter([])
    with monkeypatch.context() as m:
        m.setattr(sweeps_module, "get_model_adapter", lambda model: attn_adapter)
        with pytest.raises(ValueError):
            head_patch_sweep(object(), clean_inputs=ModelInputs(), corrupted_inputs=ModelInputs())

    causal = _causal_model()

    def _no_store_capture(attn, sink, layer_idx, **kwargs):
        orig = attn.forward

        def new(q, k, v, mask, cache=None, *, position_embeddings=None, position_ids=None):
            return orig(q, k, v, mask, cache, position_embeddings=position_embeddings, position_ids=position_ids)

        return orig, new

    with monkeypatch.context() as m:
        m.setattr(sweeps_module, "_wrap_forward_capture_heads", _no_store_capture)
        head = head_patch_sweep(causal, clean_input_ids=torch.randint(0, 8, (1, 4)), corrupted_input_ids=torch.randint(0, 8, (1, 4)))
    assert head["scores"].shape[0] == causal.cfg.n_layers

    class _Seq2SeqAdapter:
        def __init__(self, base):
            self.base = base
            self.kind = base.kind

        def forward(self, inputs):
            return self.base.forward(inputs)

        def block_targets(self, stack=None):
            return self.base.block_targets(stack=stack)

        def mlp_module(self, layer_idx, stack=None, kind=None):
            if kind == "cross":
                raise RuntimeError("skip cross")
            return self.base.mlp_module(layer_idx, stack=stack, kind=kind)

    seq2seq = _seq2seq_model()
    seq_adapter = _Seq2SeqAdapter(sweeps_module.get_model_adapter(seq2seq))
    with monkeypatch.context() as m:
        m.setattr(sweeps_module, "get_model_adapter", lambda model: seq_adapter)
        m.setattr(sweeps_module, "patched_mlp", lambda *args, **kwargs: _noop_ctx())
        mlp = mlp_neuron_patch_sweep(
            seq2seq,
            clean_inputs=ModelInputs.encoder_decoder(torch.randint(0, 8, (1, 4)), torch.randint(0, 8, (1, 3))),
            corrupted_inputs=ModelInputs.encoder_decoder(torch.randint(0, 8, (1, 4)), torch.randint(0, 8, (1, 3))),
            neurons=[0],
        )
    assert mlp["scores"].shape[1] == 1

    with monkeypatch.context() as m:
        m.setattr(sweeps_module, "get_model_adapter", lambda model: seq_adapter)
        with pytest.raises(ValueError):
            mlp_neuron_patch_sweep(object(), clean_inputs=ModelInputs(), corrupted_inputs=ModelInputs(), neurons=[0])

    with pytest.raises(KeyError):
        path_patch_effect(_causal_model(), clean_input_ids=torch.randint(0, 8, (1, 4)), corrupted_input_ids=torch.randint(0, 8, (1, 4)), source_module="missing", receiver_module="also.missing")

    patched = path_patch_effect(
        _causal_model(),
        clean_input_ids=torch.randint(0, 8, (1, 4)),
        corrupted_input_ids=torch.randint(0, 8, (1, 4)),
        source_module="blocks.0.mlp",
        receiver_module="blocks.1.mlp",
        time_slice=slice(0, 1),
    )
    assert "receiver_restore_fraction" in patched

    legacy = _LegacyCausal()
    adapter = model_adapter_module.get_model_adapter(legacy)
    corrupted_inputs = ModelInputs.causal(torch.randint(0, 8, (1, 4)))
    targets = adapter.attention_targets()
    with torch.inference_mode():
        assert _evaluate_selected_heads(
            selected=[],
            targets=targets,
            clean_heads={},
            adapter=adapter,
            corrupted_inputs=corrupted_inputs,
            model=legacy,
            position=-1,
            target_token_id=None,
            target_feature_index=0,
            score_fn=None,
            base_score=torch.tensor(0.0),
            denom=torch.tensor(1.0),
        ) == 0.0
        assert isinstance(
            _evaluate_selected_heads(
                selected=[(99, 0)],
                targets=targets,
                clean_heads={},
                adapter=adapter,
                corrupted_inputs=corrupted_inputs,
                model=legacy,
                position=-1,
                target_token_id=None,
                target_feature_index=0,
                score_fn=None,
                base_score=torch.tensor(0.0),
                denom=torch.tensor(1.0),
            ),
            float,
        )
        assert isinstance(
            _evaluate_selected_heads(
                selected=[(0, 0)],
                targets=targets,
                clean_heads={},
                adapter=adapter,
                corrupted_inputs=corrupted_inputs,
                model=legacy,
                position=-1,
                target_token_id=None,
                target_feature_index=0,
                score_fn=None,
                base_score=torch.tensor(0.0),
                denom=torch.tensor(1.0),
            ),
            float,
        )



def test_attn_ablate_and_extra_mlp_sweep_branches(monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(
            "interpret.attn.ablate.get_model_adapter",
            lambda model: types.SimpleNamespace(attention_target=lambda *args, **kwargs: types.SimpleNamespace(module=nn.Identity())),
        )
        with ablate_attention_heads(nn.Identity(), {0: [0]}):
            pass

    class _FakeGatedMLP:
        def __init__(self) -> None:
            self.gated = True
            self.activation_name = "mystery"
            self.w_in = nn.Linear(8, 8)
            self.w_out = nn.Linear(4, 8)
            self.dropout = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    x = torch.randn(1, 2, 8)
    _, wrapped = _wrap_mlp_forward_zero_channels(_FakeGatedMLP(), [0])
    assert wrapped(x).shape == (1, 2, 8)

    class _FakeMLPSweepAdapter:
        kind = "causal"

        def forward(self, inputs):
            return torch.randn(1, 2, 8)

        def block_targets(self, stack=None):
            return []

        def mlp_module(self, layer_idx, stack=None, kind=None):
            raise RuntimeError("unused")

    with monkeypatch.context() as m:
        m.setattr(sweeps_module, "get_model_adapter", lambda model: _FakeMLPSweepAdapter())
        with pytest.raises(ValueError):
            mlp_neuron_patch_sweep(nn.Identity(), clean_inputs=ModelInputs(), corrupted_inputs=ModelInputs(), neurons=[0])

def test_tracer_import_fallback_branch(monkeypatch) -> None:
    orig_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "runtime.block_modules":
            raise ImportError("forced tracer import failure")
        return orig_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", _fake_import)
        assert tracer_module._load_transformer_block() is None

    assert tracer_module._load_transformer_block() is not None
