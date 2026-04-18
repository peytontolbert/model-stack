from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

import interpret.model_adapter as model_adapter_module
from interpret.activation_cache import ActivationCache, CaptureSpec
from interpret.attn import weights as weights_module
from interpret.attn.saliency import head_grad_saliencies
from interpret.causal.head_patching import (
    _wrap_forward_capture_heads,
    _wrap_forward_patch_heads,
    causal_trace_heads_restore_table,
)
from interpret.features.mining import (
    _get_feature_tensor,
    _topk_flat,
    activation_contexts,
    feature_coactivation_matrix,
    sae_feature_dashboard,
    topk_feature_positions,
    topk_positions,
)
from interpret.importance.module_scan import module_importance_scan
from interpret.logit_diff import logit_diff_lens
from interpret.logit_lens import _get_lm_proj_weight, _get_norm_fn, logit_lens
from interpret.model_adapter import AttentionSnapshot, ModelAdapter, ModelInputs, eager_attention_forward, mlp_forward
from interpret.neuron.mlp_lens import mlp_lens as neuron_mlp_lens
from interpret.probes.linear import LinearProbe, LinearProbeConfig, evaluate, fit_linear_probe
from interpret.tracer import ActivationTracer
from runtime.attention_modules import EagerAttention
from runtime.causal import CausalLM
from runtime.encoder import EncoderModel
from runtime.seq2seq import EncoderDecoderLM
from specs.config import ModelConfig
from tensor.mlp import MLP


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


class _NoInputModule(nn.Module):
    def forward(self) -> torch.Tensor:
        return torch.ones(1)


class _NoNormCausal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([nn.Identity()])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return x


class _NoEmbedModel(nn.Module):
    pass


class _NonEagerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = MLP(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _NonEagerCausal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([_NonEagerBlock()])
        self.norm = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return x


class _AppendReadCache:
    def append_and_read(self, k: torch.Tensor, v: torch.Tensor, layer: int):
        return k, v


class _EmptyReadCache:
    def __init__(self) -> None:
        self.appended: tuple[torch.Tensor, torch.Tensor] | None = None

    def read(self, start: int, length: int):
        return None, None

    def length(self) -> int:
        return 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.appended = (k.clone(), v.clone())


def test_activation_cache_and_tracer_edge_branches(monkeypatch) -> None:
    cache = ActivationCache()
    x = torch.randn(2, 3, requires_grad=True)
    cache.store("keep", x, CaptureSpec(move_to_cpu=False, keep_grad=True, clone=False))
    assert cache.get("keep") is x
    assert cache.get("keep").requires_grad

    if torch.cuda.is_available():
        gpu_x = x.to("cuda")
        cache.store("gpu", gpu_x, CaptureSpec(move_to_cpu=True, detach=False))
        assert cache.get("gpu").device.type == "cpu"

    tracer = ActivationTracer(_NoInputModule())
    tracer.add_modules(["missing"])
    captured = tracer.add_modules_matching(lambda name, module: (_ for _ in ()).throw(RuntimeError("boom")))
    assert captured == []
    tracer._register_named_module_pre("no_input", tracer.model, key="pre.no_input")
    with tracer.trace() as activation_cache:
        _ = tracer.model()
    assert activation_cache.get("pre.no_input") is None

    tracer._handles.append(types.SimpleNamespace(remove=lambda: (_ for _ in ()).throw(RuntimeError("remove failed"))))
    tracer.close()
    assert tracer._handles == []


def test_attention_weight_and_saliency_error_branches(monkeypatch) -> None:
    class _BadAdapter:
        kind = "causal"

        def attention_target(self, *args, **kwargs):
            return types.SimpleNamespace(module=object())

    monkeypatch.setattr(weights_module, "get_model_adapter", lambda model: _BadAdapter())
    monkeypatch.setattr(weights_module, "coerce_model_inputs", lambda *args, **kwargs: ModelInputs.causal(torch.tensor([[1]])))
    with pytest.raises(TypeError):
        weights_module.attention_snapshot_for_layer(object(), torch.tensor([[1]]), layer_index=0)

    monkeypatch.setattr(weights_module, "attention_snapshot_for_layer", lambda *args, **kwargs: AttentionSnapshot(probs=None))
    with pytest.raises(RuntimeError):
        weights_module.attention_weights_for_layer(object(), torch.tensor([[1]]), layer_index=0)

    encoder = _encoder_model()
    input_ids = torch.randint(0, encoder.cfg.vocab_size, (1, 4))
    encoder.train()
    sal = head_grad_saliencies(encoder, input_ids)
    assert sal.shape == (encoder.cfg.n_layers, encoder.cfg.n_heads)
    assert encoder.training


def test_logit_lens_diff_and_mlp_lens_remaining_branches() -> None:
    no_norm = _NoNormCausal()
    norm_fn = _get_norm_fn(no_norm)
    value = torch.randn(1, 2, 4)
    assert torch.equal(norm_fn(value), value)
    assert _get_lm_proj_weight(no_norm).shape == no_norm.embed.weight.shape
    with pytest.raises(TypeError):
        _get_lm_proj_weight(_NoEmbedModel())

    seq2seq = _seq2seq_model()
    enc = torch.randint(0, seq2seq.cfg.vocab_size, (1, 4))
    dec = torch.randint(0, seq2seq.cfg.vocab_size, (1, 3))
    seq2seq.train()
    assert logit_lens(seq2seq, None, enc_input_ids=enc, dec_input_ids=dec, stack="decoder", layer_ids=[99]) == {}
    assert logit_diff_lens(seq2seq, None, 0, 1, enc_input_ids=enc, dec_input_ids=dec, stack="decoder", layer_ids=[99]) == {}
    assert neuron_mlp_lens(seq2seq, None, enc_input_ids=enc, dec_input_ids=dec, stack="decoder", layer_ids=[99]) == {}
    assert seq2seq.training


def test_feature_mining_module_scan_and_probe_branches(monkeypatch) -> None:
    cache = ActivationCache()
    feats = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    cache.store("feat", feats, CaptureSpec(move_to_cpu=False))
    assert _get_feature_tensor(cache, "feat").shape == feats.shape
    with pytest.raises(KeyError):
        _get_feature_tensor(cache, "missing")
    cache.store("flat", torch.arange(8, dtype=torch.float32).view(2, 4), CaptureSpec(move_to_cpu=False))
    with pytest.raises(ValueError):
        _get_feature_tensor(cache, "flat")
    assert _topk_flat(torch.ones(2, 2), 0) == []
    assert topk_positions(cache, "feat", k=1)
    assert topk_feature_positions(cache, "feat", 0, k=1)
    assert activation_contexts(torch.arange(6).view(2, 3), [(99, 0, 1.0)], window=1) == []
    with pytest.raises(ValueError):
        activation_contexts(torch.arange(3), [(0, 0, 1.0)])

    empty_cache = ActivationCache()
    empty_cache.store("feat", torch.empty(0, 0, 3), CaptureSpec(move_to_cpu=False))
    assert feature_coactivation_matrix(empty_cache, "feat").shape == (0, 0)

    class _BadSAE:
        pass

    monkeypatch.setattr("interpret.features.mining.sae_encode", lambda sae, x: torch.ones(2, 3))
    with pytest.raises(ValueError):
        sae_feature_dashboard(cache, "feat", _BadSAE(), 0, input_ids=torch.arange(6).view(2, 3))

    encoder = _encoder_model()
    input_ids = torch.randint(0, encoder.cfg.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        module_importance_scan(encoder, input_ids, mode="prob")
    with pytest.raises(ValueError):
        module_importance_scan(encoder, input_ids, mode="nll")
    with pytest.raises(ValueError):
        module_importance_scan(_causal_model(), torch.randint(0, 8, (1, 4)), mode="bad")
    assert module_importance_scan(_causal_model(), torch.randint(0, 8, (1, 4)), modules=["missing"]) == []

    probe = LinearProbe(3, 1)
    mse = evaluate(probe, torch.randn(4, 3), torch.randn(4), task="regression")
    assert mse >= 0.0

    x_train = torch.randn(32, 4)
    y_train = (x_train[:, 0] > 0).long()
    x_val = torch.randn(16, 4)
    y_val = (x_val[:, 0] > 0).long()
    _, score = fit_linear_probe(
        x_train,
        y_train,
        x_val,
        y_val,
        cfg=LinearProbeConfig(task="classification", lr=0.01, epochs=4, batch_size=8, patience=1, device="cpu"),
    )
    assert 0.0 <= score <= 1.0


def test_head_patching_and_model_adapter_remaining_branches(monkeypatch) -> None:
    adapter = ModelAdapter(_seq2seq_model())
    with pytest.raises(ValueError):
        adapter.embedding_tokens(ModelInputs(), stack="decoder")

    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32, vocab_size=32, attn_impl="eager")
    attn = EagerAttention(cfg)
    x = torch.randn(1, 2, cfg.d_model)

    sink: dict[int, torch.Tensor] = {}
    orig_forward, wrapped_forward = _wrap_forward_capture_heads(attn, sink, 0)
    attn.forward = wrapped_forward  # type: ignore[assignment]
    try:
        _ = attn(x, None, None, None)
    finally:
        attn.forward = orig_forward  # type: ignore[assignment]
    assert sink[0].device.type == "cpu"

    import interpret.causal.head_patching as head_patching_module

    original_eager_attention_forward = head_patching_module.eager_attention_forward
    monkeypatch.setattr(
        head_patching_module,
        "eager_attention_forward",
        lambda *args, **kwargs: kwargs["capture"](AttentionSnapshot(head_out=None)) or args[1],
    )
    sink_none: dict[int, torch.Tensor] = {}
    _, wrapped_no_head = _wrap_forward_capture_heads(attn, sink_none, 0)
    _ = wrapped_no_head(x, None, None, None)
    assert sink_none == {}
    monkeypatch.setattr(head_patching_module, "eager_attention_forward", original_eager_attention_forward)

    source: dict[int, torch.Tensor] = {}
    _, wrapped_missing = _wrap_forward_patch_heads(attn, source, 0, heads=[0])
    assert wrapped_missing(x, None, None, None).shape == x.shape

    source[0] = torch.zeros(1, attn.n_heads, x.shape[1], attn.head_dim)
    _, wrapped_invalid = _wrap_forward_patch_heads(attn, source, 0, heads=[99], time_index=0)
    assert wrapped_invalid(x, None, None, None).shape == x.shape

    non_eager = _NonEagerCausal()
    clean = torch.randint(0, 8, (1, 4))
    corrupted = clean.clone()
    corrupted[0, 0] = (corrupted[0, 0] + 1) % 8
    non_eager.train()
    table = causal_trace_heads_restore_table(
        non_eager,
        clean_input_ids=clean,
        corrupted_input_ids=corrupted,
        score_fn=lambda outputs: outputs[0, -1, 0],
    )
    assert table.shape == (1, 0)
    assert non_eager.training



def test_eager_attention_backend_fallback_and_mlp_geglu(monkeypatch) -> None:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32, vocab_size=32, attn_impl="eager")
    attn = EagerAttention(cfg, n_kv_heads=2, use_rope=False)
    attn.train()
    attn.attn_dropout_p = 0.1
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    monkeypatch.setattr(model_adapter_module, "_read_backend_from_env_or_file", lambda: None)
    monkeypatch.setattr(model_adapter_module, "select_attention_backend", lambda **kwargs: "torch")
    monkeypatch.setattr(model_adapter_module, "scaled_dot_product_attention", lambda q, k, v, attn_mask=None, dropout_p=0.0, backend="torch", is_causal=False, scale=1.0: q)
    out = eager_attention_forward(attn, torch.randn(1, 2, cfg.d_model), None, None, mask, _AppendReadCache())
    assert out.shape == (1, 2, cfg.d_model)

    empty_cache = _EmptyReadCache()
    out_empty = eager_attention_forward(EagerAttention(cfg, n_kv_heads=2, use_rope=False), torch.randn(1, 2, cfg.d_model), None, None, None, empty_cache)
    assert out_empty.shape == (1, 2, cfg.d_model)
    assert empty_cache.appended is not None

    x = torch.randn(1, 3, 8, requires_grad=True)
    geglu = MLP(8, 4, activation="geglu")
    out = mlp_forward(geglu, x, keep_grad=True)
    assert out.shape == (1, 3, 8)
