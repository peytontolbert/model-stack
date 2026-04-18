from __future__ import annotations

import torch
import pytest

from interpret import (
    ActivationCache,
    ActivationTracer,
    FeatureSlice,
    LinearProbeConfig,
    attention_rollout,
    channel_stats,
    component_logit_attribution,
    estimate_layer_costs,
    fit_linear_probe,
    flatten_features,
    logit_change_with_mask,
    logit_diff_lens,
    logit_lens,
    module_importance_scan,
    residual_norms,
    sequence_negative_log_likelihood,
    sequence_perplexity,
    token_occlusion_importance,
)
from interpret.activation_cache import CaptureSpec
from interpret.neuron.ablate import ablate_mlp_channels
from interpret.model_adapter import ModelAdapter, ModelInputs, coerce_model_inputs, resolve_model_score
from runtime.causal import CausalLM
from runtime.encoder import EncoderModel
from runtime.seq2seq import EncoderDecoderLM
from specs.config import ModelConfig


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


def test_activation_cache_full_surface_and_stats_errors(tmp_path) -> None:
    cache = ActivationCache()
    x = torch.arange(6, dtype=torch.float32).view(1, 2, 3)
    cache.store("x", x, CaptureSpec(move_to_cpu=False))
    assert "x" in cache
    assert list(cache.keys()) == ["x"]
    assert list(cache.items())[0][0] == "x"
    assert cache.pop("missing") is None
    assert cache.pop("x") is not None
    cache.store("x", x, CaptureSpec(move_to_cpu=False))
    pt = tmp_path / "cache.pt"
    cache.save_pt(str(pt))
    loaded = ActivationCache.load_pt(str(pt))
    assert torch.equal(loaded.get("x"), x)
    cache.clear()
    assert cache.get("x") is None

    bad = ActivationCache()
    bad.store("bad", torch.ones(2, 2, 2, 2), CaptureSpec(move_to_cpu=False))
    with pytest.raises(ValueError):
        channel_stats(bad, "bad")
    empty = channel_stats(ActivationCache(), "missing")
    assert empty["mean"].numel() == 0


def test_feature_helpers_and_probe_regression_branches() -> None:
    cache = ActivationCache()
    a = torch.ones(2, 3, 2)
    b = torch.ones(1, 3, 2)
    cache.store("a", a, CaptureSpec(move_to_cpu=False))
    cache.store("b", b, CaptureSpec(move_to_cpu=False))
    with pytest.raises(ValueError):
        flatten_features(cache, [FeatureSlice("a"), FeatureSlice("b")])
    assert flatten_features(cache, []).shape == (0, 0)

    x = torch.randn(16, 3)
    y = (x[:, 0] > 0).long()
    probe, score = fit_linear_probe(
        x,
        y,
        cfg=LinearProbeConfig(task="classification", lr=0.05, epochs=5, batch_size=8, patience=2, device="cpu"),
    )
    assert probe.linear.weight.shape[0] == 2
    assert 0.0 <= score <= 1.0


def test_interpret_helpers_cover_encoder_and_error_branches() -> None:
    model = _encoder_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 5))
    scores = component_logit_attribution(model, input_ids, stack="encoder", target_feature_index=0)
    assert "embed" in scores

    assert set(logit_lens(model, input_ids, stack="encoder", topk=2)) == {0, 1}
    diff = logit_diff_lens(model, input_ids, target_token_id=0, baseline_token_id=1, stack="encoder")
    assert set(diff) == {0, 1}

    with pytest.raises(ValueError):
        logit_change_with_mask(model, input_ids, stack="encoder")
    with pytest.raises(ValueError):
        residual_norms(model, None, inputs=ModelInputs())


def test_rollout_occlusion_scan_and_metrics_error_branches() -> None:
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 5))
    with pytest.raises(ValueError):
        attention_rollout(model, input_ids, head_agg="bad")
    seq2seq = _seq2seq_model()
    enc = torch.randint(0, seq2seq.cfg.vocab_size, (1, 4))
    dec = torch.randint(0, seq2seq.cfg.vocab_size, (1, 3))
    with pytest.raises(ValueError):
        attention_rollout(seq2seq, None, enc_input_ids=enc, dec_input_ids=dec, stack="decoder", kind="cross", layers=[0, 1])

    prob = token_occlusion_importance(model, input_ids, mode="prob")
    nll = token_occlusion_importance(model, input_ids, mode="nll")
    assert prob.shape == nll.shape == (input_ids.shape[1],)
    with pytest.raises(ValueError):
        token_occlusion_importance(model, input_ids, mode="bad")

    scan_prob = module_importance_scan(model, input_ids, mode="prob")
    scan_nll = module_importance_scan(model, input_ids, mode="nll")
    assert scan_prob and scan_nll

    logits = torch.randn(1, 3, 4)
    targets = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError):
        sequence_perplexity(logits, targets, dim=0)
    with pytest.raises(ValueError):
        sequence_negative_log_likelihood(logits, targets, dim=0)


def test_flops_masks_ablate_and_adapter_input_errors() -> None:
    causal = _causal_model()
    input_ids = torch.randint(0, causal.cfg.vocab_size, (1, 5))
    block_delta = logit_change_with_mask(causal, input_ids, attn_mask_type="block", block=2)
    dilated_delta = logit_change_with_mask(causal, input_ids, attn_mask_type="dilated", window=2, dilation=2)
    assert isinstance(block_delta, float) and isinstance(dilated_delta, float)
    with pytest.raises(ValueError):
        logit_change_with_mask(causal, input_ids, attn_mask_type="weird")

    info = estimate_layer_costs(_seq2seq_model(), seq_len=4, source_seq_len=6, stack="decoder")
    assert info["per_layer"][0]["attn_flops"] > 0
    with pytest.raises(AttributeError):
        estimate_layer_costs(object(), seq_len=4)

    base = causal(input_ids)
    with ablate_mlp_channels(causal, {0: [0, 1]}):
        ablated = causal(input_ids)
    assert torch.max(torch.abs(base - ablated)).item() > 0.0

    with pytest.raises(ValueError):
        coerce_model_inputs(causal, None)
    with pytest.raises(ValueError):
        coerce_model_inputs(_encoder_model(), None)
    with pytest.raises(ValueError):
        coerce_model_inputs(_seq2seq_model(), None)

    with pytest.raises(ValueError):
        resolve_model_score(causal, torch.randn(2, 3), score_fn=None)


def test_tracer_embedding_and_seq2seq_residual_paths() -> None:
    model = _seq2seq_model()
    enc = torch.randint(0, model.cfg.vocab_size, (1, 4))
    dec = torch.randint(0, model.cfg.vocab_size, (1, 3))
    tracer = ActivationTracer(model)
    tracer.add_embedding_output(stack="decoder")
    tracer.add_residual_streams(stack="decoder")
    with tracer.trace() as cache:
        _ = model(enc, dec)
    assert cache.get("decoder.embed") is not None
    residual = residual_norms(model, None, enc_input_ids=enc, dec_input_ids=dec, stack="decoder", dec_self_mask=torch.ones(1, 1, dec.shape[1], dec.shape[1]))
    assert residual["pre"].shape[0] == model.cfg.n_layers * 2
