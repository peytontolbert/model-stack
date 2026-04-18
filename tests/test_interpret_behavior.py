from __future__ import annotations

import pytest

pytest.importorskip("torch.nn")

import torch

from interpret import (
    ActivationCache,
    ActivationTracer,
    ModelAdapter,
    ProbeFeatureSlice,
    SparseAutoencoder,
    SAEConfig,
    LinearProbeConfig,
    ablate_attention_heads,
    attention_entropy_for_layer,
    attention_snapshot_for_layer,
    attention_weights_for_layer,
    block_output_patch_sweep,
    build_probe_dataset,
    component_logit_attribution,
    cross_attention_head_patch_sweep,
    causal_trace_restore_fraction,
    feature_coactivation_matrix,
    fit_linear_probe,
    fit_sae,
    greedy_head_recovery,
    grad_x_input_tokens,
    head_grad_saliencies,
    head_logit_attribution,
    head_patch_sweep,
    integrated_gradients_tokens,
    logit_diff_lens,
    logit_lens,
    mlp_lens,
    mlp_neuron_logit_attribution,
    mlp_neuron_patch_sweep,
    path_patch_effect,
    sae_feature_logit_attribution,
    sae_feature_dashboard,
    search_feature_activations,
    token_occlusion_importance,
    topk_feature_positions,
    topk_positions,
    ModelInputs,
)
from runtime.causal import CausalLM
from runtime.encoder import EncoderModel
from runtime.seq2seq import EncoderDecoderLM
from specs.config import ModelConfig


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


def _encoder_model() -> EncoderModel:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = EncoderModel(cfg)
    model.eval()
    return model


def test_model_adapter_supports_causal_and_seq2seq() -> None:
    causal = ModelAdapter(_causal_model())
    assert causal.kind == "causal"
    assert len(causal.attention_targets()) == 2

    seq2seq = ModelAdapter(_seq2seq_model())
    assert seq2seq.kind == "encoder_decoder"
    assert len(seq2seq.attention_targets(stack="decoder", kind="self")) == 2
    assert len(seq2seq.attention_targets(stack="decoder", kind="cross")) == 2


def test_tracer_and_lenses_capture_expected_shapes() -> None:
    torch.manual_seed(0)
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 5))

    tracer = ActivationTracer(model)
    tracer.add_block_outputs()
    with tracer.trace() as cache:
        logits = model(input_ids)

    assert logits.shape == (1, 5, model.cfg.vocab_size)
    assert cache.get("blocks.0") is not None
    assert set(logit_lens(model, input_ids, topk=3)) == {0, 1}
    assert set(mlp_lens(model, input_ids, topk=3)) == {0, 1}
    scores = logit_diff_lens(model, input_ids, target_token_id=0, baseline_token_id=1)
    assert set(scores) == {0, 1}


def test_tracer_interpret_surfaces_capture_runtime_points() -> None:
    torch.manual_seed(11)
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 4))
    tracer = ActivationTracer(model)
    tracer.add_interpret_surfaces()
    with tracer.trace() as cache:
        _ = model(input_ids)
    assert cache.get("embed") is not None
    assert cache.get("blocks.0.resid_pre") is not None
    assert cache.get("blocks.0.resid_post") is not None
    assert cache.get("blocks.0.attn.attn_probs") is not None
    assert cache.get("blocks.0.mlp.mlp_mid") is not None


def test_attention_snapshot_and_entropy_follow_runtime_path() -> None:
    torch.manual_seed(1)
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 4))

    snapshot = attention_snapshot_for_layer(model, input_ids, layer_index=0)
    assert snapshot.probs is not None
    assert snapshot.probs.shape == (1, model.cfg.n_heads, 4, 4)
    assert snapshot.head_out is not None
    assert attention_weights_for_layer(model, input_ids, layer_index=0).shape == snapshot.probs.shape
    assert attention_entropy_for_layer(model, input_ids, layer_index=0).shape == (1, model.cfg.n_heads, 4)


def test_attention_snapshot_supports_cross_attention() -> None:
    torch.manual_seed(2)
    model = _seq2seq_model()
    enc = torch.randint(0, model.cfg.vocab_size, (1, 4))
    dec = torch.randint(0, model.cfg.vocab_size, (1, 3))

    snapshot = attention_snapshot_for_layer(
        model,
        None,
        layer_index=0,
        enc_input_ids=enc,
        dec_input_ids=dec,
        stack="decoder",
        kind="cross",
    )
    assert snapshot.probs is not None
    assert snapshot.probs.shape[-2:] == (3, 4)


def test_ablation_and_head_saliency_are_behavioral() -> None:
    torch.manual_seed(3)
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 5))
    base = model(input_ids)
    with ablate_attention_heads(model, {0: [0]}):
        ablated = model(input_ids)
    assert base.shape == ablated.shape
    assert not torch.allclose(base, ablated)

    sal = head_grad_saliencies(model, input_ids)
    assert sal.shape == (model.cfg.n_layers, model.cfg.n_heads)
    assert torch.all(sal >= 0)


def test_direct_attribution_surfaces_return_finite_outputs() -> None:
    torch.manual_seed(4)
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 5))
    sae = SparseAutoencoder(model.cfg.d_model, 8)

    component_scores = component_logit_attribution(model, input_ids)
    assert "embed" in component_scores
    assert "blocks.0.attn" in component_scores

    head_scores = head_logit_attribution(model, input_ids, layer_index=0)
    assert head_scores.shape == (model.cfg.n_heads,)
    assert torch.isfinite(head_scores).all()

    neuron_scores = mlp_neuron_logit_attribution(model, input_ids, layer_index=0)
    assert neuron_scores.ndim == 1
    assert neuron_scores.numel() > 0
    assert torch.isfinite(neuron_scores).all()

    sae_scores = sae_feature_logit_attribution(model, input_ids, layer_index=0, sae=sae)
    assert sae_scores.shape == (8,)
    assert torch.isfinite(sae_scores).all()


def test_direct_attribution_supports_encoder_feature_scores() -> None:
    torch.manual_seed(12)
    model = _encoder_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 5))
    component_scores = component_logit_attribution(model, input_ids, target_feature_index=0, stack="encoder")
    assert "embed" in component_scores
    head_scores = head_logit_attribution(model, input_ids, layer_index=0, target_feature_index=0, stack="encoder")
    assert head_scores.shape == (model.cfg.n_heads,)
    neuron_scores = mlp_neuron_logit_attribution(model, input_ids, layer_index=0, target_feature_index=0, stack="encoder")
    assert neuron_scores.numel() > 0


def test_patch_sweeps_and_path_patch_return_expected_shapes() -> None:
    torch.manual_seed(5)
    model = _causal_model()
    clean = torch.randint(0, model.cfg.vocab_size, (1, 4))
    corrupted = clean.clone()
    corrupted[0, 1] = (corrupted[0, 1] + 1) % model.cfg.vocab_size

    block = block_output_patch_sweep(model, clean_input_ids=clean, corrupted_input_ids=corrupted)
    assert block["scores"].shape == (model.cfg.n_layers, clean.shape[1])

    heads = head_patch_sweep(model, clean_input_ids=clean, corrupted_input_ids=corrupted)
    assert heads["scores"].shape == (model.cfg.n_layers, model.cfg.n_heads, clean.shape[1])

    neurons = mlp_neuron_patch_sweep(model, clean_input_ids=clean, corrupted_input_ids=corrupted, neurons=[0, 1, 2])
    assert neurons["scores"].shape == (model.cfg.n_layers, 3, clean.shape[1])

    effect = path_patch_effect(model, clean_input_ids=clean, corrupted_input_ids=corrupted, source_module="blocks.0", receiver_module="blocks.1")
    assert "receiver_restore_fraction" in effect
    assert "target_logit_restore_fraction" in effect


def test_legacy_attribution_and_patch_helpers_follow_adapter_inputs() -> None:
    torch.manual_seed(13)
    model = _seq2seq_model()
    enc = torch.randint(0, model.cfg.vocab_size, (1, 4))
    dec = torch.randint(0, model.cfg.vocab_size, (1, 3))
    dec_cor = dec.clone()
    dec_cor[0, -1] = (dec_cor[0, -1] + 1) % model.cfg.vocab_size

    ig = integrated_gradients_tokens(model, None, enc_input_ids=enc, dec_input_ids=dec, stack="decoder", steps=4)
    gxi = grad_x_input_tokens(model, None, enc_input_ids=enc, dec_input_ids=dec, stack="decoder")
    occ = token_occlusion_importance(model, None, enc_input_ids=enc, dec_input_ids=dec, stack="decoder")
    frac = causal_trace_restore_fraction(
        model,
        clean_inputs=ModelInputs.encoder_decoder(enc, dec),
        corrupted_inputs=ModelInputs.encoder_decoder(enc, dec_cor),
        patch_points=["decoder.0.cross_block"],
    )
    assert ig.shape == gxi.shape == occ.shape == (dec.shape[1],)
    assert frac.numel() > 0


def test_cross_attention_head_patch_sweep_on_seq2seq() -> None:
    torch.manual_seed(6)
    model = _seq2seq_model()
    enc_clean = torch.randint(0, model.cfg.vocab_size, (1, 4))
    enc_cor = enc_clean.clone()
    enc_cor[0, 0] = (enc_cor[0, 0] + 2) % model.cfg.vocab_size
    dec = torch.randint(0, model.cfg.vocab_size, (1, 3))

    sweep = cross_attention_head_patch_sweep(
        model,
        clean_inputs=ModelInputs.encoder_decoder(enc_clean, dec),
        corrupted_inputs=ModelInputs.encoder_decoder(enc_cor, dec),
    )
    assert sweep["scores"].shape == (model.cfg.n_layers, model.cfg.n_heads, dec.shape[1])


def test_probe_builder_aligns_and_concatenates_features() -> None:
    cache = ActivationCache()
    a = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    b = torch.arange(2 * 5 * 2, dtype=torch.float32).view(2, 5, 2)
    cache.store("a", a)
    cache.store("b", b)
    input_ids = torch.arange(10, dtype=torch.long).view(2, 5)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.long)

    ds = build_probe_dataset(
        cache,
        [ProbeFeatureSlice("a"), ProbeFeatureSlice("b", time_offset=-1)],
        input_ids=input_ids,
        mask=mask,
        target_shift=1,
    )
    assert ds.x.shape[1] == 5
    assert ds.y.ndim == 1
    assert ds.batch_index.shape == ds.time_index.shape == ds.y.shape


def test_fit_linear_probe_handles_scalar_regression() -> None:
    torch.manual_seed(7)
    x = torch.randn(64, 4)
    y = (2 * x[:, 0] - x[:, 1]).float()
    probe, score = fit_linear_probe(
        x,
        y,
        cfg=LinearProbeConfig(task="regression", l2=0.0, lr=0.05, epochs=30, batch_size=16, patience=5, device="cpu"),
    )
    assert probe.linear.weight.shape[0] == 1
    assert score >= 0.0


def test_activation_mining_helpers_return_contexts_and_dashboard() -> None:
    cache = ActivationCache()
    feats = torch.tensor(
        [
            [[0.1, 0.2, 0.0], [0.5, 0.1, 0.4], [0.3, 0.8, 0.1]],
            [[0.7, 0.2, 0.4], [0.1, 0.9, 0.3], [0.6, 0.5, 0.2]],
        ],
        dtype=torch.float32,
    )
    cache.store("feat", feats)
    input_ids = torch.arange(6, dtype=torch.long).view(2, 3)
    sae = SparseAutoencoder(3, 4)

    assert topk_positions(cache, "feat", k=2)
    assert topk_feature_positions(cache, "feat", 1, k=2)
    coact = feature_coactivation_matrix(cache, "feat")
    assert coact.shape == (3, 3)
    contexts = search_feature_activations(cache, "feat", 1, input_ids=input_ids, k=2, window=1)
    assert len(contexts) == 2
    dash = sae_feature_dashboard(cache, "feat", sae, 0, input_ids=input_ids, k=2, window=1)
    assert "top_contexts" in dash


def test_greedy_head_recovery_evaluates_combined_patches() -> None:
    torch.manual_seed(8)
    model = _causal_model()
    clean = torch.randint(0, model.cfg.vocab_size, (1, 4))
    corrupted = clean.clone()
    corrupted[0, -1] = (corrupted[0, -1] + 3) % model.cfg.vocab_size

    res = greedy_head_recovery(model, clean_input_ids=clean, corrupted_input_ids=corrupted, k=2)
    assert len(res["selected"]) == 2
    assert len(res["curve"]) == 2
    assert res["table"].shape == (model.cfg.n_layers, model.cfg.n_heads)


def test_sae_training_config_surface_still_works() -> None:
    torch.manual_seed(9)
    feats = torch.randn(64, 6)
    sae, info = fit_sae(feats, cfg=SAEConfig(code_dim=4, epochs=2, batch_size=16, patience=1, device="cpu"))
    assert isinstance(sae, SparseAutoencoder)
    assert "loss" in info
