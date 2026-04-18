from __future__ import annotations

import math

import torch
import torch.nn as nn

from interpret import (
    ActivationCache,
    ActivationContext,
    CaptureSpec,
    FeatureSlice,
    ProbeDataset,
    ProbeDatasetSummary,
    SparseAutoencoder,
    ablate_mlp_channels,
    activation_contexts,
    attention_rollout,
    channel_stats,
    fit_sae,
    estimate_layer_costs,
    flatten_features,
    logit_change_with_mask,
    module_importance_scan,
    output_patching,
    output_patching_slice,
    residual_norms,
    sae_boost_features,
    sae_reconstruction_metrics,
    sae_encode,
    sae_feature_mask,
    sae_mask_features,
    sequence_negative_log_likelihood,
    SAEConfig,
    sequence_perplexity,
    steer_residual,
    path_patch_sweep,
    summarize_patch_sweep,
    summarize_path_patch_sweep,
    split_probe_dataset,
    summarize_probe_dataset,
    summarize_probe_training_split,
    summarize_sae_training,
    targets_from_tokens,
    token_entropy,
    token_surprisal,
)
from runtime.causal import CausalLM
from specs.config import ModelConfig


class _TimeProj(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _causal_model() -> CausalLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = CausalLM(cfg)
    model.eval()
    return model


def test_feature_cache_utilities_cover_remaining_surface() -> None:
    cache = ActivationCache()
    spec = CaptureSpec(move_to_cpu=False, dtype=torch.float64, detach=True, clone=True)
    a = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    b = (a + 100).clone()
    cache.store("a", a, spec)
    cache.store("b", b, CaptureSpec(move_to_cpu=False))

    flat = flatten_features(
        cache,
        [
            FeatureSlice("a", time_slice=slice(0, 2)),
            FeatureSlice("b", time_slice=slice(0, 2)),
        ],
    )
    assert flat.shape == (4, 8)
    assert flat.dtype == torch.float64

    input_ids = torch.arange(6, dtype=torch.long).view(2, 3)
    targets = targets_from_tokens(input_ids, time_slice=slice(0, 2), target_shift=1)
    assert targets.tolist() == [1, 2, 4, 5]

    contexts = activation_contexts(input_ids, [(0, 1, 0.5), (1, 2, 1.5)], window=1)
    assert len(contexts) == 2
    assert isinstance(contexts[0], ActivationContext)
    assert contexts[0].tokens == [0, 1, 2]

    stats = channel_stats(cache, "a")
    assert set(stats) == {"mean", "std", "max"}
    assert stats["mean"].shape == (4,)

    probe_dataset = ProbeDataset(
        x=flat,
        y=torch.arange(flat.shape[0]),
        batch_index=torch.tensor([0, 0, 1, 1]),
        time_index=torch.tensor([0, 1, 0, 1]),
        feature_keys=["a", "b"],
    )
    assert probe_dataset.feature_keys == ["a", "b"]
    summary = summarize_probe_dataset(probe_dataset)
    assert isinstance(summary, ProbeDatasetSummary)
    train_ds, val_ds = split_probe_dataset(probe_dataset, val_fraction=0.25, generator=torch.Generator().manual_seed(0))
    report = summarize_probe_training_split(probe_dataset, val_fraction=0.25, train_rows=train_ds.x.shape[0], val_rows=val_ds.x.shape[0])
    assert report["dataset"].num_rows == summary.num_rows


def test_output_patching_helpers_cover_full_and_sliced_replacement() -> None:
    torch.manual_seed(0)
    model = _TimeProj()
    x = torch.randn(1, 4, 3)
    base = model(x)
    replacement = torch.full_like(base, 5.0)

    with output_patching(model, {"proj": replacement}):
        patched = model(x)
    assert torch.allclose(patched, replacement)

    with output_patching_slice(model, {"proj": replacement}, time_slice=slice(1, 3)):
        sliced = model(x)
    assert torch.allclose(sliced[:, 1:3], replacement[:, 1:3])
    assert torch.allclose(sliced[:, 0], base[:, 0])
    assert torch.allclose(sliced[:, 3], base[:, 3])


def test_sae_helper_surface_covers_encode_mask_and_boost() -> None:
    torch.manual_seed(1)
    sae = SparseAutoencoder(4, 3)
    x = torch.randn(2, 5, 4)

    z = sae_encode(sae, x)
    masked = sae_mask_features(sae, x, [0], invert=False)
    boosted = sae_boost_features(sae, x, [1], factor=2.0)

    assert z.shape == (2, 5, 3)
    assert masked.shape == x.shape
    assert boosted.shape == x.shape
    assert not torch.allclose(masked, boosted)


def test_metric_wrappers_cover_token_and_sequence_helpers() -> None:
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]]])
    targets = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[1, 1, 0]])

    surprisal = token_surprisal(logits, targets, attn_mask=mask)
    entropy = token_entropy(logits, attn_mask=mask)
    ppl = sequence_perplexity(logits, targets, mask=mask)
    nll = sequence_negative_log_likelihood(logits, targets, mask=mask)

    assert surprisal.shape == (1, 3)
    assert entropy.shape == (1, 3)
    assert surprisal[0, 2].item() == 0.0
    assert entropy[0, 2].item() == 0.0
    assert torch.isfinite(ppl)
    assert torch.isfinite(nll)
    assert float(ppl) > 0.0


def test_remaining_model_level_interpret_helpers_cover_exports() -> None:
    torch.manual_seed(2)
    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 6))

    rollout = attention_rollout(model, input_ids)
    assert rollout.shape == (1, input_ids.shape[1], input_ids.shape[1])
    assert torch.isfinite(rollout).all()

    residual = residual_norms(model, input_ids)
    assert residual["pre"].shape == (model.cfg.n_layers,)
    assert residual["post"].shape == (model.cfg.n_layers,)

    scan = module_importance_scan(model, input_ids)
    assert scan
    assert isinstance(scan[0][0], str)
    assert isinstance(scan[0][1], float)

    costs = estimate_layer_costs(model, seq_len=input_ids.shape[1])
    assert costs["total_flops"] > 0
    assert len(costs["per_layer"]) == model.cfg.n_layers

    delta = logit_change_with_mask(model, input_ids, attn_mask_type="sliding", window=1)
    assert isinstance(delta, float)
    assert math.isfinite(delta)

    base = model(input_ids)
    with steer_residual(model, {0: torch.ones(model.cfg.d_model)}, scale=3.0):
        steered = model(input_ids)
    assert torch.max(torch.abs(base - steered)).item() > 0.0

    with ablate_mlp_channels(model, {0: [0, 1, 2]}):
        ablated = model(input_ids)
    assert torch.max(torch.abs(base - ablated)).item() > 0.0

    sae = SparseAutoencoder(model.cfg.d_model, 8)
    with sae_feature_mask(model, 0, sae, drop_codes=[0, 1]):
        masked = model(input_ids)
    assert torch.max(torch.abs(base - masked)).item() > 0.0

    trained_sae, info = fit_sae(torch.randn(8, model.cfg.d_model), cfg=SAEConfig(code_dim=4, epochs=2, batch_size=4, patience=1, device="cpu"))
    assert sae_reconstruction_metrics(trained_sae, torch.randn(4, model.cfg.d_model))["reconstruction_mse"] >= 0.0
    assert summarize_sae_training(info)["epochs_run"] >= 1
