from __future__ import annotations

import torch
import pytest

from interpret import (
    ActivationCache,
    ProbeDataset,
    CaptureSpec,
    ProbeFeatureSlice,
    SAEConfig,
    build_probe_dataset,
    fit_sae,
    path_patch_sweep,
    sae_reconstruction_metrics,
    split_probe_dataset,
    summarize_patch_sweep,
    summarize_path_patch_sweep,
    summarize_probe_dataset,
    summarize_probe_training_split,
    summarize_sae_training,
)
from runtime.causal import CausalLM
from specs.config import ModelConfig


def _causal_model() -> CausalLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = CausalLM(cfg)
    model.eval()
    return model


def test_patch_sweep_report_helpers_cover_block_and_head_shapes() -> None:
    block_report = summarize_patch_sweep(
        ["blocks.0", "blocks.1"],
        torch.tensor([[0.1, 0.4, 0.2], [0.3, 0.0, 0.5]]),
        topk=2,
        time_label="token_index",
    )
    assert block_report["shape"] == [2, 3]
    assert block_report["top_entries"][0]["name"] == "blocks.1"
    assert "token_index" in block_report["top_entries"][0]

    head_report = summarize_patch_sweep(
        ["blocks.0.attn", "blocks.1.attn"],
        torch.tensor(
            [
                [[0.1, 0.9], [0.4, 0.2]],
                [[0.3, 0.2], [0.8, 0.1]],
            ]
        ),
        topk=3,
        unit_label="head_index",
        time_label="source_index",
    )
    assert head_report["shape"] == [2, 2, 2]
    assert "head_index" in head_report["top_entries"][0]
    assert "source_index" in head_report["top_entries"][0]


def test_path_patch_sweep_and_report_surface() -> None:
    torch.manual_seed(0)
    model = _causal_model()
    clean = torch.randint(0, model.cfg.vocab_size, (1, 4))
    corrupted = clean.clone()
    corrupted[0, 1] = (corrupted[0, 1] + 1) % model.cfg.vocab_size

    result = path_patch_sweep(
        model,
        clean_input_ids=clean,
        corrupted_input_ids=corrupted,
        source_modules=["blocks.0", "blocks.1"],
        receiver_modules=["blocks.0.mlp", "blocks.1.mlp"],
    )
    assert result["target_restore"].shape == (2, 2)
    assert result["receiver_restore"].shape == (2, 2)
    report = summarize_path_patch_sweep(result, topk=2)
    assert report["shape"] == [2, 2]
    assert report["top_paths"]
    assert report["top_paths"][0]["source_module"] in {"blocks.0", "blocks.1"}


def test_probe_dataset_split_summary_and_training_report() -> None:
    cache = ActivationCache()
    feats = torch.arange(3 * 5 * 4, dtype=torch.float32).view(3, 5, 4)
    cache.store("feat", feats, CaptureSpec(move_to_cpu=False))
    targets = torch.tensor(
        [
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        dtype=torch.long,
    )
    ds = build_probe_dataset(cache, [ProbeFeatureSlice("feat")], targets=targets)
    summary = summarize_probe_dataset(ds)
    assert summary.num_rows == int(ds.x.shape[0])
    assert summary.target_kind == "classification"
    assert summary.target_counts

    train_ds, val_ds = split_probe_dataset(ds, val_fraction=0.25, generator=torch.Generator().manual_seed(0), stratify=True)
    assert train_ds.x.shape[0] + val_ds.x.shape[0] == ds.x.shape[0]
    report = summarize_probe_training_split(ds, val_fraction=0.25, train_rows=train_ds.x.shape[0], val_rows=val_ds.x.shape[0])
    assert report["train_rows"] == int(train_ds.x.shape[0])
    assert report["dataset"].num_targets == summary.num_targets


def test_sae_training_metrics_and_summary_surface() -> None:
    torch.manual_seed(1)
    features = torch.randn(16, 6)
    cfg = SAEConfig(code_dim=3, lr=0.01, epochs=4, batch_size=8, patience=2, device="cpu")
    sae, info = fit_sae(features, cfg=cfg)
    metrics = sae_reconstruction_metrics(sae, features)
    assert "reconstruction_mse" in metrics
    assert info["epochs_run"] >= 1
    assert isinstance(info["loss_history"], list)
    summary = summarize_sae_training(info, cfg=cfg)
    assert summary["requested_epochs"] == cfg.epochs
    assert "last_loss" in summary or not info["loss_history"]


def test_reporting_and_dataset_validation_branches() -> None:
    with pytest.raises(ValueError):
        summarize_patch_sweep(["a"], torch.ones(1))
    with pytest.raises(ValueError):
        summarize_patch_sweep(["a"], torch.ones(2, 2))
    with pytest.raises(ValueError):
        summarize_path_patch_sweep({"source_modules": ["s"], "receiver_modules": ["r"], "target_restore": [1.0], "receiver_restore": torch.ones(1, 1)})
    with pytest.raises(ValueError):
        summarize_path_patch_sweep({"source_modules": ["s"], "receiver_modules": ["r"], "target_restore": torch.ones(1, 1), "receiver_restore": torch.ones(1)})

    reg_ds = ProbeDataset(
        x=torch.ones(3, 4),
        y=torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32),
        batch_index=torch.tensor([0, 0, 1]),
        time_index=torch.tensor([0, 1, 0]),
        feature_keys=["feat"],
    )
    reg_summary = summarize_probe_dataset(reg_ds)
    assert reg_summary.target_kind == "regression"

    with pytest.raises(ValueError):
        split_probe_dataset(reg_ds, val_fraction=0.0)
    with pytest.raises(ValueError):
        split_probe_dataset(ProbeDataset(torch.ones(1, 4), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), ["feat"]), val_fraction=0.5)
    with pytest.raises(ValueError):
        split_probe_dataset(reg_ds, val_fraction=0.5, stratify=True)
    singleton_ds = ProbeDataset(
        x=torch.ones(2, 4),
        y=torch.tensor([0, 1]),
        batch_index=torch.tensor([0, 1]),
        time_index=torch.tensor([0, 0]),
        feature_keys=["feat"],
    )
    with pytest.raises(ValueError):
        split_probe_dataset(singleton_ds, val_fraction=0.5, stratify=True)

    with pytest.raises(ValueError):
        path_patch_sweep(_causal_model(), clean_input_ids=torch.tensor([[1, 2]]), corrupted_input_ids=torch.tensor([[1, 3]]), source_modules=[], receiver_modules=["blocks.0"])
