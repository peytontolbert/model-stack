from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch.nn")

from interpret import (
    attention_mask_summary,
    attention_receptive_field,
    attention_pattern_summary,
    centered_kernel_alignment,
    compare_model_representations,
    concept_direction_effect,
    concept_direction_from_means,
    capture_moe_router_logits,
    causal_mask_violation_count,
    channel_outlier_scores,
    embedding_anisotropy,
    embedding_norms,
    embedding_projection_scores,
    erase_direction,
    explanation_stability,
    expert_usage_from_logits,
    faithfulness_summary,
    find_moe_targets,
    feature_activation_jaccard,
    feature_correlation_graph,
    generation_logit_trace,
    logit_prism_components,
    greedy_module_circuit,
    module_attribution_patching,
    module_recovery_scores,
    randomization_rank_baseline,
    render_interpretability_html_report,
    save_interpretability_html_report,
    spearman_rank_correlation,
    stability_summary,
    summarize_attention_receptive_field,
    summarize_activation_outliers,
    summarize_attribution_patching,
    summarize_feature_circuit,
    summarize_generation_trace,
    summarize_logit_prism,
    summarize_module_circuit,
    summarize_moe_router_usage,
    tensor_norm_summary,
    token_deletion_insertion_curves,
    token_embedding_similarity,
    token_trigger_append_scan,
    token_trigger_position_scan,
)
from runtime.causal import CausalLM
from specs.config import ModelConfig


def _causal_model() -> CausalLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = CausalLM(cfg)
    model.eval()
    return model


def test_token_faithfulness_curves_and_summary() -> None:
    torch.manual_seed(0)
    model = _causal_model()
    input_ids = torch.randint(1, model.cfg.vocab_size, (1, 4))
    scores = torch.tensor([[0.2, 0.9, 0.1, 0.4]])
    curves = token_deletion_insertion_curves(model, input_ids, scores, baseline_token_id=0)
    assert curves["deletion"].shape[0] == 5
    assert curves["insertion"].shape[0] == 5
    summary = faithfulness_summary(curves)
    assert "deletion_auc" in summary
    assert "insertion_auc" in summary
    assert spearman_rank_correlation(scores, scores).item() > 0.99


def test_greedy_module_circuit_returns_selected_modules() -> None:
    torch.manual_seed(1)
    model = _causal_model()
    clean = torch.randint(1, model.cfg.vocab_size, (1, 4))
    corrupted = clean.clone()
    corrupted[0, 1] = (corrupted[0, 1] + 1) % model.cfg.vocab_size
    candidates = ["blocks.0", "blocks.1", "blocks.0.attn", "blocks.1.mlp"]

    scores = module_recovery_scores(model, clean_input_ids=clean, corrupted_input_ids=corrupted, candidate_modules=candidates)
    assert scores["scores"].shape == (len(candidates),)

    result = greedy_module_circuit(
        model,
        clean_input_ids=clean,
        corrupted_input_ids=corrupted,
        candidate_modules=candidates,
        k=2,
    )
    assert len(result["selected"]) <= 2
    assert isinstance(summarize_module_circuit(result), list)


def test_stability_report_and_trigger_helpers(tmp_path) -> None:
    scores = torch.tensor([0.1, 0.4, 0.2])
    stable = explanation_stability(scores, [scores, scores + 0.1])
    assert stable["mean"] > 0.9
    assert randomization_rank_baseline(scores, trials=3).shape == (3,)
    assert "random_mean" in stability_summary(scores, [scores], random_trials=3)

    html = render_interpretability_html_report(title="Interpret", sections=[("Scores", {"scores": scores})])
    assert "<html>" in html
    path = save_interpretability_html_report(tmp_path / "report.html", title="Interpret", sections=[("Scores", {"scores": scores})])
    assert path.endswith("report.html")

    torch.manual_seed(2)
    model = _causal_model()
    input_ids = torch.randint(1, model.cfg.vocab_size, (1, 4))
    append_rows = token_trigger_append_scan(model, input_ids, [1, 2])
    position_rows = token_trigger_position_scan(model, input_ids, 1)
    assert len(append_rows) == 2
    assert len(position_rows) == input_ids.shape[1]


class _TinyMoE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = torch.nn.Linear(4, 3, bias=False)
        self.experts = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        self.num_experts = 3
        self.k = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        idx = logits.argmax(dim=-1)
        outs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        gathered = outs.gather(-2, idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, x.shape[-1]))
        return gathered.squeeze(-2)


def test_moe_router_usage_helpers() -> None:
    model = _TinyMoE()
    targets = find_moe_targets(model)
    assert len(targets) == 1
    x = torch.randn(1, 2, 4)
    with capture_moe_router_logits(model) as cache:
        _ = model(x)
    rows = summarize_moe_router_usage(cache, targets)
    assert rows[0]["num_experts"] == 3
    usage = expert_usage_from_logits(cache.get(".router_logits"), k=1)
    assert usage["usage"].shape == (3,)


def test_representation_similarity_and_concept_effects() -> None:
    x = torch.randn(2, 3, 4)
    y = x + 0.01 * torch.randn_like(x)
    assert centered_kernel_alignment(x, x).item() > 0.99
    assert centered_kernel_alignment(x, y).item() > 0.9

    direction = concept_direction_from_means(x + 1.0, x - 1.0)
    erased = erase_direction(x, direction)
    assert erased.shape == x.shape

    torch.manual_seed(3)
    model = _causal_model()
    input_ids = torch.randint(1, model.cfg.vocab_size, (1, 4))
    sims = compare_model_representations(model, model, module_names=["blocks.0"], input_ids=input_ids)
    assert sims["blocks.0"]["cka"] > 0.99
    effect = concept_direction_effect(model, input_ids=input_ids, module_name="blocks.0", direction=torch.randn(model.cfg.d_model))
    assert "delta" in effect

    attr = module_attribution_patching(
        model,
        clean_input_ids=input_ids,
        corrupted_input_ids=input_ids.clone(),
        candidate_modules=["blocks.0", "blocks.1"],
    )
    assert attr["scores"].shape == (2,)
    assert isinstance(summarize_attribution_patching(attr), list)

    graph = feature_correlation_graph(torch.randn(8, 3), torch.randn(8, 4), topk=3)
    assert len(summarize_feature_circuit(graph)) == 3
    assert feature_activation_jaccard(torch.tensor([[1.0, -1.0], [2.0, 3.0]]), torch.tensor([[1.0, 1.0], [0.0, 3.0]])).shape == (2,)


def test_standalone_model_stack_diagnostics() -> None:
    emb = torch.nn.Embedding(8, 4)
    assert embedding_norms(emb).shape == (8,)
    assert "mean_norm" in embedding_anisotropy(emb)
    vals, idx = __import__("interpret").nearest_embedding_neighbors(emb, 0, topk=3)
    assert vals.shape == idx.shape == (3,)
    assert token_embedding_similarity(emb.weight, torch.tensor([0, 1, 2])).shape == (3, 3)
    assert embedding_projection_scores(emb, torch.randn(4)).shape == (8,)

    x = torch.randn(2, 3, 4)
    y = x + 0.1
    assert tensor_norm_summary(x)["shape"] == (2, 3, 4)
    assert __import__("interpret").residual_update_ratio(x, y).shape == (2, 3)

    mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert attention_mask_summary(mask)["shape"] == (4, 4)
    assert attention_receptive_field(mask).shape == (4,)
    assert causal_mask_violation_count(mask) == 0
    assert summarize_attention_receptive_field(mask)["max"] >= 1.0


def test_generation_trace_helpers() -> None:
    torch.manual_seed(4)
    model = _causal_model()
    input_ids = torch.randint(1, model.cfg.vocab_size, (1, 3))
    trace = generation_logit_trace(model, input_ids, steps=2, topk=3)
    summary = summarize_generation_trace(trace)
    assert trace["tokens"].shape[1] == 5
    assert summary["steps"] == 2


def test_deeper_stack_standalone_diagnostics() -> None:
    attn = torch.softmax(torch.randn(1, 2, 4, 4), dim=-1)
    patterns = attention_pattern_summary(attn, prefix_len=2)
    assert patterns["diagonal"].shape == (1, 2)
    assert patterns["distance"].shape == (1, 2)

    x = torch.randn(3, 4, 5)
    assert channel_outlier_scores(x).shape == (5,)
    assert len(summarize_activation_outliers(x, topk=2)) == 2

    lm_head = torch.randn(7, 5)
    comps = {"a": torch.randn(1, 3, 5), "b": torch.randn(1, 3, 5)}
    prism = logit_prism_components(comps, lm_head, target_token_id=1, baseline_token_id=2)
    assert set(prism) == {"a", "b"}
    assert len(summarize_logit_prism(prism)) == 2
