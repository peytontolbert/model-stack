from .activation_cache import ActivationCache, CaptureSpec
from .tracer import ActivationTracer
from .logit_lens import logit_lens
from .tuned_lens import AffineTunedLens, collect_layer_hidden_from_cache, tuned_lens_logits, tuned_lens_topk, tuned_lens_training_pairs
from .generation import generation_logit_trace, summarize_generation_trace
from .model_adapter import ModelAdapter, ModelInputs

# Subpackages
from .probes.linear import LinearProbe, LinearProbeConfig, fit_linear_probe
from .causal.patching import output_patching, causal_trace_restore_fraction
from .attribution.integrated_gradients import integrated_gradients_tokens
from .attribution.grad_x_input import grad_x_input_tokens
from .attribution.occlusion import token_occlusion_importance
from .attn.weights import attention_weights_for_layer, attention_entropy_for_layer
from .attn.weights import attention_snapshot_for_layer
from .attn.ablate import ablate_attention_heads
from .attn.rollout import attention_rollout
from .attn.saliency import head_grad_saliencies
from .features.index import flatten_features, FeatureSlice, targets_from_tokens
from .features.sae import SparseAutoencoder, SAEConfig, fit_sae, sae_reconstruction_metrics
from .features.sae_ops import sae_encode, sae_decode, sae_mask_features, sae_boost_features
from .features.stats import channel_stats
from .neuron.mlp_lens import mlp_lens
from .neuron.ablate import ablate_mlp_channels
from .logit_diff import logit_diff_lens
from .causal.steer import steer_residual
from .causal.sae_patch import sae_feature_mask
from .search.greedy import greedy_head_recovery
from .causal.sweeps import (
    block_output_patch_sweep,
    cross_attention_head_patch_sweep,
    head_patch_sweep,
    mlp_neuron_patch_sweep,
    path_patch_effect,
    path_patch_sweep,
)
from .metrics.token import token_surprisal, token_entropy
from .features.mining import (
    ActivationContext,
    activation_contexts,
    feature_coactivation_matrix,
    sae_feature_dashboard,
    search_feature_activations,
    topk_feature_positions,
    topk_positions,
)
from .importance.module_scan import module_importance_scan
from .analysis.residual import residual_norms
from .causal.slice_patching import output_patching_slice
from .analysis.flops import estimate_layer_costs
from .analysis.mask_effects import logit_change_with_mask
from .analysis.similarity import (
    cache_representation_similarity,
    centered_kernel_alignment,
    compare_model_representations,
    flatten_representation,
    representation_cosine_similarity,
)
from .analysis.embeddings import embedding_anisotropy, embedding_norms, embedding_projection_scores, nearest_embedding_neighbors, token_embedding_similarity
from .analysis.block import block_internal_keys, block_internal_norms, residual_stream_delta_table
from .analysis.norms import module_activation_norms, module_parameter_norms, residual_update_ratio, tensor_norm_summary
from .analysis.outliers import activation_kurtosis, channel_outlier_scores, summarize_activation_outliers
from .attn.masks import attention_mask_summary, attention_receptive_field, causal_mask_violation_count, summarize_attention_receptive_field
from .attn.patterns import attention_diagonal_score, attention_distance, attention_pattern_summary, attention_prefix_mass, attention_previous_token_score
from .attribution.logit_prism import logit_prism_components, summarize_logit_prism, unembed_vector_scores
from .causal.concept import boost_direction, concept_direction_effect, concept_direction_from_means, erase_direction, patch_module_direction
from .causal.attribution_patching import module_attribution_patching, module_integrated_attribution_patching, summarize_attribution_patching
from .causal.interchange import interchange_intervention_effect
from .features.circuits import feature_activation_jaccard, feature_correlation_graph, summarize_feature_circuit
from .metrics.sequence import sequence_perplexity, sequence_negative_log_likelihood
from .metrics.faithfulness import (
    area_over_curve,
    faithfulness_summary,
    spearman_rank_correlation,
    token_deletion_insertion_curves,
)
from .metrics.stability import explanation_stability, randomization_rank_baseline, stability_summary
from .search.circuit import greedy_module_circuit, module_recovery_scores, summarize_module_circuit
from .importance.moe import MoETarget, capture_moe_router_logits, expert_usage_from_logits, find_moe_targets, summarize_moe_router_usage
from .reports import render_interpretability_html_report, save_interpretability_html_report
from .safety import token_trigger_append_scan, token_trigger_position_scan
from .attribution.direct import (
    component_logit_attribution,
    head_logit_attribution,
    mlp_neuron_logit_attribution,
    sae_feature_logit_attribution,
)
from .reporting import summarize_patch_sweep, summarize_path_patch_sweep, summarize_probe_training_split, summarize_sae_training
from .probes.dataset import ProbeDataset, ProbeDatasetSummary, ProbeFeatureSlice, build_probe_dataset, split_probe_dataset, summarize_probe_dataset
from .diffusion import (
    DiffusionInputs,
    DiffusionModelAdapter,
    DiffusionPatchResult,
    DiffusionTraceDataset,
    DiffusionTracer,
    PromptTokenAttribution,
    classifier_logit_score,
    clip_similarity_score,
    collect_diffusion_attention_maps,
    cosine_distance,
    diffusion_module_patch_sweep,
    diffusion_attention_phase_summary,
    get_diffusion_adapter,
    mse_distance,
    patch_denoiser_latents,
    patch_diffusion_module_outputs,
    prompt_counterfactual_delta,
    prompt_token_occlusion_importance,
    resolve_diffusion_score,
    summarize_diffusion_steps,
    summarize_diffusion_trace_dataset,
    summarize_prompt_token_attribution,
    tensor_mean_score,
    tensor_region_score,
    token_region_attribution,
    trace_diffusion_generation,
    trace_prompt_dataset,
)
from .multimodal import (
    MultimodalInputs,
    MultimodalModelAdapter,
    MultimodalTracer,
    get_multimodal_adapter,
    trace_multimodal_forward,
)

__all__ = [
    "ActivationCache",
    "CaptureSpec",
    "ActivationTracer",
    "logit_lens",
    "AffineTunedLens",
    "collect_layer_hidden_from_cache",
    "tuned_lens_logits",
    "tuned_lens_topk",
    "tuned_lens_training_pairs",
    "generation_logit_trace",
    "summarize_generation_trace",
    "ModelAdapter",
    "ModelInputs",
    "LinearProbe",
    "LinearProbeConfig",
    "fit_linear_probe",
    "output_patching",
    "causal_trace_restore_fraction",
    "integrated_gradients_tokens",
    "grad_x_input_tokens",
    "token_occlusion_importance",
    "attention_snapshot_for_layer",
    "attention_weights_for_layer",
    "attention_entropy_for_layer",
    "ablate_attention_heads",
    "attention_rollout",
    "head_grad_saliencies",
    "flatten_features",
    "FeatureSlice",
    "targets_from_tokens",
    "SparseAutoencoder",
    "SAEConfig",
    "fit_sae",
    "sae_reconstruction_metrics",
    "sae_encode",
    "sae_decode",
    "sae_mask_features",
    "sae_boost_features",
    "channel_stats",
    "mlp_lens",
    "ablate_mlp_channels",
    "logit_diff_lens",
    "steer_residual",
    "sae_feature_mask",
    "greedy_head_recovery",
    "block_output_patch_sweep",
    "head_patch_sweep",
    "cross_attention_head_patch_sweep",
    "mlp_neuron_patch_sweep",
    "path_patch_effect",
    "path_patch_sweep",
    "component_logit_attribution",
    "head_logit_attribution",
    "mlp_neuron_logit_attribution",
    "sae_feature_logit_attribution",
    "token_surprisal",
    "token_entropy",
    "ActivationContext",
    "activation_contexts",
    "feature_coactivation_matrix",
    "sae_feature_dashboard",
    "search_feature_activations",
    "topk_feature_positions",
    "topk_positions",
    "ProbeDataset",
    "ProbeDatasetSummary",
    "ProbeFeatureSlice",
    "build_probe_dataset",
    "split_probe_dataset",
    "summarize_probe_dataset",
    "summarize_patch_sweep",
    "summarize_path_patch_sweep",
    "summarize_probe_training_split",
    "summarize_sae_training",
    "module_importance_scan",
    "residual_norms",
    "output_patching_slice",
    "estimate_layer_costs",
    "logit_change_with_mask",
    "cache_representation_similarity",
    "centered_kernel_alignment",
    "compare_model_representations",
    "flatten_representation",
    "representation_cosine_similarity",
    "embedding_anisotropy",
    "embedding_norms",
    "embedding_projection_scores",
    "nearest_embedding_neighbors",
    "token_embedding_similarity",
    "block_internal_keys",
    "block_internal_norms",
    "residual_stream_delta_table",
    "module_activation_norms",
    "module_parameter_norms",
    "residual_update_ratio",
    "tensor_norm_summary",
    "activation_kurtosis",
    "channel_outlier_scores",
    "summarize_activation_outliers",
    "attention_mask_summary",
    "attention_receptive_field",
    "causal_mask_violation_count",
    "summarize_attention_receptive_field",
    "attention_diagonal_score",
    "attention_distance",
    "attention_pattern_summary",
    "attention_prefix_mass",
    "attention_previous_token_score",
    "logit_prism_components",
    "summarize_logit_prism",
    "unembed_vector_scores",
    "boost_direction",
    "concept_direction_effect",
    "concept_direction_from_means",
    "erase_direction",
    "patch_module_direction",
    "module_attribution_patching",
    "module_integrated_attribution_patching",
    "summarize_attribution_patching",
    "interchange_intervention_effect",
    "feature_activation_jaccard",
    "feature_correlation_graph",
    "summarize_feature_circuit",
    "sequence_perplexity",
    "sequence_negative_log_likelihood",
    "area_over_curve",
    "faithfulness_summary",
    "spearman_rank_correlation",
    "token_deletion_insertion_curves",
    "explanation_stability",
    "randomization_rank_baseline",
    "stability_summary",
    "greedy_module_circuit",
    "module_recovery_scores",
    "summarize_module_circuit",
    "MoETarget",
    "capture_moe_router_logits",
    "expert_usage_from_logits",
    "find_moe_targets",
    "summarize_moe_router_usage",
    "render_interpretability_html_report",
    "save_interpretability_html_report",
    "token_trigger_append_scan",
    "token_trigger_position_scan",
    "DiffusionInputs",
    "DiffusionModelAdapter",
    "DiffusionPatchResult",
    "DiffusionTraceDataset",
    "DiffusionTracer",
    "PromptTokenAttribution",
    "classifier_logit_score",
    "clip_similarity_score",
    "collect_diffusion_attention_maps",
    "cosine_distance",
    "diffusion_module_patch_sweep",
    "diffusion_attention_phase_summary",
    "get_diffusion_adapter",
    "mse_distance",
    "patch_denoiser_latents",
    "patch_diffusion_module_outputs",
    "prompt_counterfactual_delta",
    "prompt_token_occlusion_importance",
    "resolve_diffusion_score",
    "summarize_diffusion_steps",
    "summarize_diffusion_trace_dataset",
    "summarize_prompt_token_attribution",
    "tensor_mean_score",
    "tensor_region_score",
    "token_region_attribution",
    "trace_diffusion_generation",
    "trace_prompt_dataset",
    "MultimodalInputs",
    "MultimodalModelAdapter",
    "MultimodalTracer",
    "get_multimodal_adapter",
    "trace_multimodal_forward",
]
