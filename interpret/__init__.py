from .activation_cache import ActivationCache, CaptureSpec
from .tracer import ActivationTracer
from .logit_lens import logit_lens
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
from .metrics.sequence import sequence_perplexity, sequence_negative_log_likelihood
from .attribution.direct import (
    component_logit_attribution,
    head_logit_attribution,
    mlp_neuron_logit_attribution,
    sae_feature_logit_attribution,
)
from .reporting import summarize_patch_sweep, summarize_path_patch_sweep, summarize_probe_training_split, summarize_sae_training
from .probes.dataset import ProbeDataset, ProbeDatasetSummary, ProbeFeatureSlice, build_probe_dataset, split_probe_dataset, summarize_probe_dataset

__all__ = [
    "ActivationCache",
    "CaptureSpec",
    "ActivationTracer",
    "logit_lens",
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
    "sequence_perplexity",
    "sequence_negative_log_likelihood",
]

