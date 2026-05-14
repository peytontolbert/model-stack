Mechanistic interpretability & probes

This module provides lightweight, composable tools for model interpretability across the stack. It complements `viz` and integrates naturally with models in `model/` and blocks in `blocks/`.

Highlights
- Logit lens over residual streams per layer
- Runtime-aware model adapter for `CausalLM`, `EncoderModel`, and `EncoderDecoderLM`
- Activation tracing and caching via robust forward hooks plus canonical interpret surfaces
- Linear probes (classification/regression) on cached features
- Output/head/neuron/path patching sweeps and restoration analysis
- Integrated gradients, grad×input, occlusion, and direct component/head/neuron/SAE attribution
- Attention weights/entropy extraction captured from the real runtime attention path
- Text-to-image diffusion tracing, prompt attribution, causal patching, batch summaries, and objective helpers
- Structural multimodal/VLM tracing for vision encoder, projector, and language model components
- Faithfulness metrics and greedy circuit discovery for checking whether explanations predict behavior
- Stability/randomization baselines, MoE router usage, HTML reports, and trigger localization scans
- Representation similarity/drift analysis and concept-direction erasure/boosting
- Attribution patching, feature-correlation circuit graphs, and diffusion cross-attention phase summaries
- Standalone model-stack diagnostics for embeddings, norms, residual update ratios, and attention masks
- Generation-time logit traces for entropy, margins, top-k alternatives, and decoded-token trajectories
- Deeper stack diagnostics for block internals, attention patterns, logit-prism components, and activation outliers
- Tuned-lens primitives, interchange interventions, and integrated attribution patching
- MLP lens: project MLP outputs through `lm_head` to get per-layer token predictions
- Attention head ablation (EagerAttention) via context manager
- Probe dataset builders, train/validation splits, and dataset-scale activation mining
- Reporting helpers for patch sweeps, path grids, probe datasets, and SAE training

Quick Start (Python API)
```python
import torch
from interpret import ActivationTracer, ModelInputs, ProbeFeatureSlice, attention_snapshot_for_layer, build_probe_dataset, component_logit_attribution, logit_lens

# causal model
input_ids = torch.tensor([[1, 2, 3, 4]])

# Logit lens
out = logit_lens(model, input_ids, topk=5)
for layer, (idx, val) in sorted(out.items()):
    print(layer, idx.tolist(), val.tolist())

# Activation tracing with canonical interpret surfaces
tracer = ActivationTracer(model)
tracer.add_interpret_surfaces()
with tracer.trace() as cache:
    _ = model(input_ids)
    h0 = cache.get("blocks.0.resid_post")      # [B, T, D]
    a0 = cache.get("blocks.0.attn.attn_probs") # [B, H, T, S]
    m0 = cache.get("blocks.0.mlp.mlp_mid")     # [B, T, D_ff]

# Direct residual component attribution
scores = component_logit_attribution(model, input_ids)
print(scores)

# Runtime-faithful attention capture
snap = attention_snapshot_for_layer(model, input_ids, layer_index=0)
print(snap.probs.shape, snap.head_out.shape)

# Probe dataset construction from aligned cached activations
ds = build_probe_dataset(cache, [ProbeFeatureSlice("blocks.0.resid_post")], input_ids=input_ids)
train_ds, val_ds = split_probe_dataset(ds, val_fraction=0.2, stratify=True)
print(summarize_probe_dataset(train_ds))

# Seq2seq example: decoder cross-attention weights
seq_inputs = ModelInputs.encoder_decoder(
    enc_input_ids=torch.tensor([[1, 2, 3, 4]]),
    dec_input_ids=torch.tensor([[5, 6, 7]]),
)
cross = attention_snapshot_for_layer(seq2seq_model, layer_index=0, stack="decoder", kind="cross", **seq_inputs.__dict__)
print(cross.probs.shape)

# Text-to-image diffusion example: Diffusers-style pipeline
from interpret import DiffusionTracer, patch_denoiser_latents, prompt_token_occlusion_importance, token_region_attribution

tracer = DiffusionTracer(pipe)
image, cache, steps = tracer.trace_generation("red cube on a marble table", num_inference_steps=20)
maps = [value for key, value in cache.items() if key.endswith(".attn_probs")]
heatmaps = token_region_attribution(maps, ["red", "cube", "marble", "table"])
scores = prompt_token_occlusion_importance(pipe, "red cube on a marble table")
patch = patch_denoiser_latents(
    pipe,
    clean_prompt="red cube on a marble table",
    corrupted_prompt="blue cube on a marble table",
    patch_steps=[0, 1, 2],
    num_inference_steps=20,
)

# Multimodal/VLM example
from interpret import MultimodalInputs, MultimodalTracer

vlm_tracer = MultimodalTracer(vlm)
out, vlm_cache = vlm_tracer.trace_forward(MultimodalInputs(input_ids=input_ids, pixel_values=pixel_values))
```

CLI Usage
```bash
# Provide a model factory callable: module:function -> nn.Module
python -m interpret.cli logit-lens \
  --model mypkg.models:load_small_lm \
  --tokens 1,2,3,4 \
  --topk 10

python -m interpret.cli causal-trace \
  --model mypkg.models:load_small_lm \
  --clean 1,2,3,4 \
  --corrupted 1,999,3,4 \
  --points blocks.0,blocks.3 \
  --topk 10

python -m interpret.cli path-sweep \
  --model mypkg.models:load_small_lm \
  --clean 1,2,3,4 \
  --corrupted 1,999,3,4 \
  --sources blocks.0,blocks.1 \
  --receivers blocks.2.attn,blocks.3.mlp \
  --topk 5

python -m interpret.cli diffusion-trace \
  --model mypkg.pipelines:load_t2i_pipe \
  --prompt "red cube on a marble table" \
  --steps 20

python -m interpret.cli diffusion-occlude \
  --model mypkg.pipelines:load_t2i_pipe \
  --prompt "red cube on a marble table" \
  --steps 20 \
  --topk 10

python -m interpret.cli diffusion-patch-latents \
  --model mypkg.pipelines:load_t2i_pipe \
  --clean-prompt "red cube on a marble table" \
  --corrupted-prompt "blue cube on a marble table" \
  --steps 20 \
  --patch-steps 0,1,2

python -m interpret.cli faithfulness \
  --model mypkg.models:load_small_lm \
  --tokens 1,2,3,4 \
  --scores 0.1,0.9,0.2,0.4

python -m interpret.cli module-circuit \
  --model mypkg.models:load_small_lm \
  --clean 1,2,3,4 \
  --corrupted 1,999,3,4 \
  --candidates blocks.0,blocks.1,blocks.0.attn,blocks.1.mlp \
  --k 3

python -m interpret.cli attribution-patch \
  --model mypkg.models:load_small_lm \
  --clean 1,2,3,4 \
  --corrupted 1,999,3,4 \
  --candidates blocks.0,blocks.1,blocks.0.attn,blocks.1.mlp

python -m interpret.cli trigger-append \
  --model mypkg.models:load_small_lm \
  --tokens 1,2,3,4 \
  --triggers 999,1000,1001

python -m interpret.cli compare-representations \
  --model-a mypkg.models:load_small_lm \
  --model-b mypkg.models:load_small_lm_checkpoint2 \
  --tokens 1,2,3,4 \
  --modules blocks.0,blocks.1

python -m interpret.cli diffusion-phase \
  --model mypkg.pipelines:load_t2i_pipe \
  --prompt "red cube on a marble table" \
  --steps 20
```

API Overview
- `ActivationTracer`: register forward hooks by module name or predicate, capture outputs to an `ActivationCache`.
- `ModelInputs` / `ModelAdapter`: normalize causal, encoder, and encoder-decoder interpret calls.
- `logit_lens(model, input_ids, ...)`: compute top-k tokens when projecting intermediate hidden states through `lm_head`.
- `fit_linear_probe(x_train, y_train, ...)`: train a simple linear probe on cached features.
- `causal_trace_restore_fraction(...)`: replace specified module outputs from clean run into corrupted run and measure recovery.
- `head_patch_sweep(...)`, `block_output_patch_sweep(...)`, `mlp_neuron_patch_sweep(...)`, `path_patch_effect(...)`, `path_patch_sweep(...)`: patch sweeps and path analysis.
- `component_logit_attribution(...)`, `head_logit_attribution(...)`, `mlp_neuron_logit_attribution(...)`, `sae_feature_logit_attribution(...)`: direct attribution surfaces.
- `build_probe_dataset(...)`, `split_probe_dataset(...)`, `summarize_probe_dataset(...)`: aligned feature construction plus train/validation split and dataset summaries for probes.
- `sae_feature_dashboard(...)`, `search_feature_activations(...)`, `summarize_sae_training(...)`: dataset-scale mining and reporting helpers.
- `DiffusionTracer(pipe).trace_generation(...)`: capture denoiser latent inputs/outputs, timesteps, and likely cross-attention modules from a text-to-image pipeline.
- `prompt_token_occlusion_importance(...)`, `prompt_counterfactual_delta(...)`, `token_region_attribution(...)`: prompt sensitivity and token-to-image-region attribution helpers.
- `patch_denoiser_latents(...)`, `patch_diffusion_module_outputs(...)`, `diffusion_module_patch_sweep(...)`: diffusion causal intervention utilities with recovery scoring.
- `trace_prompt_dataset(...)`, `summarize_diffusion_trace_dataset(...)`: batch diffusion trace summaries across prompt sets.
- `tensor_region_score(...)`, `classifier_logit_score(...)`, `clip_similarity_score(...)`: objective helpers for scoring generated images/latents.
- `MultimodalTracer(...)`, `MultimodalModelAdapter(...)`: structural vision-language model component tracing.
- `token_deletion_insertion_curves(...)`, `faithfulness_summary(...)`, `spearman_rank_correlation(...)`: faithfulness checks for token explanations.
- `module_recovery_scores(...)`, `greedy_module_circuit(...)`, `summarize_module_circuit(...)`: automated module-level circuit discovery with activation patching.
- `explanation_stability(...)`, `randomization_rank_baseline(...)`, `stability_summary(...)`: compare explanations against perturbations and random baselines.
- `find_moe_targets(...)`, `capture_moe_router_logits(...)`, `expert_usage_from_logits(...)`: MoE router/expert usage interpretability.
- `render_interpretability_html_report(...)`, `save_interpretability_html_report(...)`: lightweight HTML report artifacts.
- `token_trigger_append_scan(...)`, `token_trigger_position_scan(...)`: safety-oriented trigger localization helpers.
- `centered_kernel_alignment(...)`, `compare_model_representations(...)`, `cache_representation_similarity(...)`: representation similarity and checkpoint drift helpers.
- `concept_direction_from_means(...)`, `erase_direction(...)`, `boost_direction(...)`, `concept_direction_effect(...)`: concept direction intervention helpers.
- `module_attribution_patching(...)`, `summarize_attribution_patching(...)`: first-order attribution patching for ranking many modules before exact patching.
- `feature_correlation_graph(...)`, `feature_activation_jaccard(...)`: sparse feature-circuit summaries over SAE/probe activations.
- `diffusion_attention_phase_summary(...)`, `collect_diffusion_attention_maps(...)`: timestep-phase summaries for text-to-image cross-attention.
- `embedding_norms(...)`, `nearest_embedding_neighbors(...)`, `embedding_projection_scores(...)`: embedding-table diagnostics.
- `tensor_norm_summary(...)`, `module_parameter_norms(...)`, `module_activation_norms(...)`, `residual_update_ratio(...)`: norm and residual-health diagnostics.
- `attention_mask_summary(...)`, `attention_receptive_field(...)`, `causal_mask_violation_count(...)`: attention-mask and receptive-field diagnostics.
- `generation_logit_trace(...)`, `summarize_generation_trace(...)`: standalone greedy decode diagnostics.
- `block_internal_norms(...)`, `residual_stream_delta_table(...)`: block-level residual/branch norm diagnostics.
- `attention_pattern_summary(...)`, `attention_distance(...)`, `attention_previous_token_score(...)`: head pattern diagnostics.
- `logit_prism_components(...)`, `unembed_vector_scores(...)`: unembedding/logit-prism component scoring.
- `channel_outlier_scores(...)`, `activation_kurtosis(...)`, `summarize_activation_outliers(...)`: activation outlier diagnostics useful for quantization and instability checks.
- `AffineTunedLens(...)`, `tuned_lens_logits(...)`, `tuned_lens_topk(...)`: tuned-lens style translated vocabulary projections.
- `interchange_intervention_effect(...)`: causal-abstraction style source-to-base activation interchange.
- `module_integrated_attribution_patching(...)`: integrated-gradient attribution patching over activation deltas.

Notes
- Works with causal, encoder-only, and encoder-decoder runtime stacks through `ModelAdapter`.
- Attention capture and head patching preserve the native runtime attention path; masks and resolved tensors come from the real forward call.
- Diffusion helpers are dependency-optional and structural: they target Diffusers-style pipelines but do not import Diffusers directly.
- Multimodal helpers are also structural and work best when models expose common component names like `vision_tower`, `mm_projector`, and `language_model`.
- The lightweight helpers `flatten_features(...)` and `targets_from_tokens(...)` now enforce alignment instead of silently stacking mismatched rows.
- Features stored in `ActivationCache` can be serialized via `save_pt(path)` and reloaded.
