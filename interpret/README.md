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

Notes
- Works with causal, encoder-only, and encoder-decoder runtime stacks through `ModelAdapter`.
- Attention capture and head patching preserve the native runtime attention path; masks and resolved tensors come from the real forward call.
- The lightweight helpers `flatten_features(...)` and `targets_from_tokens(...)` now enforce alignment instead of silently stacking mismatched rows.
- Features stored in `ActivationCache` can be serialized via `save_pt(path)` and reloaded.
