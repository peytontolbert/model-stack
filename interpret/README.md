Mechanistic interpretability & probes

This module provides lightweight, composable tools for model interpretability across the stack. It complements `viz` and integrates naturally with models in `model/` and blocks in `blocks/`.

Highlights
- Logit lens over residual streams per layer
- Activation tracing and caching via robust forward hooks
- Linear probes (classification/regression) on cached features
- Causal tracing via activation patching (resample/restore analysis)
- Integrated gradients token attributions
- Attention weights/entropy extraction leveraging `attn.eager` and `tensor.*` utils
- MLP lens: project MLP outputs through `lm_head` to get per-layer token predictions
- Attention head ablation (EagerAttention) via context manager
- Feature flattening utilities to prepare data for probes

Quick Start (Python API)
```python
import torch
from interpret import ActivationTracer, logit_lens, integrated_gradients_tokens

# model: instance of model.CausalLM (or any nn.Module with .embed/.lm_head)
input_ids = torch.tensor([[1,2,3,4]])

# Logit lens
out = logit_lens(model, input_ids, topk=5)
for layer, (idx, val) in sorted(out.items()):
    print(layer, idx.tolist(), val.tolist())

# Activation tracing (capture block outputs)
tracer = ActivationTracer(model)
tracer.add_block_outputs()
with tracer.trace() as cache:
    _ = model(input_ids)
    h0 = cache.get("blocks.0")  # [B, T, D]

# Integrated gradients (token-level attributions)
scores = integrated_gradients_tokens(model, input_ids, steps=32)
print(scores.tolist())

# Attention weights and entropy for a specific layer
from interpret import attention_weights_for_layer, attention_entropy_for_layer
probs = attention_weights_for_layer(model, input_ids, layer_index=0)
ent = attention_entropy_for_layer(model, input_ids, layer_index=0)

# MLP lens (feedforward contributions)
from interpret import mlp_lens
mlp_out = mlp_lens(model, input_ids, topk=5)
print(mlp_out)

# Head ablation (temporarily zero specific heads)
from interpret import ablate_attention_heads
with ablate_attention_heads(model, {0: [0,1], 3: [2]}):
  _ = model(input_ids)

# Features for probes
from interpret import ActivationTracer, flatten_features, FeatureSlice
tr = ActivationTracer(model)
tr.add_block_outputs()
with tr.trace() as cache:
  _ = model(input_ids)
X = flatten_features(cache, [FeatureSlice("blocks.0"), FeatureSlice("blocks.1.mlp", time_slice=slice(-1,None))])
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
```

API Overview
- `ActivationTracer`: register forward hooks by module name or predicate, capture outputs to an `ActivationCache`.
- `logit_lens(model, input_ids, ...)`: compute top-k tokens when projecting intermediate hidden states through `lm_head`.
- `fit_linear_probe(x_train, y_train, ...)`: train a simple linear probe on cached features.
- `causal_trace_restore_fraction(...)`: replace specified module outputs from clean run into corrupted run and measure recovery per token id.
- `integrated_gradients_tokens(...)`: token-level attributions using zero baseline and path integration.

Notes
- Works out-of-the-box with `model.CausalLM` and `blocks.*` transformer blocks.
- For non-standard models, ensure there is an `embed` layer and `lm_head.weight` (or tied `embed.weight`) for logit lens.
- Features stored in `ActivationCache` can be serialized via `save_pt(path)` and reloaded.