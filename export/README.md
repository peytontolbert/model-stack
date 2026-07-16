# export

Full(er) exporter with ONNX / TorchScript targets, optional weight-only INT8, and delta artifacts.

## CLI

```bash
# From a saved model directory (config + weights)
python -m export.cli \
  --model-dir ckpts/my_model \
  --target onnx --opset 19 --dynamic-axes \
  --quantize int8 \
  --outdir artifacts/

# TorchScript
python -m export.cli --model-dir ckpts/my_model --target torchscript --outdir artifacts/
```

Outputs:
- artifacts/model.onnx or artifacts/model.ts
- artifacts/modelcard.json (format, sha256, etc.)
- artifacts/delta.pt (compression delta if available)

## Browser BitNet

```bash
python -m export.cli \
  --model-dir ckpts/my_model \
  --target browser-bitnet \
  --quantize bitnet \
  --quant-weight-opt none \
  --outdir artifacts/browser-bitnet/
```

Outputs:

- `manifest.json`
- `layers/*.bin` packed ternary weights, scales, layout headers, offsets, and bias tensors
- `runtime/bitnet_webgpu.js`
- `runtime/bitnet_wasm_runtime.js`
- `runtime/model_stack_bitnet_wasm.js`
- `runtime/model_stack_bitnet_wasm_bg.wasm`
- `runtime/bitnet_linear.wgsl`
- `modelcard.json`

This target is for browser runtimes with WebGPU first and packed BitNet WASM as
the compatibility fallback. It does not export a fake ONNX BitNet op or expand
packed weights to dense ONNX weights. The v1 browser path requires `spin=False`
and `quant_weight_opt=none`; AWQ/pre-scale and spin transforms need dedicated
browser epilogues before they can be enabled.

When the model config enables auxiliary heads, the browser manifest also records
their execution surface:

- retrieval embeddings through `retrieval_query_head` and `retrieval_doc_head`
- agent intent logits through `agent_intent_head`
- scalar agent policy heads such as `should_answer`, `should_clarify`,
  `memory_write`, and `retrieval_coverage`

The browser encoder-decoder runtime consumes these heads from pooled encoder
states. Export should keep them as explicit named linears so WebGPU/WASM can
route them through the same packed BitNet layer loader instead of adding
task-specific JavaScript math.

## Programmatic

```python
from specs.export import ExportConfig
from export.exporter import export_from_dir

cfg = ExportConfig(target="onnx", opset=19, quantize="int8", outdir="artifacts/")
out = export_from_dir("ckpts/my_model", cfg)
print(out)
```

Notes:
- INT8 uses weight-only replacements via `compress.apply.apply_compression`.
- ONNX exports inputs `input_ids[B,T]`, `attn_mask[B,T]`; dynamic axes optional.
- TensorRT path is a placeholder plan file + model card; use your TRT toolchain to build.
