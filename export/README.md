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
- `runtime/bitnet_linear.wgsl`
- `modelcard.json`

This target is for Safari/WebGPU and other browser runtimes. It does not export
a fake ONNX BitNet op. The v1 browser path requires `spin=False` and
`quant_weight_opt=none`; AWQ/pre-scale and spin transforms need dedicated browser
epilogues before they can be enabled.

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
