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
