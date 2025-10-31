SDK, CLIs, templates, and integration glue

pipx/CLI entrypoints (init, train, export, serve), project templates (single-node, multi-node), config validators, env bootstrap.

Thin wrappers around dist/train/export/serve.

# CLIs (examples)
init --template=llm-small
train --config cfgs/sft.yaml
export --checkpoint ckpt.pt --target onnx
serve --artifact artifacts/model.onnx

## End-to-end quickstart

```bash
python -m pack.cli e2e \
  --d-model 256 --n-heads 8 --n-layers 4 --d-ff 1024 --vocab-size 32000 \
  --steps 50 --seq-len 256 --batch-size 8 --log-dir .viz \
  --interpret-probes \
  --export-target onnx --export-opset 19 --export-outdir artifacts/

# Open dashboard
python -m viz.cli render --log-dir .viz --title "E2E Training"
```

This builds a model via `model.factory.build_model(cfg)`, runs a toy training loop
with `viz` logging (and optional `interpret` activation probes), then exports an artifact.
