# Model Stack

Model Stack is a transformer runtime and training stack built for measurable
systems work: custom CUDA kernels, paged KV cache, packed BitNet/ternary
execution, quantization, export, and H100 benchmarking.

The core advantage is that performance work lands in a real runtime instead of
staying as isolated kernels. The same repository contains model assembly,
generation, native dispatch, compression/export, tests, and restartable H100
recipes, so an optimization can be trained, packed, exported, benchmarked, and
served without rewriting glue code.

## Why Model Stack

- Beats Hugging Face Transformers BitNet module execution on H100 for the
  focused projection and `lm_head` benchmark.
- Keeps packed BitNet/ternary weights in runtime-owned formats instead of
  paying Python/module overhead at every layer.
- Uses native paged KV cache and a decode-side SDPA bridge to keep long-context
  decode fast at 1024/2048/4096 context.
- Provides restartable 8xH100 Parameter Golf scripts for training runtime-row
  ternary artifacts and exporting them into the Model Stack runtime bundle.
- Keeps CUDA kernels, runtime policy, export verification, and source-surface
  tests in one tree so benchmark wins are reproducible.

## H100 Benchmark Wins

Focused BitNet module benchmark against `microsoft/bitnet-b1.58-2B-4T` through
Transformers:

| Module | Transformers BitNet | Model Stack | Speedup |
| --- | ---: | ---: | ---: |
| `q_proj` | 0.0771 ms | 0.0413 ms | 1.87x |
| `o_proj` | 0.0779 ms | 0.0411 ms | 1.89x |
| `gate_proj` | 0.0774 ms | 0.0408 ms | 1.90x |
| `down_proj` | 0.0801 ms | 0.0432 ms | 1.85x |
| `lm_head` | 0.2252 ms | 0.1238 ms | 1.82x |

Native 4-layer decode benchmark with packed BitNet runtime:

| Context / batch | Dense BF16 reference | Model Stack BitNet | Speedup |
| --- | ---: | ---: | ---: |
| 1024 / batch 4 | 2.411 ms | 2.336 ms | 1.03x |
| 2048 / batch 2 | 2.595 ms | 2.573 ms | 1.01x |
| 4096 / batch 1 | 1.656 ms | 1.619 ms | 1.02x |

Paged decode attention on the H100 path also moved from the old custom paged
kernel path to the SDPA-bridged native-cache path, reducing the 4096-context
4-layer decode run from roughly 33.75 ms to 1.62 ms.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `specs/` | versioned configs, tensor specs, export schemas |
| `tensor/` | stateless tensor utilities: init, masks, RoPE, numerics |
| `attn/` | attention modules, masks, KV cache contracts |
| `blocks/` | transformer block wiring, norms, MLPs, residual policies |
| `model/` and `runtime/` | model assembly, generation, native runtime dispatch |
| `kernel/` and `runtime/csrc/` | CUDA/C++ kernels and extension bindings |
| `compress/` | quantization, BitNet, LoRA, pruning, compression deltas |
| `data/` and `corpus/` | tokenization, sharding, datasets, corpus tooling |
| `train/` and `dist/` | training loops, optimizers, distributed launch helpers |
| `eval/` and `tests/` | correctness, source-surface tests, perf harnesses |
| `serve/` | decode loops, sampling, paged cache serving pieces |
| `export/` | TorchScript/ONNX/export flows and metadata |
| `governance/` | SBOMs, checksums, receipts, lineage, model cards |
| `examples/` | reproducible demos and operational recipes |

## Setup

Create an environment with PyTorch and the dependencies needed for the part of
the stack you are using, then run tests or examples from the repo root.

```bash
python -m pytest tests/test_runtime_paged_attention_source_surface.py -q
python examples/00_tiny_lm/run.py
```

The repository intentionally keeps large generated artifacts out of Git. Model
outputs, packed bundles, logs, and training artifacts should live under ignored
paths such as `artifacts/`, `checkpoints/`, or machine-local storage.

## Native CUDA Build

The C++/CUDA runtime extension is optional. Build it explicitly when you need
native kernels:

```bash
MODEL_STACK_BUILD_NATIVE=1 \
MODEL_STACK_BUILD_CUDA=1 \
MODEL_STACK_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX" \
python setup.py build_ext --inplace
```

Useful build variables:

- `MODEL_STACK_CUDA_ARCH_LIST`: repo-local alias for `TORCH_CUDA_ARCH_LIST`.
  If `TORCH_CUDA_ARCH_LIST` is already set, PyTorch's value wins.
- `MODEL_STACK_MAX_JOBS`: repo-local alias for PyTorch/Ninja `MAX_JOBS`.
- `MODEL_STACK_USE_NINJA=0`: use the slower setuptools fallback.
- `MODEL_STACK_CUTLASS_PATH=/path/to/cutlass`: enable optional CUTLASS-backed
  kernels.

For H100-only work, build with:

```bash
MODEL_STACK_BUILD_NATIVE=1 \
MODEL_STACK_BUILD_CUDA=1 \
MODEL_STACK_CUDA_ARCH_LIST="9.0" \
MODEL_STACK_MAX_JOBS=4 \
python setup.py build_ext --inplace
```

## Runtime Flags

Common dispatch controls:

- `MODEL_STACK_DISABLE_CUDA_EMBEDDING_NATIVE=1`: force PyTorch embedding.
- `MODEL_STACK_DISABLE_CUDA_ADD_LAYER_NORM_NATIVE=1`: force PyTorch add+norm.
- `MODEL_STACK_ENABLE_CUDA_ADD_LAYER_NORM_NATIVE=1`: force native add+norm for
  experiments.
- `MODEL_STACK_PREFER_NATIVE_SM80_INFERENCE_ATTENTION=1`: opt into the SM80/Ada
  native prefill attention lane.
- `MODEL_STACK_ENABLE_ATTENTION_PREFILL_SM80_FLASH=1`: opt into the local
  experimental SM80 FlashAttention-style prefill lane.
- `MODEL_STACK_PAGED_DECODE_SDPA_MAX_LENGTH=8192`: max decode cache length for
  the paged decode SDPA bridge.
- `MODEL_STACK_DISABLE_PAGED_DECODE_SDPA_BRIDGE=1`: disable the paged decode
  SDPA bridge and use the older paged decode kernel/fallback path.

## Examples

See [examples/README.md](examples/README.md) for runnable demos.

Important current examples:

- `examples/00_tiny_lm/`: tiny LM training smoke
- `examples/02_int8_export/`: tiny export and quantization smoke
- `examples/03_fsdp_8gpu/`: multi-GPU launcher notes
- `examples/11_compress_quantize/`: weight-only quantization demo
- `examples/13_parameter_golf_h100/`: restartable 8xH100 Parameter Golf
  ternary training/export recipe

Parameter Golf H100 quick start:

```bash
cd /root/transformer_10_h100
bash examples/13_parameter_golf_h100/build_runtime_h100.sh

PG_DIR=/root/parameter_golf \
PG_PRESET=runtime_row_1024 \
bash examples/13_parameter_golf_h100/run_pg_8xh100.sh
```

Export the trained ternary artifact:

```bash
PG_DIR=/root/parameter_golf \
ARTIFACT=/root/parameter_golf/final_model.ternary.ptz \
OUT=/root/transformer_10_h100/artifacts/parameter_golf/final_model.runtime_bitnet.pt \
bash examples/13_parameter_golf_h100/export_runtime_bundle.sh
```

## Minimal Usage

```python
from specs.config import ModelConfig
from runtime.factory import build_causal_lm

cfg = ModelConfig(
    d_model=512,
    n_heads=8,
    n_layers=8,
    d_ff=2048,
    vocab_size=32000,
    dtype="bf16",
)

model = build_causal_lm(cfg, block="llama")
```

For a complete smoke run:

```bash
python example.py
```

## Governance Utilities

The `governance/` package can emit release metadata for artifacts:

```bash
python -m governance sbom --out artifacts/SBOM.spdx.json --name "$MODEL_NAME"
python -m governance sign --files artifacts/model.onnx artifacts/tokenizer.json \
  --out artifacts/CHECKSUMS.sha256 --key path/to/ed25519.key
python -m governance receipt --artifacts artifacts/model.onnx \
  --out artifacts/RECEIPT.json --metadata train/metadata.json
python -m governance lineage --metadata train/metadata.json --out artifacts/LINEAGE.dot
python -m governance card artifacts/model.onnx --metadata train/metadata.json \
  --sbom artifacts/SBOM.spdx.json --out artifacts/MODEL_CARD.md
python -m governance verify artifacts/model.onnx
```

## Compatibility Notes

- `specs/` schema major-version changes are expected to ripple outward.
- Kernel implementations can change as long as runtime names and semantics are
  preserved.
- Checkpoint loaders should gate behavior on model-card/checkpoint versions and
  provide migrations for incompatible layouts.
- Serving entry points should keep generation and sampler interfaces stable
  across minor changes.
