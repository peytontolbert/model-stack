# transformer_10 Compression And Quantization Runtime Spec

This document defines the target C++/CUDA ownership for `compress/` and the quantization-related pieces of `tensor/quant_utils.py` and `attn/quant.py`.

## 1. Scope

The runtime must own:

- weight-only and activation-aware quantization
- quant metadata and packed formats
- LoRA application and merge paths
- pruning masks and sparse transforms
- distillation hooks and losses
- exportable compression deltas

## 2. Compression Manager

Required C++ objects:

- `CompressionManager`
- `CompressionConfig`
- `CompressionDelta`
- `CompressionRegistry`

Responsibilities:

- parse compression config
- apply transforms during model build or checkpoint load
- expose merge/unmerge operations
- persist metadata with checkpoints and exports

## 3. Quantization

## Required formats

- INT8 per-channel weight-only
- INT8 groupwise or blockwise variants later
- FP8 metadata tracking
- NF4 / 4-bit weight formats later if adopted

## Required components

- `QuantizedWeightFormat`
- `Quantizer`
- `QuantCalibration`
- `QuantizedLinear`
- `QuantMeta`
- `ScalePack`

## Ownership split

- packing, metadata, calibration policy -> C++
- dequantize, scale-apply, activation clip, quantized matmul glue -> CUDA kernels and library wrappers

## File mapping

| Current file | Target implementation |
|---|---|
| `compress/quantization.py` | `t10::compress::Quantizer`, `QuantizedLinear` |
| `tensor/quant_utils.py` | `t10::compress::QuantMeta`, packing utilities |
| `attn/quant.py` | quantized attention path and dequant glue |

## 4. LoRA

Required components:

- `LoRAAdapter`
- `LoRARegistry`
- `LoRAMerger`
- `LoRAState`

Required behavior:

- inject adapters into selected linear modules
- forward path can be unfused first, fused later
- merge and unmerge at runtime or export time
- save and load LoRA-only deltas

Important rule:

- LoRA is not a Python-only adapter trick in the long-term runtime
- it becomes a first-class model-stack feature

## 5. Pruning And Sparse Compression

Required components:

- `Pruner`
- `PruningMask`
- `SparseTransform`
- `PruningDelta`

Responsibilities:

- magnitude and movement-style pruning metadata
- runtime mask application when retained
- optional conversion to sparse kernel-friendly layouts

## 6. Distillation

Required components:

- `DistillationTeacher`
- `DistillationHooks`
- `DistillationLoss`

Responsibilities:

- teacher logits collection
- intermediate feature capture
- KD / MSE matching losses

Distillation belongs to the training runtime, but compression config must own its metadata and activation-capture hooks.

## 7. KV Cache Compression

`compress/kv_cache.py` should not remain a separate Python-owned paging system.

Target:

- merge cache paging and compaction logic into `t10::attention::AttentionCache`
- keep any compression or compaction policy as part of cache manager configuration

## 8. Export And Delta Format

Required exportable artifacts:

- LoRA-only delta
- pruning masks
- quant scales and format metadata
- combined compression manifest

`compress/export.py` maps to:

- `CompressionDeltaWriter`
- `CompressionDeltaReader`
- `CompressionDeltaApplier`

## 9. Migration Order

1. weight-only INT8
2. LoRA injection and merge/unmerge
3. compression delta format
4. pruning metadata and optional sparse layouts
5. distillation hooks
6. advanced quant formats and quantized attention paths

## 10. Definition Of Coverage

Compression coverage is complete when:

- every compression mode in the repository has an owned runtime representation
- checkpoint and export metadata are explicit
- LoRA, quantization, pruning, and distillation no longer depend on Python module replacement for core behavior
