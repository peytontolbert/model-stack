# transformer_10 Repository Coverage Audit For C++/CUDA Migration

Historical note:

This audit identified the doc gaps that prevented a claim of full repository coverage at the time it was written.
The follow-up docs listed in `README.md`, especially:

- `transformer10-module-target-state-matrix.md`
- `transformer10-cpp-model-stack-api-spec.md`
- `transformer10-complete-tensor-math-function-inventory.md`
- `transformer10-training-backward-cuda-spec.md`
- `transformer10-data-checkpoint-tokenizer-cpp-spec.md`
- `transformer10-serving-engine-cpp-spec.md`
- `transformer10-autotune-eval-benchmark-spec.md`
- `transformer10-compression-quantization-runtime-spec.md`
- `transformer10-noncore-systems-targets.md`
- `transformer10-full-repository-scope-closure.md`

were added specifically to close the gaps recorded here.

This audit answers a narrow question:

Can the current research package be treated as 100% complete for migrating the whole `transformer_10` repository into a modular C++/CUDA model-stack?

Answer: no.

The current docs are strong enough to start focused implementation on the core inference path, but they are not yet comprehensive enough to claim full-repository migration coverage. The remaining gap is not CUDA knowledge. The gap is explicit ownership: every Python module needs a declared target state before migration starts.

Target states are:

- `cuda_kernel`: GPU tensor execution implemented as handwritten CUDA, Triton, Fuser-generated code, or a maintained CUDA library wrapper.
- `cpp_runtime`: C++ owns configuration, object lifecycle, planning, scheduling, checkpointing, serving, training control, validation, or file/data systems.
- `python_binding`: Python remains only as API compatibility over C++/CUDA.
- `python_reference`: Python remains as test/reference/demo code, not production execution.
- `remove_or_merge`: module is superseded by the C++/CUDA stack and should be deleted or merged.
- `defer`: intentionally outside the first migration wave, with an explicit reason.

Coverage is complete only when every module has one of those states plus an intended target component.

## 1. Current Explicit Coverage

This count checks whether a non-`__init__` Python module is explicitly named by full path or filename in the current `docs/research` package.

| Package | Explicitly covered | Missing explicit target docs | Assessment |
|---|---:|---:|---|
| `attn/` | 11 / 14 | 3 | mostly covered, but backend/interface wrappers need classification |
| `autotune/` | 0 / 6 | 6 | not covered enough |
| `blocks/` | 15 / 28 | 13 | block families covered conceptually, support modules undercovered |
| `compress/` | 4 / 7 | 3 | core quantization mentioned, LoRA/distill/apply need targets |
| `corpus/` | 0 / 5 | 5 | not covered |
| `data/` | 1 / 4 | 3 | tokenizer mentioned, data pipeline undercovered |
| `dist/` | 2 / 6 | 4 | distributed execution mentioned, launch/checkpoint/data loader undercovered |
| `eval/` | 3 / 12 | 9 | eval concept covered, harness modules undercovered |
| `examples/` | 0 / 3 | 3 | should be reference/demo classification |
| `experiments/` | 0 / 1 | 1 | should be defer or remove classification |
| `export/` | 0 / 2 | 2 | export path not covered |
| `governance/` | 1 / 8 | 7 | metadata/compliance systems not covered |
| `interpret/` | 0 / 5 | 5 | model diagnostics not covered |
| `kernel/` | 1 / 5 | 4 | existing kernel API surface undercovered |
| `model/` | 12 / 18 | 6 | model families covered, support modules undercovered |
| `pack/` | 0 / 1 | 1 | packaging not covered |
| `rag/` | 1 / 3 | 2 | RAG is not covered as product surface |
| `registry/` | 0 / 1 | 1 | remote/local registry target missing |
| `rl/` | 0 / 3 | 3 | training-adjacent RL not covered |
| `safety/` | 0 / 1 | 1 | guard path not covered |
| `serve/` | 4 / 5 | 1 | serving mostly covered, instrumentation classification missing |
| `specs/` | 1 / 6 | 5 | config/spec resolution undercovered |
| `tensor/` | 15 / 30 | 15 | core math covered, utility/math/storage modules undercovered |
| `train/` | 0 / 2 | 2 | training implementation docs still missing |
| `viz/` | 1 / 4 | 3 | visualization target missing |

Repository total:

- 180 non-`__init__` Python modules scanned.
- 72 explicitly named by current research docs.
- 108 not explicitly named.

That does not mean 108 modules must become CUDA kernels. It means 108 modules still lack an explicit migration target.

## 2. Missing Module Targets

## `attn/`

Missing explicit targets:

- `attn/decoding.py`
- `attn/flash.py`
- `attn/interfaces.py`

Required target doc:

- classify `decoding.py` as C++ decode-attention policy or merge into `serve/` decode engine
- classify `flash.py` as compatibility wrapper, external backend adapter, or remove after native attention lands
- classify `interfaces.py` as C++ interface/ABI contract plus Python binding mirror

## `autotune/`

Missing explicit targets:

- `autotune/callbacks.py`
- `autotune/cli.py`
- `autotune/presets.py`
- `autotune/spaces.py`
- `autotune/study.py`
- `autotune/trial.py`

Required target doc:

- C++ autotune database schema
- kernel-plan cache format
- benchmark trial runner
- shape/dtype/device search-space model
- Python CLI as binding only

Autotune is mandatory for a serious CUDA stack because handwritten kernels, cuBLASLt heuristics, Triton variants, and CUDA Graph buckets need persisted shape-specific decisions.

## `blocks/`

Missing explicit targets:

- `blocks/adapters.py`
- `blocks/config.py`
- `blocks/decoder_block.py`
- `blocks/dilated_local_attn_block.py`
- `blocks/encoder_block.py`
- `blocks/gpt_block.py`
- `blocks/init.py`
- `blocks/inspect.py`
- `blocks/policies.py`
- `blocks/registry.py`
- `blocks/schedules.py`
- `blocks/stack.py`
- `blocks/targets.py`

Required target doc:

- C++ block registry
- block config structs
- block stack construction API
- policy objects for norms, residuals, attention masks, position bias, MLP type, adapters, dropout, and train/inference behavior
- introspection/debug metadata API

The block variants are mentioned conceptually, but support modules still need concrete C++ ownership.

## `compress/`

Missing explicit targets:

- `compress/apply.py`
- `compress/distill.py`
- `compress/lora.py`

Required target doc:

- C++ compression registry
- CUDA LoRA application and merge paths
- distillation training path ownership
- quantized weight binding and checkpoint metadata

LoRA is model-stack relevant and cannot stay an implicit PyTorch adapter layer if the repository is meant to own the full runtime.

## `corpus/`

Missing explicit targets:

- `corpus/build.py`
- `corpus/cli.py`
- `corpus/dedup.py`
- `corpus/manifest.py`
- `corpus/pii.py`

Required target doc:

- decide whether corpus tooling remains Python utility code or becomes a C++ data-prep subsystem
- define if PII/dedup/manifest are production model-stack responsibilities or offline tooling

This package probably does not need CUDA, but it needs an explicit `python_reference`, `python_binding`, or `defer` decision.

## `data/`

Missing explicit targets:

- `data/batch.py`
- `data/iterable.py`
- `data/tokenizer.py`

Required target doc:

- C++ tokenizer interface
- batch collation layout
- pinned-memory and async H2D prefetch policy
- packed/ragged batch descriptors shared with CUDA kernels

## `dist/`

Missing explicit targets:

- `dist/checkpoint.py`
- `dist/cli.py`
- `dist/dataloader.py`
- `dist/launch.py`

Required target doc:

- C++ launcher interface
- distributed checkpoint format
- sharded data-loader ownership
- NCCL rank/world topology config
- failure semantics and resume behavior

NCCL collectives are covered, but the surrounding distributed runtime is not.

## `eval/`

Missing explicit targets:

- `eval/bench.py`
- `eval/calibration.py`
- `eval/cli.py`
- `eval/compare.py`
- `eval/latency.py`
- `eval/llama_hf_parity.py`
- `eval/loop.py`
- `eval/report.py`
- `eval/suite.py`

Required target doc:

- C++ benchmark harness
- numerical parity runner
- latency/memory profiler integration
- report schema
- HF/PyTorch reference comparison mode
- Python CLI as binding/reference only

Eval is a migration blocker because replacing PyTorch without a parity harness is not defensible.

## `examples/` And `experiments/`

Missing explicit targets:

- `examples/debug_attention.py`
- `examples/debug_parity.py`
- `examples/debug_single_layer.py`
- `experiments/repo_conditioned_fast_weights.py`

Required target doc:

- classify examples as Python reference/demo wrappers over C++/CUDA
- classify experimental code as defer, remove, or migrate into a formal model-stack feature

## `export/`

Missing explicit targets:

- `export/cli.py`
- `export/exporter.py`

Required target doc:

- C++ export interface
- supported formats
- whether export is still Python-owned tooling or a runtime feature
- mapping between C++ checkpoint/model descriptors and exported artifacts

## `governance/`

Missing explicit targets:

- `governance/__main__.py`
- `governance/card.py`
- `governance/cli.py`
- `governance/lineage.py`
- `governance/receipt.py`
- `governance/sbom.py`
- `governance/signature.py`

Required target doc:

- decide whether governance remains Python tooling
- if retained in the model-stack, define C++ metadata hooks for checkpoints, model cards, signatures, lineage, SBOM, and receipts

This is mostly non-kernel work, but still part of a complete repository product.

## `interpret/`

Missing explicit targets:

- `interpret/activation_cache.py`
- `interpret/cli.py`
- `interpret/logit_diff.py`
- `interpret/logit_lens.py`
- `interpret/tracer.py`

Required target doc:

- C++ activation capture hooks
- CUDA-safe tracing points
- logit lens/logit diff execution ownership
- Python notebooks/CLI as consumers only

Interpretability features need runtime hooks before the Python implementation is removed.

## `kernel/`

Missing explicit targets:

- `kernel/bench.py`
- `kernel/flash.py`
- `kernel/registry.py`
- `kernel/rope.py`

Required target doc:

- C++ kernel registry
- benchmark entrypoint
- kernel selection metadata
- treatment of existing flash/RoPE wrappers once native runtime kernels exist

The implementation docs mention new `runtime/cuda`, but the old `kernel/` package still needs a retirement or compatibility plan.

## `model/`

Missing explicit targets:

- `model/checkpoint.py`
- `model/inspect.py`
- `model/llama_bootstrap.py`
- `model/lm.py`
- `model/prefix_lm.py`
- `model/registry.py`

Required target doc:

- C++ model registry
- checkpoint loader and weight binder
- bootstrap/import flow for Llama/HF models
- model inspection API
- base LM interface and prefix-LM ownership

Core model execution is covered, but model lifecycle and import paths still need explicit design.

## `pack/`

Missing explicit targets:

- `pack/cli.py`

Required target doc:

- package/build artifact strategy for C++/CUDA runtime
- wheel/binary layout if Python bindings remain
- model bundle packaging if this repo exports runnable artifacts

## `rag/`

Missing explicit targets:

- `rag/cli.py`
- `rag/config.py`

Required target doc:

- decide whether RAG is inside the C++/CUDA model-stack boundary
- if yes, define C++ retrieval/generation integration points
- if no, classify as Python application-layer tooling

## `registry/`

Missing explicit targets:

- `registry/client.py`

Required target doc:

- model/artifact registry API target
- checkpoint download/cache ownership
- C++ versus Python boundary

## `rl/`

Missing explicit targets:

- `rl/cli.py`
- `rl/config.py`
- `rl/trainer.py`

Required target doc:

- decide whether RL training is a first-class migration target
- if yes, define C++ trainer loop, rollout data layout, reward/loss kernel needs, and distributed ownership
- if no, mark as defer explicitly

## `safety/`

Missing explicit targets:

- `safety/guard.py`

Required target doc:

- generation guard integration point
- tokenizer/logit/filter ownership
- whether guard remains Python policy code or becomes C++ serving policy

## `serve/`

Missing explicit target:

- `serve/instrumented_generate.py`

Required target doc:

- C++ instrumentation hooks
- NVTX/profiling integration
- request trace schema
- Python instrumentation wrapper status

Serving itself is mostly covered; instrumentation is not.

## `specs/`

Missing explicit targets:

- `specs/config.py`
- `specs/dist.py`
- `specs/ops.py`
- `specs/resolve.py`
- `specs/viz.py`

Required target doc:

- C++ spec schema
- op registry and resolver
- distributed spec validation
- visualization metadata spec
- Python loader/binding compatibility

Specs are foundational because they determine how the C++ runtime preserves the repository's configurability.

## `tensor/`

Missing explicit targets:

- `tensor/checkpoint.py`
- `tensor/debug.py`
- `tensor/einsum.py`
- `tensor/export_safe.py`
- `tensor/init.py`
- `tensor/io_safetensors.py`
- `tensor/io_utils.py`
- `tensor/lowrank.py`
- `tensor/numerics.py`
- `tensor/optim.py`
- `tensor/quant_utils.py`
- `tensor/random.py`
- `tensor/regularization.py`
- `tensor/sparse.py`
- `tensor/windows.py`

Required target doc:

- safetensors I/O and checkpoint tensor binding
- tensor initialization kernels and CPU fallback policy
- random/RNG model for sampling, dropout, initialization, and training
- optimizer kernels
- sparse/low-rank tensor policy
- einsum lowering policy
- debug/export-safe utilities classification
- regularization and dropout ownership
- numerics module file-vs-directory correction

This is one of the largest remaining blockers. Core transformer math is covered, but tensor utilities and training math are not.

## `train/`

Missing explicit targets:

- `train/run.py`
- `train/trainer.py`

Required target doc:

- C++ trainer loop
- autograd replacement strategy or explicit hand-written backward plan
- optimizer ownership
- loss scaling and mixed precision
- activation checkpointing
- distributed training interaction
- checkpoint/resume behavior

Training is not migration-ready yet.

## `viz/`

Missing explicit targets:

- `viz/cli.py`
- `viz/render.py`
- `viz/session.py`

Required target doc:

- decide whether visualization remains Python tooling
- if retained, define C++ runtime trace/session data emitted for visualization

## 3. Migration Readiness By Scope

| Scope | Current readiness | Reason |
|---|---|---|
| Core causal LM inference hot path | ready to start implementation | RMSNorm, RoPE, QKV/O/MLP GEMMs, attention, KV cache, embeddings, sampling, and serving loop are covered enough |
| First C++/CUDA runtime skeleton | ready to start implementation | runtime/module spec and first-wave kernel spec exist |
| Full inference model-stack | close but not complete | model registry, checkpoint import, specs, eval, autotune, instrumentation, and compatibility surfaces need explicit docs |
| Distributed inference | partial | NCCL and tensor parallel are covered, but launch/checkpoint/dataloader/topology docs are incomplete |
| Training | not ready | forward op coverage exists, but backward, optimizer, RNG, activation checkpointing, and trainer loop docs are missing |
| Quantization/compression | partial | quant ops are covered, but LoRA/distill/apply/runtime composition need docs |
| Full repository replacement | not ready | 108 Python modules still lack explicit target state |

## 4. Required Docs Before Claiming 100% Coverage

Before saying the repository has full migration coverage, add these docs:

1. `transformer10-module-target-state-matrix.md`
   - one row per Python module
   - target state, target C++/CUDA component, migration priority, and delete/binding/reference decision

2. `transformer10-cpp-model-stack-api-spec.md`
   - C++ classes and interfaces for specs, tensors, modules, blocks, models, registries, checkpoints, serving, training, distributed, eval, and compression

3. `transformer10-complete-tensor-math-function-inventory.md`
   - function-level tensor inventory, including utility math, RNG, optimizers, sparse, low-rank, I/O, initialization, and debug/export helpers

4. `transformer10-training-backward-cuda-spec.md`
   - backward kernels, optimizer kernels, autograd replacement policy, activation storage, mixed precision, loss scaling, and distributed training

5. `transformer10-data-checkpoint-tokenizer-cpp-spec.md`
   - tokenizer, safetensors/checkpoint loading, weight binding, batching, pinned memory, async prefetch, and sharded checkpoint behavior

6. `transformer10-serving-engine-cpp-spec.md`
   - request lifecycle, batching, paged KV cache, CUDA Graph buckets, sampling, streaming, cancellation, instrumentation, and safety hooks

7. `transformer10-autotune-eval-benchmark-spec.md`
   - autotune database, benchmark harness, parity runner, latency/memory reports, and shape-specific kernel-plan validation

8. `transformer10-compression-quantization-runtime-spec.md`
   - quantized weight formats, LoRA, pruning, distillation, calibration, runtime composition, and checkpoint metadata

9. `transformer10-noncore-systems-targets.md`
   - corpus, governance, registry, RAG, RL, safety, visualization, examples, experiments, packaging, and export target states

## 5. Practical Recommendation

Start implementation only for the first scoped lane:

- C++/CUDA runtime skeleton
- tensor descriptor and device buffer ownership
- RMSNorm
- RoPE
- KV cache append/read
- cuBLASLt linears
- attention prefill/decode
- embedding
- sampler
- benchmark/parity harness

Do not claim full-stack migration readiness yet.

The next docs pass should produce a module-by-module target-state matrix. That matrix is the missing bridge between "we understand CUDA/C++ references" and "we can migrate the whole repository without leaving hidden PyTorch-owned subsystems behind."
