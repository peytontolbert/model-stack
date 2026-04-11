# transformer_10 Module Target-State Matrix

This is the repository-wide migration ledger.

Every non-`__init__` Python module in `transformer_10` has an explicit target state:

- `cuda_kernel`
- `cpp_runtime`
- `python_binding`
- `python_reference`
- `remove_or_merge`
- `defer`

Priority:

- `P0`: core migration blocker
- `P1`: required for full inference/runtime ownership
- `P2`: supporting system
- `P3`: reference or intentionally deferred

## `attn/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `attn/backends.py` | `cpp_runtime` | `t10::attention::AttentionPlanner` | P0 | backend selection moves into C++ planner |
| `attn/decoding.py` | `cpp_runtime` | `t10::attention::decode_policy` | P0 | decode-specific attention orchestration |
| `attn/eager.py` | `cpp_runtime` | `t10::attention::AttentionModule` | P0 | orchestrates owned CUDA attention kernels and GEMM wrappers |
| `attn/factory.py` | `cpp_runtime` | `t10::attention::factory` | P1 | attention object construction |
| `attn/flash.py` | `python_binding` | compatibility backend adapter | P2 | transitional external backend wrapper |
| `attn/gqa.py` | `cpp_runtime` | `t10::attention::layout` | P0 | MQA/GQA layout policy |
| `attn/interfaces.py` | `cpp_runtime` | `t10::attention::interfaces` | P0 | C++ ABI/interface boundary |
| `attn/kv_cache.py` | `cpp_runtime` | `t10::attention::AttentionCache` | P0 | cache ownership in C++, kernels under `t10_cuda::cache` |
| `attn/moe.py` | `cpp_runtime` | `t10::compress::moe` / `t10_cuda::routing` | P1 | routing and expert policy |
| `attn/optim_utils.py` | `remove_or_merge` | `t10::train::optim` | P2 | duplicate training helpers merge into trainer/optim subsystem |
| `attn/quant.py` | `cpp_runtime` | `t10::compress::quant_attention` | P1 | quantized attention glue |
| `attn/reference.py` | `python_reference` | reference parity path | P3 | keep for validation only |
| `attn/triton.py` | `python_binding` | Triton compatibility adapter | P2 | transitional kernel authoring path |
| `attn/xformers.py` | `python_binding` | external backend adapter | P2 | parity/transition only |

## `autotune/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `autotune/callbacks.py` | `python_binding` | callback bindings over autotune events | P2 | UI/report glue can stay Python |
| `autotune/cli.py` | `python_binding` | CLI wrapper over `t10::autotune` | P2 | scripting only |
| `autotune/presets.py` | `cpp_runtime` | `t10::autotune::PresetLibrary` | P1 | canonical search-space presets |
| `autotune/spaces.py` | `cpp_runtime` | `t10::autotune::SearchSpace` | P1 | typed parameter spaces |
| `autotune/study.py` | `cpp_runtime` | `t10::autotune::Study` | P1 | optimization driver |
| `autotune/trial.py` | `cpp_runtime` | `t10::autotune::Trial` | P1 | trial state and results |

## `blocks/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `blocks/adapters.py` | `cpp_runtime` | `t10::blocks::AdapterPolicy` | P1 | bottleneck and IA3 adapters |
| `blocks/banded_attn_block.py` | `cpp_runtime` | `t10::blocks::BandedAttentionBlock` | P1 | specialized block variant |
| `blocks/block_sparse_attn_block.py` | `cpp_runtime` | `t10::blocks::BlockSparseAttentionBlock` | P1 | block-sparse variant |
| `blocks/config.py` | `cpp_runtime` | `t10::blocks::BlockConfig` | P0 | block config schema |
| `blocks/cross_attn_block.py` | `cpp_runtime` | `t10::blocks::CrossAttentionBlock` | P1 | cross-attention block |
| `blocks/decoder_block.py` | `cpp_runtime` | `t10::blocks::DecoderBlock` | P1 | decoder block type |
| `blocks/dilated_local_attn_block.py` | `cpp_runtime` | `t10::blocks::DilatedLocalAttentionBlock` | P1 | niche mask variant |
| `blocks/encoder_block.py` | `cpp_runtime` | `t10::blocks::EncoderBlock` | P1 | encoder-only block |
| `blocks/factory.py` | `cpp_runtime` | `t10::blocks::BlockFactory` | P0 | block construction |
| `blocks/gpt_block.py` | `cpp_runtime` | `t10::blocks::GPTBlock` | P1 | GPT layout variant |
| `blocks/init.py` | `cpp_runtime` | `t10::blocks::init` | P1 | block init recipes |
| `blocks/inspect.py` | `cpp_runtime` | `t10::blocks::BlockInspector` | P2 | introspection metadata |
| `blocks/llama_block.py` | `cpp_runtime` | `t10::blocks::LlamaBlock` | P0 | core causal LM block |
| `blocks/local_attn_block.py` | `cpp_runtime` | `t10::blocks::LocalAttentionBlock` | P1 | local mask variant |
| `blocks/moe_block.py` | `cpp_runtime` | `t10::blocks::MoEBlock` | P1 | MoE block and expert MLP |
| `blocks/parallel_block.py` | `cpp_runtime` | `t10::blocks::ParallelTransformerBlock` | P1 | parallel residual topology |
| `blocks/policies.py` | `cpp_runtime` | `t10::blocks::PolicySet` | P0 | norm/attn/MLP policy validation |
| `blocks/prefix_lm_block.py` | `cpp_runtime` | `t10::blocks::PrefixLMBlock` | P1 | prefix-LM block variant |
| `blocks/registry.py` | `cpp_runtime` | `t10::blocks::BlockRegistry` | P0 | named block registry |
| `blocks/schedules.py` | `cpp_runtime` | `t10::blocks::schedule` | P2 | drop path and depth schedules |
| `blocks/segment_bidir_attn_block.py` | `cpp_runtime` | `t10::blocks::SegmentBidirAttentionBlock` | P1 | segment-bidir variant |
| `blocks/shared.py` | `cpp_runtime` | `t10::blocks::CausalSelfAttentionBlockBase` | P0 | shared block base |
| `blocks/stack.py` | `cpp_runtime` | `t10::blocks::TransformerStack` | P0 | block stack composition |
| `blocks/strided_attn_block.py` | `cpp_runtime` | `t10::blocks::StridedAttentionBlock` | P1 | strided mask variant |
| `blocks/targets.py` | `cpp_runtime` | `t10::blocks::target_map` | P2 | target metadata routing |
| `blocks/transformer_block.py` | `cpp_runtime` | `t10::blocks::TransformerBlock` | P0 | core transformer block |
| `blocks/utils.py` | `remove_or_merge` | `t10::core::utils` | P2 | small helper functions merge into core utils |
| `blocks/window_pattern_attn_block.py` | `cpp_runtime` | `t10::blocks::WindowPatternAttentionBlock` | P1 | patterned local attention |

## `compress/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `compress/apply.py` | `cpp_runtime` | `t10::compress::CompressionManager` | P1 | compression config application |
| `compress/distill.py` | `cpp_runtime` | `t10::compress::DistillationLoss` | P1 | teacher/student runtime support |
| `compress/export.py` | `cpp_runtime` | `t10::compress::CompressionDelta` | P1 | export/apply deltas |
| `compress/kv_cache.py` | `remove_or_merge` | `t10::attention::AttentionCache` | P1 | merge cache paging into serving/cache subsystem |
| `compress/lora.py` | `cpp_runtime` | `t10::compress::LoRAAdapter` | P1 | LoRA inject/merge/save |
| `compress/pruning.py` | `cpp_runtime` | `t10::compress::Pruner` | P1 | pruning masks and scores |
| `compress/quantization.py` | `cpp_runtime` | `t10::compress::Quantizer` | P1 | quantized module wrappers and metadata |

## `corpus/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `corpus/build.py` | `defer` | offline corpus tooling | P3 | not a core model-runtime blocker |
| `corpus/cli.py` | `defer` | offline corpus CLI | P3 | scripting only |
| `corpus/dedup.py` | `defer` | offline dedup tooling | P3 | can remain Python |
| `corpus/manifest.py` | `defer` | offline corpus manifest | P3 | non-blocking |
| `corpus/pii.py` | `defer` | offline PII redaction tooling | P3 | non-blocking |

## `data/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `data/batch.py` | `cpp_runtime` | `t10::data::Batch` | P0 | runtime batch descriptor |
| `data/iterable.py` | `cpp_runtime` | `t10::data::StreamingDataset` | P1 | streaming shards and rank slicing |
| `data/loader.py` | `cpp_runtime` | `t10::data::DataLoader` | P1 | map-style shard dataset and loader |
| `data/tokenizer.py` | `cpp_runtime` | `t10::data::TokenizerFactory` | P0 | tokenizer ownership must move below Python |

## `dist/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `dist/checkpoint.py` | `cpp_runtime` | `t10::checkpoint::ShardedCheckpointCoordinator` | P1 | distributed save/load |
| `dist/cli.py` | `python_binding` | CLI wrapper over `t10::dist` | P2 | scripting only |
| `dist/dataloader.py` | `cpp_runtime` | `t10::data::DistributedDataLoader` | P1 | distributed loader runtime |
| `dist/engine.py` | `cpp_runtime` | `t10::dist::DistRuntime` | P0 | distributed execution owner |
| `dist/launch.py` | `cpp_runtime` | `t10::dist::launcher` | P1 | topology init and bootstrap |
| `dist/utils.py` | `cpp_runtime` | `t10::dist::utils` | P1 | rank/world/device helpers |

## `eval/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `eval/bench.py` | `cpp_runtime` | `t10::eval::BenchmarkRunner` | P1 | forward and generate benchmarks |
| `eval/calibration.py` | `cpp_runtime` | `t10::eval::CalibrationRunner` | P1 | ECE and calibration metrics |
| `eval/cli.py` | `python_binding` | CLI wrapper over `t10::eval` | P2 | scripting only |
| `eval/compare.py` | `cpp_runtime` | `t10::eval::CompareRunner` | P1 | model comparison |
| `eval/latency.py` | `cpp_runtime` | `t10::eval::LatencyRunner` | P1 | percentiles and latency dist |
| `eval/llama_hf_parity.py` | `python_reference` | HF parity script | P2 | reference-only validation |
| `eval/loop.py` | `cpp_runtime` | `t10::eval::ParityRunner` | P0 | next-token eval loop |
| `eval/memory.py` | `cpp_runtime` | `t10::eval::MemoryRunner` | P1 | runtime memory reporting |
| `eval/metrics.py` | `cpp_runtime` | `t10::eval::metrics` | P1 | eval metric library |
| `eval/report.py` | `cpp_runtime` | `t10::eval::ReportWriter` | P1 | persisted results |
| `eval/seq.py` | `cpp_runtime` | `t10::eval::seq_metrics` | P2 | text metrics and sequence reports |
| `eval/suite.py` | `cpp_runtime` | `t10::eval::SuiteRunner` | P1 | bundled eval suites |

## `examples/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `examples/debug_attention.py` | `python_reference` | debug example | P3 | example over owned runtime |
| `examples/debug_parity.py` | `python_reference` | parity example | P3 | example only |
| `examples/debug_single_layer.py` | `python_reference` | single-layer debug example | P3 | example only |

## `experiments/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `experiments/repo_conditioned_fast_weights.py` | `defer` | experimental research path | P3 | not a migration blocker |

## `export/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `export/cli.py` | `python_binding` | export CLI wrapper | P2 | scripting only |
| `export/exporter.py` | `cpp_runtime` | `t10::export::Exporter` | P1 | owned artifact export |

## `governance/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `governance/__main__.py` | `defer` | Python governance entrypoint | P3 | non-core tooling |
| `governance/card.py` | `defer` | model-card tooling | P3 | can consume C++ metadata later |
| `governance/cli.py` | `defer` | governance CLI | P3 | non-core |
| `governance/lineage.py` | `defer` | lineage tooling | P3 | non-core |
| `governance/receipt.py` | `defer` | reproducibility receipt tooling | P3 | non-core |
| `governance/sbom.py` | `defer` | SBOM tooling | P3 | non-core |
| `governance/signature.py` | `defer` | artifact signature tooling | P3 | non-core |
| `governance/utils.py` | `defer` | governance helpers | P3 | non-core |

## `interpret/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `interpret/activation_cache.py` | `cpp_runtime` | `t10::interpret::ActivationCache` | P2 | runtime capture hooks |
| `interpret/cli.py` | `python_binding` | interpret CLI wrapper | P2 | analysis UX can stay Python |
| `interpret/logit_diff.py` | `cpp_runtime` | `t10::interpret::logit_diff` | P2 | analysis over owned activations |
| `interpret/logit_lens.py` | `cpp_runtime` | `t10::interpret::logit_lens` | P2 | analysis helper with Python bindings |
| `interpret/tracer.py` | `cpp_runtime` | `t10::interpret::Tracer` | P2 | owned trace hooks |

## `kernel/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `kernel/bench.py` | `remove_or_merge` | `t10::eval::BenchmarkRunner` | P2 | merge legacy microbench layer |
| `kernel/flash.py` | `remove_or_merge` | `t10::attention::planner` | P2 | legacy backend loader |
| `kernel/registry.py` | `remove_or_merge` | `t10::autotune::KernelPlanDb` | P1 | kernel registry merges into owned runtime |
| `kernel/rope.py` | `remove_or_merge` | `t10_cuda::tensor::positional` | P1 | legacy RoPE wrapper |
| `kernel/triton.py` | `python_binding` | Triton compatibility adapter | P2 | transitional only |

## `model/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `model/causal.py` | `cpp_runtime` | `t10::model::CausalLM` | P0 | core causal model owner |
| `model/checkpoint.py` | `cpp_runtime` | `t10::checkpoint::CheckpointReader/Writer` | P0 | config and safetensors binding |
| `model/compile.py` | `remove_or_merge` | `t10::serve::graph` / build config | P2 | torch.compile path not long-term owner |
| `model/encoder.py` | `cpp_runtime` | `t10::model::EncoderModel` | P1 | encoder model class |
| `model/export.py` | `remove_or_merge` | `t10::export::Exporter` | P2 | merge into export subsystem |
| `model/factory.py` | `cpp_runtime` | `t10::model::ModelFactory` | P0 | model construction |
| `model/generate.py` | `python_binding` | compatibility wrapper over serving runtime | P2 | generation wrapper only |
| `model/heads.py` | `cpp_runtime` | `t10::model::heads` | P1 | classification heads |
| `model/hf_llama_loader.py` | `cpp_runtime` | `t10::checkpoint::HfLlamaImporter` | P1 | HF import path |
| `model/hf_snapshot.py` | `cpp_runtime` | `t10::checkpoint::SnapshotCache` | P1 | snapshot resolution |
| `model/inspect.py` | `cpp_runtime` | `t10::model::ModelInspector` | P2 | inspection and target-shape metadata |
| `model/llama_bootstrap.py` | `cpp_runtime` | `t10::model::bootstrap` | P1 | snapshot bootstrap helper |
| `model/lm.py` | `remove_or_merge` | `t10::model::Model` aliases | P2 | alias file folds into model API |
| `model/prefix_lm.py` | `cpp_runtime` | `t10::model::PrefixLM` | P1 | prefix LM model |
| `model/registry.py` | `cpp_runtime` | `t10::model::ModelRegistry` | P0 | named model registry |
| `model/runtime_utils.py` | `cpp_runtime` | `t10::model::runtime_utils` | P1 | runtime helper layer |
| `model/seq2seq.py` | `cpp_runtime` | `t10::model::EncoderDecoderLM` | P1 | seq2seq model |
| `model/utils.py` | `cpp_runtime` | `t10::model::utils` | P2 | model metadata helpers |

## `pack/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `pack/cli.py` | `python_binding` | packaging/test CLI | P3 | scripting around built artifacts |

## `rag/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `rag/cli.py` | `defer` | application-layer RAG CLI | P3 | not a core migration blocker |
| `rag/config.py` | `defer` | RAG app config | P3 | application layer |
| `rag/pipeline.py` | `defer` | RAG pipeline | P3 | can integrate over serving runtime later |

## `registry/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `registry/client.py` | `cpp_runtime` | `t10::checkpoint::ArtifactRegistryClient` | P2 | model/artifact registry |

## `rl/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `rl/cli.py` | `defer` | RL CLI | P3 | after supervised training runtime |
| `rl/config.py` | `defer` | RL config | P3 | deferred training extension |
| `rl/trainer.py` | `defer` | RL trainer | P3 | not first-wave migration |

## `safety/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `safety/guard.py` | `cpp_runtime` | `t10::serve::SafetyHook` | P2 | serving policy interface |

## `runtime/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `runtime/native.py` | `python_binding` | extension loader and capability probe over `_model_stack_native` | P0 | canonical Python entrypoint for runtime ABI detection |
| `runtime/ops.py` | `python_binding` | extension-backed op dispatch surface | P0 | routes tensor ops to native kernels when enabled and preserves exact fallback behavior otherwise |

## `serve/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `serve/api.py` | `python_binding` | API server wrapper over C++ runtime | P2 | service boundary can remain Python |
| `serve/engine.py` | `cpp_runtime` | `t10::serve::ServingEngine` | P0 | serving loop owner |
| `serve/generate.py` | `python_binding` | compatibility wrapper | P2 | convenience API only |
| `serve/instrumented_generate.py` | `python_binding` | instrumentation wrapper | P2 | wraps owned trace sink |
| `serve/runtime.py` | `cpp_runtime` | `t10::serve::RuntimeConfig` and loader | P0 | runtime load path |

## `specs/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `specs/config.py` | `cpp_runtime` | `t10::specs::ModelConfig` | P0 | primary model config |
| `specs/dist.py` | `cpp_runtime` | `t10::specs::DistConfig` | P0 | distributed config |
| `specs/export.py` | `cpp_runtime` | `t10::specs::ExportConfig` | P1 | export config schema |
| `specs/ops.py` | `cpp_runtime` | `t10::specs::ResolvedOpSet` | P0 | op registry and names |
| `specs/resolve.py` | `cpp_runtime` | `t10::specs::SpecResolver` | P0 | config-to-runtime resolution |
| `specs/viz.py` | `cpp_runtime` | `t10::specs::VizConfig` | P2 | visualization config metadata |

## `tensor/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `tensor/activations.py` | `cuda_kernel` | `t10_cuda::tensor::activation` | P0 | pointwise and GLU kernels |
| `tensor/arena.py` | `cpp_runtime` | `t10::core::Workspace` | P1 | allocator and arena logic |
| `tensor/checkpoint.py` | `cpp_runtime` | `t10::train::ActivationCheckpointPolicy` | P1 | rematerialization policy |
| `tensor/compile.py` | `cpp_runtime` | `t10::serve::graph` / `t10::core::stream` | P0 | graph and stream orchestration |
| `tensor/debug.py` | `cpp_runtime` | `t10::tensor::debug` | P2 | parity and debug helpers |
| `tensor/dtypes.py` | `cpp_runtime` | `t10::tensor::dtype_policy` | P0 | dtype and precision metadata |
| `tensor/einsum.py` | `cpp_runtime` | `t10::tensor::einsum_plan` | P2 | planner only |
| `tensor/export_safe.py` | `python_reference` | export compatibility refs | P3 | reference-only helpers |
| `tensor/init.py` | `cpp_runtime` | `t10::tensor::init` | P1 | init policies with CUDA fills underneath |
| `tensor/io_safetensors.py` | `cpp_runtime` | `t10::checkpoint::SafeTensorReader/Writer` | P1 | checkpoint I/O |
| `tensor/io_utils.py` | `cpp_runtime` | `t10::data::PinnedTransfer` | P1 | async transfer helpers |
| `tensor/losses.py` | `cuda_kernel` | `t10_cuda::tensor::loss` | P0 | training and eval losses |
| `tensor/lowrank.py` | `cpp_runtime` | `t10::compress::lowrank` | P2 | low-rank utilities |
| `tensor/masking.py` | `cpp_runtime` | `t10::attention::MaskPolicy` | P0 | mask builders plus device helpers |
| `tensor/metrics.py` | `cpp_runtime` | `t10::eval::metrics` | P2 | metrics library |
| `tensor/mlp.py` | `cpp_runtime` | `t10::nn::MLP` | P0 | MLP module wrapper |
| `tensor/norms.py` | `cuda_kernel` | `t10_cuda::tensor::norms` | P0 | norm kernels |
| `tensor/numerics.py` | `cuda_kernel` | `t10_cuda::tensor::numerics` | P0 | softmax/reductions/numerics |
| `tensor/optim.py` | `cpp_runtime` | `t10::train::optim` | P0 | update routing and trainer helpers; CUDA update kernels underneath |
| `tensor/positional.py` | `cuda_kernel` | `t10_cuda::tensor::positional` | P0 | RoPE and positional kernels |
| `tensor/quant_utils.py` | `cpp_runtime` | `t10::compress::QuantMeta` | P1 | quant metadata and packing |
| `tensor/ragged.py` | `cuda_kernel` | `t10_cuda::tensor::ragged` | P1 | packed/ragged kernels |
| `tensor/random.py` | `cpp_runtime` | `t10::core::RngState` | P0 | deterministic RNG policy |
| `tensor/regularization.py` | `cuda_kernel` | `t10_cuda::tensor::regularization` | P1 | dropout and regularization kernels |
| `tensor/residual.py` | `cuda_kernel` | `t10_cuda::tensor::residual` | P0 | residual and fusion kernels |
| `tensor/sampling.py` | `cuda_kernel` | `t10_cuda::tensor::sampling` | P0 | logits transform and token-select kernels |
| `tensor/shape.py` | `cpp_runtime` | `t10::tensor::layout` | P0 | shape/layout descriptors and checks |
| `tensor/shard.py` | `cpp_runtime` | `t10::dist::planner` | P0 | partition planning and collective wrappers |
| `tensor/sparse.py` | `cuda_kernel` | `t10_cuda::tensor::sparse` | P2 | sparse helpers and block-sparse ops |
| `tensor/windows.py` | `cpp_runtime` | `t10::attention::window_policy` | P2 | ring-buffer/window metadata |

## `train/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `train/run.py` | `cpp_runtime` | `t10::train::RunLoop` | P0 | training entrypoint |
| `train/trainer.py` | `cpp_runtime` | `t10::train::Trainer` | P0 | trainer and step orchestration |

## `viz/`

| Module | State | Target | Prio | Notes |
|---|---|---|---|---|
| `viz/attention.py` | `python_reference` | debug rendering helper | P3 | visualization only |
| `viz/cli.py` | `python_binding` | viz CLI wrapper | P2 | consumes runtime traces |
| `viz/render.py` | `python_binding` | report rendering | P2 | consumes runtime reports |
| `viz/session.py` | `python_binding` | logging/session wrapper | P2 | runtime trace sink adapter |

## Completion Rule

The matrix is complete only if every listed module is accounted for and no runtime-critical behavior is left without a target state.

This file is the authoritative answer to "what happens to each module during the migration?"
