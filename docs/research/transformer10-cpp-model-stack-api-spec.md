# transformer_10 C++ Model-Stack API Spec

This document defines the target C++ API surface for replacing the current Python/PyTorch repository with a modular C++/CUDA model stack.

The design rule is fixed:

- preserve the current repository shape and configurability
- replace implementation ownership with C++ runtime objects and CUDA kernel families
- keep Python as bindings, scripting, and compatibility only

## 1. Namespace Layout

Recommended long-term namespace tree:

```text
t10::
  core
  specs
  tensor
  nn
  attention
  blocks
  model
  checkpoint
  data
  serve
  train
  dist
  compress
  eval
  autotune

t10_cuda::
  tensor
  kernels
  gemm
  attention
  cache
  collectives
  quant
```

## 2. Core Runtime Types

## `t10::core`

Required types:

- `DeviceRef`
  - device kind, ordinal, stream handle, optional event handle
- `DType`
  - `fp32`, `fp16`, `bf16`, `int8`, `fp8_e4m3`, `fp8_e5m2`, `nf4`
- `Layout`
  - row-major, col-major, packed QKV, BHSD, BSHD, paged KV, block-sparse
- `Shape`
  - rank and static-or-dynamic extents
- `Stride`
  - explicit stride descriptor
- `TensorDesc`
  - dtype, shape, stride, layout, alignment, device
- `TensorView`
  - non-owning pointer plus `TensorDesc`
- `DeviceBuffer`
  - owning allocation backed by CUDA async allocator / pool
- `PinnedHostBuffer`
  - async H2D and D2H staging
- `Workspace`
  - scratch allocator with stream-scoped lifetime
- `Status`
  - error code, message, optional source location
- `Expected<T>`
  - result or status
- `RngState`
  - seed, subsequence, offset, graph-safe replay behavior
- `ExecutionMode`
  - inference, training, eval, calibration, export
- `EventTrace`
  - NVTX labels, timestamps, counters

Required behavior:

- no hidden global state for dtype, stream, allocator, or RNG
- every hot-path API accepts explicit execution context
- tensor shape and layout mismatches fail at the C++ boundary before kernel launch

## 3. Spec Objects

## `t10::specs`

Target C++ config objects mirroring current Python `specs/` and `blocks/config.py`:

- `ModelConfig`
  - `d_model`, `n_heads`, `n_kv_heads`, `n_layers`, `d_ff`, `vocab_size`
  - norm type, activation type, positional policy, residual policy
  - attention backend preferences
  - sliding-window, prefix, block-sparse, cross-attention flags
  - dtype and precision policy
- `BlockConfig`
  - block variant, adapter policy, MoE policy, dropout policy, init recipe
- `AttentionConfig`
  - causal/prefix/encoder-decoder mode
  - MHA/MQA/GQA layout
  - paged KV flags, head grouping, mask pattern
- `SamplerConfig`
  - top-k, top-p, min-p, repetition/presence/frequency penalties
  - grammar/regex/schema constraints
  - mirostat, TFS, eta sampling
- `DistConfig`
  - tensor, context, sequence, pipeline, data parallel layout
  - NCCL topology and overlap policy
- `TrainConfig`
  - optimizer, scheduler, loss scaling, checkpointing, accumulation, EMA/SWA/SAM
- `CompressionConfig`
  - quant scheme, LoRA, pruning, distillation options
- `ExportConfig`
  - artifact target, metadata, compatibility policy
- `VizConfig`
  - trace level, sampling policy, retention
- `ResolvedOpSet`
  - resolved kernel and library choices for a model instance

Required resolver layer:

- `SpecResolver`
  - validates config combinations
  - resolves named ops to concrete runtime implementations
  - binds autotuned plans and fallback order

## 4. Tensor And Kernel API

## `t10::tensor`

Primary interfaces:

- `TensorFactory`
  - alloc, alias, reshape, view, cast, pack, shard
- `TensorOps`
  - high-level entrypoints dispatching to CUDA or CPU reference implementations
- `TensorInit`
  - initialization and reset policies
- `TensorIo`
  - safetensors, pinned copies, async transfer
- `TensorDebug`
  - NaN checks, reproducibility checks, shape assertions

## `t10_cuda::tensor`

Required kernel families:

- norms
- positional
- activation
- residual
- numerics
- losses
- sampling
- ragged/segment
- sparse helpers
- quant/dequant
- layout transforms

Kernel entrypoint conventions:

- no Python-style implicit broadcasting decisions inside kernels
- every launch consumes fully resolved descriptors
- kernels return `Status` plus optional telemetry record

## 5. Neural Network Modules

## `t10::nn`

Base interfaces:

- `Module`
  - `forward(OpContext&, TensorView in, TensorView out_or_workspace)`
  - `backward(GradContext&, ModuleTape&, TensorView d_out)`
  - `parameters()`
  - `buffers()`
  - `state_dict_keys()`
- `ParameterizedModule : Module`
  - owns parameter descriptors and optimizer metadata
- `CompositeModule : Module`
  - owns submodule list and execution order

Leaf modules:

- `Embedding`
- `Linear`
- `QuantizedLinear`
- `LoRALinear`
- `Norm`
- `Activation`
- `MLP`
- `MoERouter`
- `Residual`
- `Sampler`

Notes:

- `Linear` is a C++ wrapper over cuBLASLt plan selection, epilog policy, and optional fused bias/activation
- `Norm`, `Activation`, `Residual`, `Sampler`, and layout transforms are primarily CUDA-kernel backed

## 6. Attention API

## `t10::attention`

Required classes:

- `AttentionPlanner`
  - chooses dense/paged/block-sparse/Triton/custom path
- `AttentionCache`
  - paged KV ownership, block tables, append/read/evict/reorder
- `AttentionModule`
  - QKV projection wrapper
  - RoPE / positional integration
  - prefill and decode dispatch
  - output projection wrapper
- `MaskPolicy`
  - causal, prefix-LM, banded, strided, local, segment-bidir, block-sparse
- `AttentionLayout`
  - MHA/MQA/GQA head grouping and pack/unpack rules
- `FlashCompatibilityAdapter`
  - optional compatibility wrapper for external kernels during migration only

Kernel families under it:

- `qkv_prepare`
- `attention_prefill`
- `attention_decode`
- `paged_kv_append`
- `paged_kv_gather`
- `mask_apply`

## 7. Block Layer

## `t10::blocks`

Required C++ block classes:

- `TransformerBlock`
- `LlamaBlock`
- `GPTBlock`
- `ParallelTransformerBlock`
- `MoEBlock`
- `CrossAttentionBlock`
- `EncoderBlock`
- `DecoderBlock`
- `PrefixLMBlock`
- `LocalAttentionBlock`
- `BandedAttentionBlock`
- `StridedAttentionBlock`
- `DilatedLocalAttentionBlock`
- `WindowPatternAttentionBlock`
- `SegmentBidirAttentionBlock`
- `BlockSparseAttentionBlock`

Support classes:

- `BlockRegistry`
- `BlockFactory`
- `BlockPolicySet`
- `BlockSchedule`
- `AdapterPolicy`
- `BlockInspector`

Rules:

- preserve current block configurability
- block classes own orchestration and parameter binding
- block internals call CUDA kernel families and library wrappers, not PyTorch ops

## 8. Model Layer

## `t10::model`

Required classes:

- `Model`
  - base interface for forward, generation hooks, checkpoint load, export hooks
- `CausalLM`
- `PrefixLM`
- `EncoderModel`
- `EncoderDecoderLM`
- `SequenceClassificationHead`
- `TokenClassificationHead`
- `ModelRegistry`
- `ModelFactory`
- `ModelInspector`
- `RuntimeUtils`

Support/import classes:

- `HfSnapshotResolver`
- `HfLlamaImporter`
- `CheckpointBinder`
- `ModelBootstrap`

Rules:

- C++ model objects own module construction and layer ordering
- Python `nn.Module` inheritance is not part of the long-term implementation contract
- model importers convert external checkpoints into internal parameter layout once, not on every forward path

## 9. Checkpoint And Artifact API

## `t10::checkpoint`

Required classes:

- `CheckpointReader`
- `CheckpointWriter`
- `SafeTensorReader`
- `SafeTensorWriter`
- `ShardedCheckpointReader`
- `WeightBinder`
- `ArtifactRegistryClient`
- `DeltaExporter`

Responsibilities:

- config load/save
- tensor storage metadata
- dtype conversion policy
- shard index parsing
- weight remapping for HF and local formats
- delta artifacts for LoRA, pruning, quant metadata

## 10. Data API

## `t10::data`

Required classes:

- `Tokenizer`
- `TokenizerFactory`
- `SentencePieceTokenizer`
- `TokenizerJsonUnigramTokenizer`
- `Batch`
- `PackedBatch`
- `BatchBuilder`
- `ShardReader`
- `StreamingDataset`
- `MapDataset`
- `DataLoader`
- `DistributedDataLoader`

Responsibilities:

- tokenization
- shard discovery and iteration
- packed/ragged batch construction
- pinned-memory staging
- async H2D prefetch
- distributed shard partitioning

## 11. Serving API

## `t10::serve`

Required classes:

- `RuntimeConfig`
- `Request`
- `BatchSlot`
- `DecodeBatch`
- `ServingEngine`
- `Scheduler`
- `GenerationSession`
- `Streamer`
- `InstrumentationSink`
- `SafetyHook`

Hot-path responsibilities:

- request admission
- batch assembly
- cache allocation
- graph bucket selection
- decode loop
- sampler invocation
- partial token streaming
- cancellation

## 12. Training API

## `t10::train`

Required classes:

- `Trainer`
- `TrainStep`
- `Optimizer`
- `AdamW`
- `Lion`
- `Adafactor`
- `Scheduler`
- `GradScaler`
- `LossComputer`
- `CheckpointPolicy`
- `ActivationCheckpointPolicy`
- `EmaTracker`
- `SwaTracker`
- `SamHelper`

Key decision:

- do not rebuild a general dynamic autograd engine first
- use explicit module forward/backward contracts plus a narrow tape for saved activations

## 13. Distributed API

## `t10::dist`

Required classes:

- `Communicator`
- `NcclCommunicator`
- `DistRuntime`
- `RankTopology`
- `TensorParallelPlanner`
- `SequenceParallelPlanner`
- `ContextParallelPlanner`
- `PipelinePlanner`
- `ShardedCheckpointCoordinator`

Operations:

- allreduce
- allgather
- reduce-scatter
- all-to-all
- point-to-point pipeline sends/recvs
- overlap with GEMM and attention stages

## 14. Compression API

## `t10::compress`

Required classes:

- `CompressionManager`
- `Quantizer`
- `QuantizedWeightFormat`
- `LoRAAdapter`
- `LoRAMerger`
- `Pruner`
- `DistillationTeacher`
- `DistillationHooks`
- `CompressionDelta`

## 15. Eval And Autotune API

## `t10::eval`

Required classes:

- `BenchmarkRunner`
- `ParityRunner`
- `LatencyRunner`
- `MemoryRunner`
- `CalibrationRunner`
- `SuiteRunner`
- `ReportWriter`

## `t10::autotune`

Required classes:

- `SearchSpace`
- `Trial`
- `Study`
- `KernelPlan`
- `KernelPlanDb`
- `Searcher`
- `PresetLibrary`

## 16. Python Binding Boundary

Target Python package structure:

```text
runtime/python/bindings/
  core.py
  specs.py
  model.py
  serve.py
  train.py
  data.py
  eval.py
  autotune.py
  compress.py
```

Rules:

- Python bindings may mirror current APIs where useful
- Python bindings do not own correctness-critical execution logic
- Python compatibility wrappers can remain for HF integration, notebooks, CLIs, and parity scripts

## 17. Mapping From Current Python Surface

| Current Python surface | Target C++ owner |
|---|---|
| `specs/*.py` | `t10::specs::*` |
| `tensor/*.py` | `t10::tensor::*` and `t10_cuda::*` |
| `attn/*.py` | `t10::attention::*` and `t10_cuda::attention::*` |
| `blocks/*.py` | `t10::blocks::*` |
| `model/*.py` | `t10::model::*` |
| `serve/*.py` | `t10::serve::*` |
| `train/*.py` | `t10::train::*` |
| `dist/*.py` | `t10::dist::*` |
| `compress/*.py` | `t10::compress::*` |
| `data/*.py` | `t10::data::*` |
| `eval/*.py` | `t10::eval::*` |
| `autotune/*.py` | `t10::autotune::*` |
| `model/checkpoint.py`, `dist/checkpoint.py`, `tensor/io_safetensors.py` | `t10::checkpoint::*` |

## 18. Implementation Rule

If a current Python class or function is part of the model-stack product, the migration target is not "delete Python and hope the kernels are enough".

The migration target is:

- a C++ object if it owns state, planning, policy, metadata, orchestration, or lifecycle
- a CUDA kernel family if it owns GPU tensor execution
- a library wrapper if cuBLASLt, NCCL, or cuTENSOR is the right backend
- a Python binding or reference implementation only when execution ownership has already moved below it
