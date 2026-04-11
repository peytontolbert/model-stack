# transformer_10 Autotune, Eval, And Benchmark Spec

This document defines the runtime-owned replacement for `autotune/` and `eval/`.

## 1. Scope

The model stack needs owned systems for:

- kernel-plan search and persistence
- benchmark execution
- parity validation
- latency and memory reporting
- calibration and evaluation suites

These are not optional. Once PyTorch eager is removed, the repository needs its own verification and tuning layer.

## 2. Autotune Runtime

## Core classes

- `SearchSpace`
- `Trial`
- `TrialStatus`
- `Study`
- `Searcher`
- `KernelPlan`
- `KernelPlanDb`
- `PresetLibrary`
- `CallbackSink`

## Required persistent data

Each kernel plan record should include:

- op family
- device compute capability
- dtype
- shape signature
- layout signature
- selected backend
- launch parameters or library heuristic ids
- workspace requirement
- measured latency
- accuracy/parity tags
- graph-capture compatibility
- timestamp and runtime version

## Search scope

Autotune must support:

- kernel launch parameters for handwritten CUDA
- Triton meta-parameters
- cuBLASLt algorithm / epilog / workspace choices
- attention backend choice
- graph-bucket selection
- collective overlap strategy where relevant

## File mapping

| Current file | Target implementation |
|---|---|
| `autotune/spaces.py` | `t10::autotune::SearchSpace` |
| `autotune/trial.py` | `t10::autotune::Trial` |
| `autotune/study.py` | `t10::autotune::Study` |
| `autotune/presets.py` | `t10::autotune::PresetLibrary` |
| `autotune/callbacks.py` | callback and report sinks, with Python bindings optional |
| `autotune/cli.py` | Python CLI binding only |

## 3. Evaluation Runtime

## Required runners

- `ParityRunner`
- `BenchmarkRunner`
- `LatencyRunner`
- `MemoryRunner`
- `CalibrationRunner`
- `SuiteRunner`
- `ReportWriter`

## Parity coverage

The repository must support explicit parity comparisons against reference paths:

- current Python/PyTorch implementation
- HF snapshot import path when relevant
- reference attention path

Parity modes:

- forward logits parity
- per-block activation parity
- attention score/value parity
- sampler parity where deterministic
- training loss parity

## Benchmarks

Required benchmark modes:

- forward throughput
- generate/decode throughput
- per-op microbenchmarks
- graph-on versus graph-off
- single-GPU versus distributed

## Latency reports

Required outputs:

- p50, p90, p95, p99
- warmup-separated measurements
- prompt and decode phase split
- kernel-plan id and backend tags

## Memory reports

Required outputs:

- model parameter memory
- optimizer-state memory
- activation peak
- KV cache usage
- workspace peak
- allocator fragmentation stats

## 4. Calibration

Calibration runtime should own:

- quantization calibration passes
- ECE and temperature calibration
- teacher/student calibration when distillation is enabled

This replaces ad hoc calibration logic spread across Python-only utilities.

## 5. Report Formats

Persisted reports should be machine-readable and stable:

- JSON for rich results
- CSV for sweep summaries
- optional protobuf or compact binary later

Each report should include:

- model config hash
- runtime version
- device metadata
- selected kernel-plan ids
- seed
- precision mode

## 6. Python Bindings

Python can still provide:

- CLI wrappers
- notebook utilities
- HF/PyTorch reference helpers
- quick report rendering

But the measured/tuned execution should be owned by the C++ runtime.

## 7. File Mapping

| Current file | Target implementation |
|---|---|
| `eval/bench.py` | `BenchmarkRunner` |
| `eval/latency.py` | `LatencyRunner` |
| `eval/memory.py` | `MemoryRunner` |
| `eval/loop.py` | `ParityRunner` / eval loop |
| `eval/calibration.py` | `CalibrationRunner` |
| `eval/compare.py` | model-to-model compare runner |
| `eval/suite.py` | `SuiteRunner` |
| `eval/report.py` | `ReportWriter` |
| `eval/metrics.py`, `eval/seq.py` | metric library |
| `eval/llama_hf_parity.py` | Python reference parity script |
| `eval/cli.py` | Python CLI binding only |

## 8. Definition Of Coverage

Coverage is complete when:

- every kernel family has a persisted plan format
- every major runtime path has a benchmark harness
- parity against the current stack is explicit
- latency and memory are reportable without PyTorch internals

Without this layer, a direct CUDA/C++ migration is not operationally credible.
