# transformer_10 Non-Core Systems Target Targets

This document classifies the repository areas that are not on the first hot path but still need an explicit target state for a full migration.

Target states:

- `cpp_runtime`
- `python_binding`
- `python_reference`
- `remove_or_merge`
- `defer`

## 1. Governance

Files:

- `governance/__main__.py`
- `governance/card.py`
- `governance/cli.py`
- `governance/lineage.py`
- `governance/receipt.py`
- `governance/sbom.py`
- `governance/signature.py`
- `governance/utils.py`

Target:

- short term: `defer`
- long term: partial `cpp_runtime` metadata emitters plus `python_binding` CLIs

Reason:

- these are not CUDA or PyTorch hot-path blockers
- but checkpoint, artifact, and export flows should emit stable metadata hooks from C++

## 2. Corpus Tooling

Files:

- `corpus/build.py`
- `corpus/cli.py`
- `corpus/dedup.py`
- `corpus/manifest.py`
- `corpus/pii.py`

Target:

- `defer`

Reason:

- offline dataset tooling can remain Python until the owned training/runtime stack stabilizes
- it still needs explicit non-blocking classification so it is not mistaken for an unresolved migration hole

## 3. Visualization

Files:

- `viz/attention.py`
- `viz/cli.py`
- `viz/render.py`
- `viz/session.py`

Target:

- `python_binding` over runtime trace output
- `viz/attention.py` stays `python_reference`

Reason:

- rendering and HTML/report generation do not need C++
- the runtime must emit stable traces and scalar streams that these tools consume

## 4. Interpretability

Files:

- `interpret/activation_cache.py`
- `interpret/cli.py`
- `interpret/logit_diff.py`
- `interpret/logit_lens.py`
- `interpret/tracer.py`

Target:

- activation capture and tracing hooks: `cpp_runtime`
- CLI and notebook-facing entrypoints: `python_binding`

Reason:

- the runtime needs owned capture points
- the analysis UX can remain Python

## 5. Export

Files:

- `export/cli.py`
- `export/exporter.py`
- `model/export.py`

Target:

- exporter core: `cpp_runtime`
- CLI and thin wrappers: `python_binding`
- TorchScript- and ONNX-specific PyTorch tracing helpers: `remove_or_merge` once true exporter exists

## 6. Packaging

Files:

- `pack/cli.py`

Target:

- `python_binding`

Reason:

- packaging and end-to-end smoke tests can stay scripting-oriented
- binary artifact creation and metadata should be driven by the C++ runtime build outputs

## 7. Registry

Files:

- `registry/client.py`

Target:

- `cpp_runtime`

Reason:

- model and artifact registry integration belongs with checkpoint and artifact ownership

## 8. RAG

Files:

- `rag/config.py`
- `rag/pipeline.py`
- `rag/cli.py`

Target:

- short term: `defer`
- long term: mixed
  - retrieval and store interfaces can remain outside the core CUDA runtime
  - generation integration can use `python_binding` over the serving engine

Reason:

- RAG is application-layer stack on top of the core model runtime
- do not block core migration on it

## 9. RL

Files:

- `rl/config.py`
- `rl/trainer.py`
- `rl/cli.py`

Target:

- short term: `defer`
- long term: `cpp_runtime` over the training subsystem if RL becomes first-class

Reason:

- supervised training migration comes first
- RL should not remain accidental PyTorch-only code forever, but it is not the first blocker

## 10. Safety

Files:

- `safety/guard.py`

Target:

- `cpp_runtime` interface with optional `python_binding` policies

Reason:

- serving runtime should support guard hooks explicitly
- actual policy implementations may remain scripting-friendly in some deployments

## 11. Examples And Experiments

Files:

- `examples/debug_attention.py`
- `examples/debug_parity.py`
- `examples/debug_single_layer.py`
- `experiments/repo_conditioned_fast_weights.py`

Target:

- examples: `python_reference`
- experiments: `defer`

Reason:

- these are validation and exploration helpers, not runtime ownership surfaces

## 12. Legacy Kernel Wrappers

Files:

- `kernel/bench.py`
- `kernel/flash.py`
- `kernel/registry.py`
- `kernel/rope.py`
- `kernel/triton.py`

Target:

- `remove_or_merge` into `runtime/cuda` and `autotune`
- `kernel/triton.py` may remain temporary `python_binding` during transition

Reason:

- the long-term runtime tree should not keep a second legacy kernel-API package alive

## 13. Rule For Non-Core Systems

A non-core package is considered covered if:

- it has an explicit final state
- it is no longer silently required for runtime correctness
- any Python retained there is intentionally binding, reference, or deferred tooling

That is enough to stop these packages from becoming hidden blockers while the core C++/CUDA model stack is implemented.
