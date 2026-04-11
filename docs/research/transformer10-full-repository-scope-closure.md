# transformer_10 Full Repository Scope Closure

This document closes the remaining documentation gap between:

- "the core model-stack modules are covered"
- and "the whole repository tree is covered well enough that migration will not get blocked by an unclassified subtree"

The earlier module matrix was correct for the core product surface, but it was not a literal row-for-row ledger for every Python file in the repository tree.

This document makes the remaining tree explicit.

## 1. Why This Exists

As of this pass, the repository contains 1517 non-`__init__` Python files.

The earlier core migration ledger in `transformer10-module-target-state-matrix.md` explicitly covers 182 of them. Those 182 are the core product modules that define the actual model-stack, including the new `runtime/` Python extension boundary.

The remaining 1335 files fall into four categories:

- nested product-support subpackages that still need target classification
- test/reference trees
- example/demo trees
- external reference corpora under `other_repos/`

Coverage is only complete if those are classified too.

## 2. Coverage Model

For full-tree closure, every Python file must fall into one of these buckets:

- `core_migration_target`
  - authoritative product code that must migrate into C++/CUDA, Python bindings, or explicit removal
- `python_binding`
  - Python remains as a surface over runtime-owned C++/CUDA logic
- `python_reference`
  - reference, parity, test, debug, or example code only
- `defer`
  - intentionally not part of the first migration-critical product surface
- `external_reference_repo`
  - repository-local research corpus used as reference, not part of the shipping `transformer_10` product

The first 180 product modules are already covered by the module matrix. This document covers the remainder using explicit subtree rules.

## 3. Residual Product-Support Trees

These subtrees are part of the repository’s own functionality and must not stay implicit.

| Path | Count | Classification | Target | Notes |
|---|---:|---|---|---|
| `autotune/algorithms/**` | 5 | `core_migration_target` | `t10::autotune::Searcher` implementations | grid, random, LHS, Sobol, Halton searchers move below Python |
| `autotune/bench/**` | 5 | `core_migration_target` | `t10::autotune::bench` and `t10::eval::microbench` | microbench drivers for attention, KV, MLP, norm, kernel timing |
| `autotune/schedulers/**` | 2 | `core_migration_target` | `t10::autotune::Scheduler` | ASHA and scheduler base become runtime policy |
| `autotune/storage/**` | 1 | `core_migration_target` | `t10::autotune::KernelPlanDb` storage backend | persisted plan and study storage |
| `dist/parallel/**` | 2 | `core_migration_target` | `t10::dist::parallel` | pipeline and tensor-parallel logic must move below Python |
| `dist/strategy/**` | 3 | `python_reference` | legacy torch strategy wrappers | DDP/FSDP/DeepSpeed wrappers are transition-only, not long-term runtime owners |
| `interpret/analysis/**` | 3 | `python_binding` | runtime trace consumers | analysis consumes runtime-captured traces and activations |
| `interpret/attn/**` | 4 | `python_binding` | runtime attention-inspection helpers | attribution and visualization over owned runtime tensors |
| `interpret/attribution/**` | 3 | `python_binding` | runtime attribution tools | can stay Python over runtime capture APIs initially |
| `interpret/causal/**` | 5 | `python_binding` | runtime causal-intervention tooling | patching/steering logic may remain Python-facing if activation access is runtime-owned |
| `interpret/features/**` | 5 | `python_binding` | feature/SAE analysis helpers | analysis layer, not serving/runtime blocker |
| `interpret/importance/**` | 1 | `python_binding` | module scan and attribution consumer | tooling over runtime metadata |
| `interpret/metrics/**` | 2 | `python_binding` | interpret metrics helpers | metrics can remain Python if inputs come from runtime-owned traces |
| `interpret/neuron/**` | 2 | `python_binding` | neuron-level analysis helpers | analysis layer |
| `interpret/probes/**` | 1 | `python_binding` | probe training/eval wrapper | not a serving blocker |
| `interpret/search/**` | 1 | `python_binding` | search helper over runtime outputs | tooling only |
| `rag/components/**` | 5 | `defer` | application-layer RAG components | embedder, retriever, reranker, splitter, store are not first-wave model-runtime blockers |
| `rl/algorithms/**` | 2 | `defer` | RL extension algorithms | PPO and DPO sit after supervised/runtime migration |
| `tensor/masking/**` | 1 | `core_migration_target` | `t10::attention::MaskPolicy` and `t10_cuda::tensor::masking` | window-mask helpers are part of owned masking policy |
| `tensor/numerics/**` | 2 | `core_migration_target` | `t10::tensor::numerics` / `t10_cuda::tensor::numerics` | compensated and stable numerics are part of the owned math stack |

## 4. Test, Debug, And Example Trees

These trees are real repository assets, but they are not authoritative execution ownership.

| Path | Count | Classification | Target | Notes |
|---|---:|---|---|---|
| `tensor/tests/**` | 29 | `python_reference` | parity and regression suite, later mirrored by C++ tests | keep as validation assets during migration |
| `blocks/examples/**` | 1 | `python_reference` | example-only | not part of the runtime product surface |
| `example.py` | 1 | `python_reference` | example-only | root demo script |
| `setup.py` | 1 | `python_reference` | local extension build entrypoint | repository packaging/build glue, not model-runtime product logic |
| `tools/**` | 1 | `python_reference` | repository-maintenance tooling | verification and repo-health scripts are support tooling, not runtime product code |
| `examples/00_tiny_lm/**` through `examples/12_data_tokenize_shard/**` | 12 | `python_reference` | runnable examples over bindings | keep as smoke and usage examples |
| `examples/debug_attention.py` | 1 | `python_reference` | debug example | parity/debug only |
| `examples/debug_parity.py` | 1 | `python_reference` | debug example | parity/debug only |
| `examples/debug_single_layer.py` | 1 | `python_reference` | debug example | parity/debug only |
| `examples/repo_grounded_adapters/**` | 30 | `defer` | experimental application examples | not part of the core model-stack migration |
| `examples/program_conditioned_adapter/**` | 225 | `defer` | experimental/program-conditioned example corpus | not part of the core C++/CUDA migration |
| `examples/be_great/**` | 9 | `defer` | third-party example package | not part of the core model-stack migration |

Rule:

- examples may remain Python even after the runtime is complete
- examples must call the extension/runtime layer rather than silently preserving old eager ownership

## 5. External Reference Corpus

The `other_repos/` tree is not part of the shipping `transformer_10` product. It is a local reference corpus used to inform the migration.

| Path | Count | Classification | Purpose |
|---|---:|---|---|
| `other_repos/ThunderKittens/**` | 94 | `external_reference_repo` | kernel, tile, and layout reference code |
| `other_repos/cuda-kernels/**` | 1 | `external_reference_repo` | small kernel reference |
| `other_repos/extension-cpp/**` | 5 | `external_reference_repo` | extension-boundary reference |
| `other_repos/flash-attention/**` | 192 | `external_reference_repo` | attention and normalization reference code |
| `other_repos/good-kernels/**` | 12 | `external_reference_repo` | utility kernel reference code |
| `other_repos/tiny-cuda-nn/**` | 9 | `external_reference_repo` | graph/runtime utility reference |
| `other_repos/tinygrad/**` | 658 | `external_reference_repo` | compact runtime and tensor-system reference |

These files do not need migration target rows because they are not product modules. They need explicit scope exclusion so they do not become accidental blockers or accidental implementation dependencies.

Rules:

- do not treat `other_repos/**` as product code to be ported
- do not wire runtime ownership through `other_repos/**`
- use them only as reference inputs for design and implementation

## 6. Completion Rule For Full Repository Coverage

Documentation coverage is complete when both of these are true:

1. every core product module is covered by `transformer10-module-target-state-matrix.md`
2. every remaining Python file in the tree is covered by a subtree rule in this document

That is the standard that prevents migration from getting stuck on "what is this subtree supposed to become?"

## 7. What This Means For Migration Readiness

After this scope-closure pass:

- core runtime, tensor, model, serving, training, compression, data, eval, autotune, and distributed modules have migration targets
- nested product-support trees have explicit target states
- tests and examples are explicitly classified as reference or deferred
- `other_repos/` is explicitly classified as external reference corpus, not product scope

That is enough documentation coverage to start migration work without hidden repository-scope ownership gaps.

## 8. Remaining Risk That Docs Cannot Eliminate

This closes the documentation ownership gap. It does not eliminate implementation risk.

Migration can still fail on:

- incorrect ABI decisions
- performance regressions
- unsupported checkpoint formats
- distributed runtime edge cases
- training parity failures
- graph-capture bugs

Those are implementation and verification problems, not missing scope-definition problems.
