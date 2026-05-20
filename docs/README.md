# Documentation Index

`docs/` contains repository-level architecture notes, parity references,
research/migration plans, and implementation playbooks. Package-specific docs
usually live next to the package in that package's `README.md`.

This directory should hold documents that cut across packages or define a
repository-wide contract.

## Main Documents

| Document | Purpose |
| --- | --- |
| `custom-kernel-architecture.md` | Source-complete architecture for the native C++/CUDA kernel stack, Python dispatch, runtime flags, validation, and kernel contribution rules. |
| `llama_hf_parity.md` | LLaMA/Hugging Face parity goals, invariants, and verification checklist. |
| `llama3b-asm.md` | Target design notes for a future low-level LLaMA assembly path. This is a design document, not an implemented subsystem. |
| `research/README.md` | Index for CUDA/C++ migration research, implementation plans, target-state matrices, and repository coverage audits. |

## Documentation Ownership

Use this split:

- Put package usage and local ownership in the package README.
- Put native kernel architecture in `custom-kernel-architecture.md`.
- Put migration/research plans under `docs/research/`.
- Put implemented API contracts near the code they describe.
- Put target designs here only when they cross multiple packages.

## Adding Repository Docs

When adding a document:

1. Give it a clear status: implemented contract, migration plan, research note,
   target design, or audit.
2. Link it from this README or the relevant package README.
3. Prefer concrete file paths and owned surfaces over broad prose.
4. If the document claims source or API completeness, add a source-surface test
   that checks the claim.

For package-level documentation coverage, see
`tests/test_repository_documentation_surface.py`.

