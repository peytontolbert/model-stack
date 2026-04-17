## selection.py — Policy-Driven Region Selection

Canonical repo-file/module selection guidance is documented in the [repo-grounded selection doc](../../../repo_grounded_adapters/docs/modules/selection.md).

### What is different here
- This variant frames selection around a `RetrievalPolicy` over a backend-provided `ProgramGraph`.
- Example backends may still expose repo-specific helpers like `question_aware_modules_and_files(...)`, but those are not assumed by the core.
- The important abstraction here is region/entity scoring, not a fixed Python repository selector.

Use the canonical doc for prompt-shaping advice and practical selection heuristics; use this page as the program-conditioned delta.
