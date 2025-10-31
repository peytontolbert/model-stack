## Vision & Philosophy

Tensor is a stateless numerics library for transformer stacks.

### Core principles
- Stateless by default: APIs are pure functions with no hidden state, caches, or global flags. Any state (e.g., online statistics) is explicit in inputs/outputs.
- Shape-first: clear axis semantics, batch-first defaults, and helpers to assert/inspect shapes and masks.
- Numerical robustness: fp32 accumulation where it matters, stable transforms (log-space, Welford, Kahan), and explicit dtype management.
- Composability: fine-grained building blocks that compose without side-effects, with minimal opinions about modeling.
- Portability and predictability: CPU/CUDA parity where possible; no implicit device/dtype moves; memory layout utilities that don’t leak state.
- Ergonomics without magic: small convenience wrappers, rich docstrings and examples, consistent naming; no metaprogramming surprises.

### What “stateless numerics” means here
- Functions return new tensors; in-place variants are opt-in and clearly suffixed (`_`).
- Layers that must hold parameters (e.g., `RMSNorm`) are thin wrappers around functional ops and never retain run-time statistics.
- “Stateful” routines (e.g., online RMS) return updated state, leaving ownership to the caller.

### API design tenets
- Axis-aware: most ops expose an `axis` or `dim` argument and support tuples of axes where meaningful.
- Mask semantics: boolean masks use True=masked consistently; shape helpers enforce broadcastability.
- Determinism: utilities to assert reproducibility and guard against NaN/Inf are first-class.

### Out of scope
- No training loops, model classes, or dataset plumbing.
- No heavyweight kernel dependencies by default; optional accelerators may be provided behind feature flags.


