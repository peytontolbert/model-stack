## Roadmap / Future Development

This is a living document capturing near-term and aspirational work. Priorities favor numerics, composability, and performance without sacrificing statelessness.

### 1) Segment and ragged operations
- Expand segment ops: `segment_prod`, `segment_argmax/argmin`, segmented scans.
- Fused GPU-friendly ragged reductions and gathers; blockwise segmented softmax/logsumexp.

### 2) Streaming and online numerics
- Streaming attention-friendly utilities (ring buffers, windowed stats with exact semantics).
- Online normalization family (RMS, mean-only LN) with explicit state carriers and grad-stable updates.

### 3) Norms and weight parametrizations
- Batched spectral norm (conv/nd), weight norm for convs, orthogonalization across batched shapes.
- Chunked/blocked reductions for large-dim numerics to smooth memory peaks.

### 4) Geometry and constraints
- Riemannian utilities (Poincaré, sphere): exp/log maps, geodesics, projections, retractions.
- PSD/simplex/sphere projections with jacobian-aware variants.

### 5) FFT/DCT/SSM suite
- Optimized 1D/2D FFT wrappers and padding modes; multiple DCT/DST types.
- SSM discretization variants (ZOH, bilinear, exact expm); stability parameterizations; impulse/spectrum tools.

### 6) Scans and parallel prefix
- Inclusive/exclusive with ops {add, max, min}; segmented scans; Triton/JAX-like scan adapters (optional).

### 7) Shape system and proofs
- Richer symbolic shapes `S(...)`, unification with constraints, human-readable derivations.
- Assert macros that provide actionable shape errors and auto-suggest fixes.

### 8) Numerics and precision policy
- Log-space families, guarded softmaxes, compensated sums; FP8/INT8-friendly scales (percentile/MSE/histogram).
- Mixed-precision policies per-op with consistent promotion/restore utilities.

### 9) Performance and planning
- Roofline-guided heuristics (choose algorithm by intensity), chunk-size auto-tuning.
- Microbatch planning helpers and overlap utilities (copy/compute streams) with clean CPU fallbacks.

### 10) Ergonomics and tests
- Uniform axis parameters, doc examples, shape diagrams.
- Property-based tests, deterministic harnesses, cross-dtype tolerances; autograd gradchecks and finite-diff fallbacks.

### Compatibility and versioning
- Semantic versioning; deprecations with clear migration notes.


