## Contributing

Thanks for helping improve Tensor. Please align contributions with the project’s stateless numerics philosophy.

### Guidelines
- Prefer pure functional APIs that return new tensors; make in-place variants explicit with a trailing `_`.
- Avoid global state, hidden caches, implicit device/dtype changes, and side-effects.
- Maintain axis clarity and mask semantics (boolean, True=masked). Add shape assertions where helpful.
- Use fp32 accumulation for reductions/normalizations unless there is a compelling alternative.
- Annotate function signatures with precise types; keep names descriptive and axis-aware.
- Provide small docs/examples and unit tests (CPU and CUDA if applicable); test across common dtypes.

### Code style
- Follow the repository’s code style and linting. Prefer readable multi-line code over clever one-liners.
- Comments: explain non-obvious rationale, invariants, numerical caveats; avoid restating the code.

### Performance
- Be mindful of memory formats and strides; avoid unexpected `.contiguous()` unless necessary.
- Consider numerically-stable algorithms and chunked reductions for very large dimensions.
- Add microbenchmarks if performance is the goal; include before/after notes.

### Testing
- Include property-based tests when possible; add gradchecks for differentiable ops or finite-diff fallbacks.
- Verify determinism/reproducibility where relevant and guard against NaN/Inf.

### Proposing changes
- Open an issue or RFC outlining: problem, API proposal (signature + semantics), numerics considerations, and alternatives.
- Keep scope focused; additive changes preferred over breaking ones. Document deprecations.


