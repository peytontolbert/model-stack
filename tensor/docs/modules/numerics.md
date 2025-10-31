## numerics

Numerically stable operations, masked reductions, and chunked variants.

### Distributions
- `safe_softmax`, `masked_log_softmax`, `masked_logsumexp`, `softmax_zloss`

### Stability and utilities
- `log1mexp`, `softplus_safe`, `safe_sigmoid`, `safe_tanh`, `log_sigmoid_safe`, `log1pexp_safe`, `logaddexp_many`
- Masked ops: `masked_mean`, `masked_var`, `masked_std`, `masked_softmax`, `masked_cumsum`, `masked_cummax`, `segment_logsumexp`
- Linear algebra/special: `banded_mm`, `triangular_mask_mm`, `pinv_safe`, `solve_cholesky_safe`, `assert_prob_simplex`

### Chunked/stable variants (submodules)
- `chunked_softmax`, `blockwise_logsumexp`, `masked_softmax_chunked`, `safe_softmax_with_logsumexp`, `kahan_sum`


