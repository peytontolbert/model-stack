## Numerical stability

APIs: `safe_softmax`, `masked_logsumexp`, `logcumsumexp`, `pairwise_sum`, `stable_norm`, `softplus_inv`.

Guidelines: prefer fp32 accumulation for reductions; clamp/guard exponentials; log-space transforms for products.


