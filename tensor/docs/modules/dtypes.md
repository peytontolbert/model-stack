## dtypes

Precision, autocast, and dtype-related helpers.

### Casting
- `cast_for_softmax`, `cast_for_norm`, `restore_dtype`, `to_dtype_like`, `cast_logits_for_loss`

### Policies and checks
- `set_matmul_precision`, `maybe_autocast`, `expect_dtype`, `promote_mixed`, `amp_policy_for_op`
- `is_fp16`, `is_bf16`, `is_int8`, `is_fp8`

### FP8 support
- `fp8_dynamic_scale_update`, `FP8AmaxTracker`, `fp8_scale_from_amax`, `with_logits_precision`


