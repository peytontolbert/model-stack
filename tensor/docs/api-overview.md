## API overview

This is a high-level map of modules and notable functions/classes exposed by `tensor`. Refer to source for full signatures and additional helpers.

### Activations (`tensor/activations.py`)
- `gelu`, `silu`, `bias_gelu`, `bias_silu`
- Variants: `fast_gelu`, `quick_gelu`, `mish`, `tanh_gelu`, `swiglu`, `geglu`, `reglu`

### Norms (`tensor/norms.py`)
- Modules: `RMSNorm`, `ScaleNorm`
- Functions: `masked_rmsnorm`, `rmsnorm`, `layer_norm`, `mean_only_layer_norm`, `lp_norm`, `power_norm`, `chunked_rmsnorm`

### MLP (`tensor/mlp.py`)
- `MLP` (standard, SwiGLU variants)

### Positional (`tensor/positional.py`)
- RoPE: `build_rope_cache`, `apply_rotary`, `apply_rotary_scaled`, `build_rope_cache_2d`, `apply_rotary_2d`
- Bias/indices: `build_alibi_bias`, `alibi_slopes`, `fit_alibi_slopes`, `build_relative_position_indices`, `relative_position_bias_from_table`, `relative_position_bucket`
- Utilities: `build_sinusoidal_cache`, `rope_ntk_scaling`, `rope_yarn_factors`, `rescale_positions`

### Masking (`tensor/masking.py`, `tensor/masking/windows.py`)
- Core: `build_causal_mask`, `build_sliding_window_causal_mask`, `build_prefix_lm_mask`, `build_padding_mask`
- Ops: `apply_mask`, `broadcast_mask`, `attention_mask_from_lengths`, `lengths_from_attention_mask`
- Windows: `build_block_causal_mask`, `build_dilated_causal_mask`, `window_pattern_from_spans`
- Utils: `invert_mask`, `as_bool_mask`

### Numerics (`tensor/numerics.py` and submodules)
- Distributions: `safe_softmax`, `masked_log_softmax`, `masked_logsumexp`, `softmax_zloss`
- Stability: `log1mexp`, `softplus_safe`, `safe_sigmoid`, `safe_tanh`, `log_sigmoid_safe`, `log1pexp_safe`, `logaddexp_many`
- Masked reductions/math: `masked_mean`, `masked_var`, `masked_std`, `masked_softmax`, `masked_cumsum`, `masked_cummax`, `segment_logsumexp`
- Linear algebra/special: `banded_mm`, `triangular_mask_mm`, `pinv_safe`, `solve_cholesky_safe`, `assert_prob_simplex`
- Chunked/stable: `chunked_softmax`, `blockwise_logsumexp`, `masked_softmax_chunked`, `safe_softmax_with_logsumexp`, `kahan_sum`

### DTypes (`tensor/dtypes.py`)
- Casting: `cast_for_softmax`, `cast_for_norm`, `restore_dtype`, `to_dtype_like`, `cast_logits_for_loss`
- Checks/util: `is_fp16`, `is_bf16`, `is_int8`, `is_fp8`, `set_matmul_precision`, `maybe_autocast`, `expect_dtype`, `promote_mixed`, `amp_policy_for_op`, `fp8_dynamic_scale_update`, `FP8AmaxTracker`, `fp8_scale_from_amax`, `with_logits_precision`

### Shape (`tensor/shape.py`)
- Splits/merges: `split_heads`, `merge_heads`, `split_qkv`, `merge_qkv`, `split_gqa_heads`, `merge_gqa_heads`
- Padding/format: `center_pad`, `pad_to_multiple`, `right_trim_to`, `ensure_contiguous_lastdim`, `reorder_to_channels_last_2d`
- Assertions/introspection: `assert_mask_shape`, `assert_boolean_mask`, `assert_broadcastable`, `expect_shape`, `same_shape`, `enforce_static_shape`, `trace_shape`, `expect_memory_format`, `stride_equal`, `is_view_of`

### Residual (`tensor/residual.py`)
- `residual_add`, `gated_residual_add`, `residual_bias_dropout_add`, `prenorm`, `postnorm`

### Init (`tensor/init.py`)
- Linear/conv inits and scaling helpers: `xavier_uniform_linear`, `kaiming_uniform_linear`, `mu_param_linear_init_`, `deepnet_residual_scale`, `init_swiglu_bias`, `init_rmsnorm_`, `mu_param_conv_init_`, `xavier_fanfix_linear`, `scaled_silu_init_`, `zero_out_proj_bias`

### Regularization (`tensor/regularization.py`)
- `drop_path`, `StochasticDepth`, label smoothing and z-loss utilities, token/sequence drop helpers

### Losses (`tensor/losses.py`)
- Sequence/token losses with masking: `masked_cross_entropy`, `sequence_nll`, `masked_cross_entropy_ls`, `masked_kl_div`, `masked_js_div`, `sequence_nll_zloss`, `masked_mse`, `masked_huber`, `bce_with_logits_masked`, `nll_tokenwise`, `masked_label_margin_loss`, `masked_log_score`, `masked_spherical_loss`, `masked_perplexity`

### Sampling (`tensor/sampling.py`)
- Temperature/penalties: `apply_temperature`, `apply_repetition_penalty`, `apply_presence_frequency_penalty`
- Top-k/p/typical/min-p/TFS: `apply_topk_mask`, `apply_topp_mask`, `apply_typical_mask`, `apply_min_p_mask`, `apply_min_tokens_to_keep_mask`, `apply_tfs_mask`, `apply_eta_mask`
- Constraints/tools: `build_regex_constraint_mask`, `apply_no_repeat_ngram_mask`, stop phrases, JSON/CFG masks; Gumbel: `sample_gumbel`, `gumbel_topk`, `gumbel_softmax`

### Metrics (`tensor/metrics.py`)
- Mask-aware metrics: `masked_accuracy`, `masked_topk_accuracy`, `masked_token_f1`, ECE tools, entropy/logprob, n-gram distinctness, BLEU, calibration helpers

### Compile/runtime (`tensor/compile.py`)
- Graph helpers: `allow_in_graph`, `masked_fill_where`, `infer_attn_shapes`, CUDA/NCCL seed/stream guards, overlap and replay utilities

### Optim (`tensor/optim.py`)
- Grad norms/clip: `grad_norm_parameters`, `global_grad_norm_fp32`, `clip_grad_norm_`, `clip_grad_norm_masked_`, `unitwise_clip_`, `unitwise_l2_norm`, value clipping
- Regularization/weight decay: decoupled/apply mask utilities
- Stability: NaN/Inf handling, loss scaling, overflow detect/unscale
- EMA/SWA/SAM/ASAM utilities
- Schedules: linear/cosine/poly/piecewise/restart/floor
- Reports/assertions: `grad_norm_report`, `assert_global_norm_below`, bucketed/reduced norm utils

### Quantization (`tensor/quant_utils.py`)
- Int8 packing/scales, per-channel quant utilities and calibrators; `QuantMeta`

### Windows (`tensor/windows.py`)
- `window_partition`, `window_merge`, `ring_buffer_indices`

### Low-rank (`tensor/lowrank.py`)
- `svd_lowrank`, `factorized_linear`, `rank_selective_update_`

### IO (`tensor/io_utils.py`, `tensor/io_safetensors.py`)
- Device/async helpers: `pin_if_cpu`, `async_to_device`
- SafeTensors: `safetensor_dump`, `safetensor_load`, `stable_tensor_hash`

### Sharding/utils (`tensor/shard.py`)
- Partitioning, FLOPs/bytes estimates, latency/activation estimators, collective helpers, sequence all-to-all/partition/gather

### Arena/checkpoint/einsum/debug
- `TensorArena`, `remat`, `checkpoint_sequential`, `plan_einsum`, numerical/debug guards and checks


