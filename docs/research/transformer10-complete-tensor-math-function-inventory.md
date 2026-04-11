# transformer_10 Complete Tensor-Math Function Inventory

This is the function-level migration inventory for the `tensor/` package and directly related math surfaces in `attn/`, `blocks/`, `compress/`, `train/`, and `eval/`.

Target classes:

- `cuda_kernel`
- `library_wrapper`
- `cpp_runtime`
- `python_reference`
- `remove_or_merge`

The intent is not that every current Python helper becomes its own standalone kernel. The intent is that every function lands in the correct owned implementation lane.

## 1. Core Tensor Math

| File | Functions / classes | Target lane | Intended target |
|---|---|---|---|
| `tensor/activations.py` | `gelu`, `silu`, `identity`, `relu2`, `_apply_act`, `with_bias_act`, `bias_gelu`, `bias_silu`, `fast_gelu`, `quick_gelu`, `mish`, `tanh_gelu`, `swiglu`, `geglu`, `reglu`, `split_for_glu`, `glu_chunk_act` | `cuda_kernel` | `t10_cuda::tensor::activation`, fused bias+act and GLU families |
| `tensor/mlp.py` | `MLP` | `cpp_runtime` | `t10::nn::MLP` wrapping cuBLASLt GEMMs plus activation kernels |
| `tensor/norms.py` | `RMSNorm`, `ScaleNorm`, `layer_norm_fp32`, `masked_rmsnorm`, `rmsnorm`, `layer_norm`, `mean_only_layer_norm`, `lp_norm`, `power_norm`, `chunked_rmsnorm` | `cuda_kernel` | `t10_cuda::tensor::norms`, plus `t10::nn::Norm` wrappers |
| `tensor/norms.py` | `spectral_norm`, `weight_norm`, `orthogonalize`, `rolling_norm`, `online_rms` | `cpp_runtime` | training/debug numerics helpers, optional CUDA utility kernels later |
| `tensor/residual.py` | `residual_add`, `gated_residual_add`, `residual_bias_dropout_add`, `prenorm`, `postnorm`, `residual_alpha_schedule` | `cuda_kernel` | `t10_cuda::tensor::residual`, fused residual/norm/dropout families |
| `tensor/positional.py` | `build_rope_cache`, `apply_rotary`, `apply_rotary_scaled`, `rope_ntk_scaling`, `rope_yarn_factors`, `build_rope_cache_2d`, `apply_rotary_2d`, `rotary_fft` | `cuda_kernel` | `t10_cuda::tensor::positional` and `t10::attention::rope_policy` |
| `tensor/positional.py` | `relative_position_bucket`, `build_sinusoidal_cache`, `alibi_slopes`, `rescale_positions`, `build_relative_position_indices`, `relative_position_bias_from_table`, `fit_alibi_slopes`, `RotaryEmbeddingHF` | `cpp_runtime` | positional metadata and compatibility helpers |
| `tensor/masking.py` | `apply_mask`, `to_additive_mask`, `apply_additive_mask_` | `cuda_kernel` | mask application kernels and fused score transforms |
| `tensor/masking.py` | `build_causal_mask`, `build_padding_mask`, `broadcast_mask`, `build_sliding_window_causal_mask`, `build_prefix_lm_mask`, `attention_mask_from_lengths`, `lengths_from_attention_mask`, `make_block_indices`, `build_block_causal_mask`, `build_dilated_causal_mask`, `combine_masks`, `build_banded_mask`, `build_strided_mask`, `build_segment_bidir_mask`, `window_pattern_from_spans`, `invert_mask`, `as_bool_mask`, `build_block_sparse_mask`, `create_causal_mask` | `cpp_runtime` | mask policy builders and layout metadata |
| `tensor/shape.py` | `split_heads`, `merge_heads`, `split_qkv`, `merge_qkv`, `split_gqa_heads`, `merge_gqa_heads`, `pack_heads`, `unpack_heads`, `bhdt_to_bthd`, `rechunk`, `tile`, `reorder` | `cuda_kernel` | layout transforms and pack/unpack helpers |
| `tensor/shape.py` | shape assertions, `pad_to_multiple`, `left_pad`, `right_trim_to`, `expect_shape`, `same_shape`, `enforce_static_shape`, `trace_shape`, `graph_shapes`, `infer`, `broadcast_plan` | `cpp_runtime` | descriptor validation and planning |
| `tensor/dtypes.py` | casts, dtype queries, `cast_logits_for_loss`, `expect_dtype`, `promote_mixed`, `amp_policy_for_op`, `FP8AmaxTracker`, `fp8_scale_from_amax`, `fp8_dynamic_scale_update` | `cpp_runtime` | dtype policy and precision metadata |

## 2. Numerics And Reductions

| File | Functions / classes | Target lane | Intended target |
|---|---|---|---|
| `tensor/numerics.py` | `safe_softmax`, `masked_log_softmax`, `masked_logsumexp`, `masked_softmax`, `chunked_softmax`, `blockwise_logsumexp`, `masked_softmax_chunked`, `safe_softmax_with_logsumexp`, `segment_softmax`, `segment_logsumexp`, `softmax_zloss` | `cuda_kernel` | softmax/logsumexp/reduction kernels used by attention, loss, and sampling |
| `tensor/numerics.py` | `mask_topk`, `mask_topp`, `router_topk_with_capacity` | `cuda_kernel` | routing and sampler-support kernels |
| `tensor/numerics.py` | `welford_update`, `finite_mask`, `nan_to_num_`, `global_l2_norm`, `masked_mean`, `masked_var`, `masked_std`, `kahan_sum`, `l2_normalize`, `entropy_from_logits`, `entropy_from_probs`, `logcumsumexp`, `masked_cumsum`, `masked_cummax`, `stable_cumprod`, `pairwise_sum`, `stable_norm` | `cuda_kernel` | generic numerics kernel family |
| `tensor/numerics.py` | `log1mexp`, `expm1_clip`, `softplus_safe`, `safe_sigmoid`, `safe_tanh`, `log_sigmoid_safe`, `log1pexp_safe`, `logaddexp_many`, `expm1mexp_safe`, `logdiffexp` | `cuda_kernel` | pointwise numerics kernels, often fused into higher-level ops |
| `tensor/numerics.py` | `banded_mm`, `triangular_mask_mm`, `pinv_safe`, `solve_cholesky_safe`, `proj_simplex`, `proj_psd`, `hyperbolic_project` | `cpp_runtime` | optional utility math; not first-wave kernels |
| `tensor/numerics.py` | `percentile_scale`, `mse_scale`, `range_track`, `roofline`, `accum_steps`, `microbatch_plan`, `bucket_sizes`, `tensor_shards`, `estimate_bytes`, `activation_bytes`, `flops_linear`, `flops_conv1d`, `flops_conv2d`, `params_count_linear`, `time_op`, `ema_range_track` | `cpp_runtime` | benchmarking, calibration, and planner utilities |
| `tensor/numerics.py` | `inclusive_scan`, `exclusive_scan`, `fft_conv1d`, `dct`, `idct`, `hilbert_transform`, `bilinear_discretize`, `zoh_discretize`, `ssm_step`, `ssm_stability_margin`, `power_spectrum`, `power_spectrum_generic` | `cuda_kernel` | optional kernels for advanced models and analysis, lower priority than transformer core |
| `tensor/einsum.py` | `plan_einsum` | `cpp_runtime` | einsum lowering/planning layer; do not implement generic runtime einsum first |

## 3. Loss, Metrics, Sampling

| File | Functions / classes | Target lane | Intended target |
|---|---|---|---|
| `tensor/losses.py` | CE/NLL family: `masked_cross_entropy`, `sequence_nll`, `masked_cross_entropy_ls`, `sequence_nll_zloss`, `nll_tokenwise` | `cuda_kernel` | `t10_cuda::tensor::loss` |
| `tensor/losses.py` | divergence family: `masked_kl_div`, `masked_js_div`, `masked_entropy_from_logits`, `masked_log_score`, `masked_spherical_loss` | `cuda_kernel` | training/eval loss kernels |
| `tensor/losses.py` | regression/classification variants: `masked_mse`, `masked_huber`, `masked_focal_loss`, `masked_bce_multilabel`, `bce_with_logits_masked`, `masked_ece`, `masked_label_margin_loss`, `masked_perplexity` | `cuda_kernel` | reusable loss kernels and eval primitives |
| `tensor/metrics.py` | `masked_accuracy`, `masked_token_f1`, `masked_topk_accuracy`, `ece_binning`, `sequence_logprob`, `brier_score`, `masked_entropy`, `auroc_binary`, `calibrate_temperature_grid`, `plascale_logits`, `vector_scale_logits`, `brier_score_masked`, `ece_temperature_sweep`, `sequence_entropy`, `moe_load_balance_loss`, `uniq_ngrams`, `masked_span_f1`, `kl_divergence_many`, `distinct_n`, `self_bleu` | `cpp_runtime` | mostly eval/report code, with optional CUDA kernels for large reductions |
| `tensor/sampling.py` | temperature, penalties, constraints: `apply_temperature`, `apply_repetition_penalty`, `apply_min_p_mask`, `apply_typical_mask`, `ban_tokens`, `force_tokens`, `apply_min_tokens_to_keep_mask`, `apply_topk_mask`, `apply_topp_mask`, `apply_presence_frequency_penalty`, `apply_tfs_mask`, `apply_eta_mask`, `apply_no_repeat_ngram_mask`, `build_regex_constraint_mask`, `apply_stop_phrases_mask`, `json_schema_mask`, `cfgrammar_mask` | `cuda_kernel` | `t10_cuda::tensor::sampling` plus C++ constraint-policy layer |
| `tensor/sampling.py` | `mixture_of_logits`, `mirostat_state`, `mirostat_update`, `sample_gumbel`, `gumbel_topk`, `gumbel_softmax` | `cuda_kernel` | sampler kernels and RNG-dependent helpers |

## 4. Ragged, Sparse, Quant, Low-Rank

| File | Functions / classes | Target lane | Intended target |
|---|---|---|---|
| `tensor/ragged.py` | `pack_sequences`, `unpack_sequences`, `pack_qkv`, `unpack_qkv`, `packed_softmax`, `packed_logsumexp`, `ragged_inclusive_cumsum`, `ragged_exclusive_cumsum`, `ragged_scatter`, `ragged_gather`, `segment_sum`, `segment_mean`, `segment_max`, `segment_min`, `ragged_block_sum`, `ragged_block_mean`, `ragged_block_max` | `cuda_kernel` | ragged segment kernels and packed-sequence helpers |
| `tensor/sparse.py` | `blocksparse_mask`, `bsr_mm`, `sparsify_topk`, `magnitude_prune`, `gather_combine` | `cuda_kernel` | block-sparse and gather/scatter kernels |
| `tensor/quant_utils.py` | `quant_scale_per_channel`, `pack_int8_weight_linear`, `QuantMeta`, `groupwise_absmax`, `fold_scales`, `unfold_scales`, `hist_calibrator`, `mse_calibrator`, `int8_clip_activation_` | `cpp_runtime` | quant metadata and packing utilities, with CUDA helpers underneath |
| `tensor/lowrank.py` | `svd_lowrank`, `factorized_linear`, `rank_selective_update_` | `cpp_runtime` | low-rank utilities for compression and analysis; not first-wave kernels |
| `tensor/windows.py` | `ring_buffer_indices` | `cpp_runtime` | sliding-window and cache indexing policy |

## 5. Training Utilities

| File | Functions / classes | Target lane | Intended target |
|---|---|---|---|
| `tensor/optim.py` | grad norms and clipping: `global_grad_norm`, `grad_norm_parameters`, `clip_grad_norm_`, `unitwise_l2_norm`, `unitwise_clip_`, `clip_grad_value_`, `global_grad_norm_fp32`, `clip_grad_norm_masked_`, `bucketed_grad_norm`, `reduce_grad_norm`, `assert_global_norm_below`, `zero_nan_inf_grad_` | `cpp_runtime` | trainer-side optimizer helpers backed by CUDA reductions where needed |
| `tensor/optim.py` | decay and routing: `decoupled_weight_decay_`, `decay_mask_from_names`, `apply_weight_decay_masked_`, `decay_mask_from_params`, `apply_weight_decay_routed_` | `cpp_runtime` | optimizer metadata and update routing |
| `tensor/optim.py` | optimizer updates: `adamw_update_`, `lamb_update_`, `lion_update_`, `adafactor_update_` | `cuda_kernel` | fused optimizer update kernels |
| `tensor/optim.py` | scheduler and stability helpers: `schedule_linear_with_warmup`, `schedule_cosine_with_warmup`, `schedule_poly`, `schedule_piecewise`, `schedule_cosine_restart`, `schedule_linear_floor`, `loss_scale_update_`, `unscale_grads_`, `detect_overflow`, `sam_compute_rho_`, `asam_scale_`, `sam_should_skip`, `ema_update_`, `ema_compute_decay`, `ema_update_bc_`, `swa_merge_`, `swa_collect_`, `swa_finalize_`, `grad_norm_report`, `loss_scaler_step_safe`, `gradient_centralization_`, `project_grad_orthogonal_`, `add_grad_noise_`, `sam_perturbation_`, `sam_restore_` | `cpp_runtime` | training-control logic with CUDA utilities underneath |
| `tensor/random.py` | `seed_everything`, `set_deterministic`, `_mix_seed_counter`, `philox_stream`, `rng_scope`, `dropout_mask` | `cpp_runtime` | deterministic RNG and graph-safe replay control |
| `tensor/checkpoint.py` | `remat`, `checkpoint_sequential` | `cpp_runtime` | activation checkpointing planner and runtime hooks |
| `tensor/regularization.py` | `drop_path`, `StochasticDepth`, `z_loss_from_logits`, `label_smooth`, `grad_noise_std`, `compute_grad_noise_std`, `build_tokendrop_mask`, `build_sequencedrop_mask`, `schedule_linear`, `magnitude_mask`, `prune_topk_`, `mixout`, `stochastic_depth_mask` | `cuda_kernel` | dropout/regularization kernels plus C++ schedule wrappers |

## 6. I/O, Init, Compile, Debug

| File | Functions / classes | Target lane | Intended target |
|---|---|---|---|
| `tensor/init.py` | all initializers and init schedules | `cpp_runtime` | `t10::tensor::init` with CUDA RNG-backed fills |
| `tensor/io_safetensors.py` | `safetensor_dump`, `safetensor_load`, `stable_tensor_hash` | `cpp_runtime` | checkpoint/artifact I/O |
| `tensor/io_utils.py` | `pin_if_cpu`, `async_to_device` | `cpp_runtime` | async copy and pinned-memory utilities |
| `tensor/compile.py` | `masked_fill_where`, `infer_attn_shapes`, `graph_safe_seed`, `record_stream_guard`, `cuda_graph_seed_scope`, `nccl_stream_guard`, `overlap_copy_compute`, `cuda_graph_warmup`, `graph_replay_step`, `stop_grad`, `custom_grad` | `cpp_runtime` | CUDA graph and stream orchestration |
| `tensor/debug.py` | all debug/repro/gradcheck helpers | `cpp_runtime` | debug hooks and parity harness support |
| `tensor/export_safe.py` | `gelu_export`, `rmsnorm_export`, `gather2d_export`, `scatter_add_export`, `gather1d_export` | `python_reference` | export compatibility references, not runtime hot path |
| `tensor/arena.py` | `ArenaStats`, `ArenaScope`, `TensorArena` | `cpp_runtime` | async allocator, workspace, and memory-pool wrappers |

## 7. Related Non-`tensor/` Math Surfaces

| Surface | Target lane | Intended target |
|---|---|---|
| `attn/eager.py`, `attn/gqa.py`, `attn/kv_cache.py`, `attn/quant.py`, `attn/moe.py` | `cpp_runtime` plus `cuda_kernel` | attention, cache, routing, quantized attention runtime |
| `blocks/*.py` block forward math | `cpp_runtime` plus `cuda_kernel` | block orchestration over owned kernel families |
| `compress/quantization.py`, `compress/lora.py`, `compress/distill.py`, `compress/pruning.py` | `cpp_runtime` plus selected `cuda_kernel` | compression-runtime composition |
| `train/trainer.py`, `train/run.py` | `cpp_runtime` | explicit train-step orchestration |
| `eval/*.py` | `cpp_runtime` and `python_binding` | parity, benchmark, report, and calibration harness |

## 8. Implementation Rule

Coverage is complete only when each tensor function is assigned to one of these outcomes:

- owned CUDA kernel family
- owned C++ runtime or planner
- vendor library wrapper
- retained Python reference helper
- removed or merged into a different owned component

The important point is not "write a kernel for every helper". The important point is "do not leave any math responsibility silently attached to PyTorch eager."
