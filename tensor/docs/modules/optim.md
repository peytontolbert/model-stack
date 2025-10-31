## optim

Optimizer-adjacent utilities: gradient norms, clipping, weight decay, schedules, EMA/SWA/SAM, and stability.

### Gradients
- `grad_norm_parameters`, `global_grad_norm_fp32`, `reduce_grad_norm`, `bucketed_grad_norm`, `grad_norm_report`, `assert_global_norm_below`
- Clipping: `clip_grad_norm_`, `clip_grad_norm_masked_`, `unitwise_clip_`, `unitwise_l2_norm`, `clip_grad_value_`

### Weight decay and masks
- `decoupled_weight_decay_`, `decay_mask_from_names`, `decay_mask_from_params`, `apply_weight_decay_masked_`, `apply_weight_decay_routed_`

### Stability and scaling
- `assert_no_nan_grad`, `loss_scaler_step_safe`, `loss_scale_update_`, `unscale_grads_`, `detect_overflow`, `zero_nan_inf_grad_`, `gradient_centralization_`, `project_grad_orthogonal_`, `add_grad_noise_`

### EMA/SWA/SAM/ASAM
- `ema_update_`, `ema_update_bc_`, `ema_compute_decay`
- `swa_collect_`, `swa_merge_`, `swa_finalize_`
- `sam_perturbation_`, `sam_restore_`, `sam_compute_rho_`, `asam_scale_`, `sam_should_skip`

### Schedules
- `schedule_linear_with_warmup`, `schedule_cosine_with_warmup`, `schedule_poly`, `schedule_piecewise`, `schedule_cosine_restart`, `schedule_linear_floor`

### Parameter updates
- `adamw_update_`, `lamb_update_`, `lion_update_`, `adafactor_update_`, `clip_by_policy_`


