from typing import Callable, Dict, Iterable, Optional

import tensor as T

# Curated, stable mapping of canonical spec names to tensor implementations.
# This lets configs reference ops by name and keeps model code decoupled.

CATEGORIES: Dict[str, Dict[str, Callable]] = {
    "activations": {
        "gelu": T.gelu,
        "silu": T.silu,
        "fast_gelu": T.fast_gelu,
        "quick_gelu": T.quick_gelu,
        "mish": T.mish,
        "tanh_gelu": T.tanh_gelu,
        "swiglu": T.swiglu,
        "geglu": T.geglu,
        "reglu": T.reglu,
    },
    "regularization": {
        "drop_path": T.drop_path,
        "StochasticDepth": T.StochasticDepth,
        "z_loss_from_logits": T.z_loss_from_logits,
        "label_smooth": T.label_smooth,
    },
    "windows": {
        "window_partition": T.window_partition,
        "window_merge": T.window_merge,
        "ring_buffer_indices": T.ring_buffer_indices,
    },
    "ragged": {
        "pack_sequences": T.pack_sequences,
        "unpack_sequences": T.unpack_sequences,
        "segment_sum": T.ragged_segment_sum,
        "segment_mean": T.ragged_segment_mean,
        "segment_max": T.ragged_segment_max,
        "segment_min": T.ragged_segment_min,
        "gather": T.ragged_gather,
        "scatter": T.ragged_scatter,
        "packed_softmax": T.ragged_packed_softmax,
        "packed_logsumexp": T.ragged_packed_logsumexp,
    },
    "sparse": {
        "blocksparse_mask": T.blocksparse_mask,
        "bsr_mm": T.bsr_mm,
        "magnitude_prune": T.magnitude_prune,
    },
    "fft": {
        "fft_conv1d": T.fft_conv1d,
        "dct": T.dct,
        "idct": T.idct,
        "hilbert_transform": T.hilbert_transform,
    },
    "state_space": {
        "ssm_step": T.ssm_step,
        "ssm_stability_margin": T.ssm_stability_margin,
        "power_spectrum": T.power_spectrum,
    },
    "scan": {
        "inclusive_scan": T.inclusive_scan,
        "exclusive_scan": T.exclusive_scan,
        "stable_cumprod": T.stable_cumprod,
    },
    "norms": {
        "RMSNorm": T.RMSNorm,
        "ScaleNorm": T.ScaleNorm,
        "rmsnorm": T.rmsnorm,
        "masked_rmsnorm": T.masked_rmsnorm,
        "layer_norm": T.layer_norm,
        "mean_only_layer_norm": T.mean_only_layer_norm,
        "chunked_rmsnorm": T.chunked_rmsnorm,
    },
    "positional": {
        "apply_rotary": T.apply_rotary,
        "apply_rotary_scaled": T.apply_rotary_scaled,
        "build_rope_cache": T.build_rope_cache,
        "alibi": T.build_alibi_bias,
        "relative_bias": T.relative_position_bias_from_table,
        "build_relative_position_indices": T.build_relative_position_indices,
    },
    "masking": {
        "build_causal_mask": T.build_causal_mask,
        "build_sliding_window_causal_mask": T.build_sliding_window_causal_mask,
        "build_prefix_lm_mask": T.build_prefix_lm_mask,
        "build_padding_mask": T.build_padding_mask,
        "broadcast_mask": T.broadcast_mask,
        "invert_mask": T.invert_mask,
        "as_bool_mask": T.as_bool_mask,
    },
    "numerics": {
        "safe_softmax": T.safe_softmax,
        "masked_log_softmax": T.masked_log_softmax,
        "masked_logsumexp": T.masked_logsumexp,
        "safe_softmax_with_logsumexp": T.safe_softmax_with_logsumexp,
        "chunked_softmax": T.chunked_softmax,
        "blockwise_logsumexp": T.blockwise_logsumexp,
    },
    "shape": {
        "split_heads": T.split_heads,
        "merge_heads": T.merge_heads,
        "split_qkv": T.split_qkv,
        "merge_qkv": T.merge_qkv,
        "center_pad": T.center_pad,
        "pad_to_multiple": T.pad_to_multiple,
        "right_trim_to": T.right_trim_to,
        "ensure_even_last_dim": T.ensure_even_last_dim,
    },
    "dtypes": {
        "cast_for_softmax": T.cast_for_softmax,
        "cast_for_norm": T.cast_for_norm,
        "restore_dtype": T.restore_dtype,
        "to_dtype_like": T.to_dtype_like,
        "maybe_autocast": T.maybe_autocast,
        "set_matmul_precision": T.set_matmul_precision,
    },
    "residual": {
        "residual_add": T.residual_add,
        "gated_residual_add": T.gated_residual_add,
        "bias_dropout_add": T.residual_bias_dropout_add,
        "prenorm": T.prenorm,
        "postnorm": T.postnorm,
    },
    "init": {
        "xavier_uniform_linear": T.xavier_uniform_linear,
        "kaiming_uniform_linear": T.kaiming_uniform_linear,
    },
    "losses": {
        "masked_cross_entropy": T.masked_cross_entropy,
        "sequence_nll": T.sequence_nll,
        "masked_mse": T.masked_mse,
        "masked_huber": T.masked_huber,
    },
    "metrics": {
        "masked_accuracy": T.masked_accuracy,
        "masked_topk_accuracy": T.masked_topk_accuracy,
        "sequence_perplexity": T.masked_perplexity,
    },
    "sampling": {
        "apply_topk_mask": T.apply_topk_mask,
        "apply_topp_mask": T.apply_topp_mask,
        "mixture_of_logits": T.mixture_of_logits,
    },
    "quant": {
        "quant_scale_per_channel": T.quant_scale_per_channel,
        "pack_int8_weight_linear": T.pack_int8_weight_linear,
        "QuantMeta": T.QuantMeta,
        "groupwise_absmax": T.groupwise_absmax,
        "hist_calibrator": T.hist_calibrator,
        "mse_calibrator": T.mse_calibrator,
    },
    "shard": {
        "tp_linear_partition": T.tp_linear_partition,
        "kv_partition": T.kv_partition,
        "allreduce_": T.allreduce_,
        "reduce_scatter": T.reduce_scatter,
        "allgather": T.allgather,
        "shard_linear_weight": T.shard_linear_weight,
        "seq_alltoall": T.seq_alltoall,
        "seq_partition": T.seq_partition,
        "seq_gather_restore": T.seq_gather_restore,
    },
    "compile": {
        "allow_in_graph": T.allow_in_graph,
        "masked_fill_where": T.masked_fill_where,
        "infer_attn_shapes": T.infer_attn_shapes,
        "graph_safe_seed": T.graph_safe_seed,
        "record_stream_guard": T.record_stream_guard,
        "cuda_graph_seed_scope": T.cuda_graph_seed_scope,
        "nccl_stream_guard": T.nccl_stream_guard,
        "overlap_copy_compute": T.overlap_copy_compute,
        "cuda_graph_warmup": T.cuda_graph_warmup,
        "graph_replay_step": T.graph_replay_step,
        "custom_grad": T.custom_grad,
        "stop_grad": T.stop_grad,
    },
    "random": {
        "seed_everything": T.seed_everything,
        "set_deterministic": T.set_deterministic,
        "philox_stream": T.philox_stream,
        "rng_scope": T.rng_scope,
        "dropout_mask": T.dropout_mask,
    },
    "io": {
        "pin_if_cpu": T.pin_if_cpu,
        "async_to_device": T.async_to_device,
        "safetensor_dump": T.safetensor_dump,
        "safetensor_load": T.safetensor_load,
        "stable_tensor_hash": T.stable_tensor_hash,
    },
    "debug": {
        "install_nan_guard": T.install_nan_guard,
        "detect_fp16_overflow": T.detect_fp16_overflow,
        "bitwise_equal_forward": T.bitwise_equal_forward,
        "assert_reproducible_step": T.assert_reproducible_step,
        "numeric_grad_check": T.numeric_grad_check,
        "nan_window_scan": T.nan_window_scan,
        "assert_same_rng_scope": T.assert_same_rng_scope,
        "gradcheck_stateless": T.gradcheck_stateless,
    },
}


def get_op(category: str, name: str) -> Callable:
    return CATEGORIES[category][name]


def has_op(category: str, name: str) -> bool:
    return category in CATEGORIES and name in CATEGORIES[category]


def list_ops(category: Optional[str] = None) -> Dict[str, Iterable[str]]:
    if category is None:
        return {k: tuple(v.keys()) for k, v in CATEGORIES.items()}
    return {category: tuple(CATEGORIES.get(category, {}).keys())}


