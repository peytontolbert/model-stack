from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_prefill_attention_kernel_uses_shared_score_cache_on_non_sm90() -> None:
    source = _read("runtime/csrc/backend/attention/cuda_attention_prefill.cuh")

    assert "prefill_attention_hdim_tiled_forward_kernel" in source
    assert "prefill_attention_hdim_legacy_forward_kernel" in source
    assert "const float dot = WarpReduceSum(partial);" in source
    assert "extern __shared__ float shared_mem[]" in source
    assert "float* scores_shared = shared_mem;" in source
    assert "scores_shared[s] = score;" in source
    assert "scores_shared[s] *= inv_denom;" in source
    assert "shared_score_bytes <= kMaxSharedScoreBytes" in source


def test_prefill_attention_exposes_sm80_tensorcore_lane_for_hdim64() -> None:
    source = _read("runtime/csrc/backend/attention/cuda_attention_prefill.cuh")

    assert "prefill_attention_hdim_tensorcore_forward_kernel" in source
    assert "compute_prefill_score_tile_tensorcore" in source
    assert "nvcuda::wmma::load_matrix_sync" in source
    assert "nvcuda::wmma::mma_sync" in source
    assert "MODEL_STACK_DISABLE_ATTENTION_PREFILL_TENSORCORE" in source
    assert "DeviceIsSm80OrLater(q_contig)" in source
    assert "desc.mask_kind == t10::desc::AttentionMaskKind::kNone" in source


def test_prefill_attention_exposes_optional_cutlass_fmha_lane() -> None:
    prefill_source = _read("runtime/csrc/backend/attention/cuda_attention_prefill.cuh")
    cutlass_source = _read("runtime/csrc/backend/attention/cuda_attention_cutlass_prefill.cuh")
    memeff_source = _read("runtime/csrc/backend/attention/cuda_attention_pytorch_memeff_prefill.cuh")
    memeff_impl_source = _read("runtime/csrc/backend/attention/cuda_attention_pytorch_memeff_prefill.cu")
    sm80_inference_source = _read("runtime/csrc/backend/attention/cuda_attention_sm80_inference_prefill.cuh")
    sm80_inference_impl_source = _read("runtime/csrc/backend/attention/cuda_attention_sm80_inference_prefill.cu")
    setup_source = _read("setup.py")

    assert "TryLaunchModelStackSm80InferenceAttentionPrefill<scalar_t, HeadDim>" in prefill_source
    assert "TryLaunchPyTorchMemEffAttentionPrefill<scalar_t, HeadDim>" in prefill_source
    assert "TryLaunchCutlassAttentionPrefill<scalar_t, HeadDim>" in prefill_source
    assert "const bool prefer_cutlass_for_strided_kv =" in prefill_source
    assert "if (prefer_cutlass_for_strided_kv) {" in prefill_source
    assert "AttentionPrefillModelStackSm80InferenceDisabled" in sm80_inference_source
    assert "MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_INFERENCE" in sm80_inference_source
    assert "MODEL_STACK_SM80_INFERENCE_PREFILL_KERNEL" in sm80_inference_source
    assert "if (!k_contig.is_contiguous() || !v_contig.is_contiguous())" in sm80_inference_source
    assert "desc.phase != t10::desc::AttentionPhase::kPrefill || !desc.causal" in sm80_inference_source
    assert "ModelStackSm80CausalPrefillKernel" in sm80_inference_impl_source
    assert "false,\n      false>;" in sm80_inference_impl_source
    assert "static_assert(kKeepOutputInRF" in sm80_inference_impl_source
    assert "iterative_softmax_64x64_rf" in sm80_inference_impl_source
    assert "iterative_softmax_64x64_rf<true" in sm80_inference_impl_source
    assert "iterative_softmax_64x64_rf<false" in sm80_inference_impl_source
    assert "iterative_softmax_64x64_rf<true, true" in sm80_inference_impl_source
    assert "iterative_softmax_64x64_rf<true, false" in sm80_inference_impl_source
    assert "iterative_softmax_64x64_rf<false, true" in sm80_inference_impl_source
    assert "iterative_softmax_64x64_rf<false, false" in sm80_inference_impl_source
    assert "const bool diagonal_tile =" in sm80_inference_impl_source
    assert "static_assert(kWarpColumns == 2" in sm80_inference_impl_source
    assert "warp_column_scratch[accum_m + kQueriesPerBlock * warp_col]" in sm80_inference_impl_source
    assert "BaseKernel::template iterative_softmax" in sm80_inference_impl_source
    assert "MM0::B2bGemm::accumToSmem" in sm80_inference_impl_source
    assert "ModelStackSm80CausalPrefillForwardKernel<Kernel>" in sm80_inference_impl_source
    assert "return try_64x64();" in sm80_inference_impl_source
    assert "MM1FullTileRunnerFromSmem" not in sm80_inference_impl_source
    assert "64x64_exact" not in sm80_inference_source
    assert "ExactFullTilesOnly" not in sm80_inference_impl_source
    assert "ModelStackSm80InferenceKernel64x64Exact" not in sm80_inference_impl_source
    assert "return try_64x64_exact();" not in sm80_inference_impl_source
    assert "k64x128Rf" not in sm80_inference_source
    assert "64x128_rf" not in sm80_inference_source
    assert "ModelStackSm80InferenceKernel64x128" not in sm80_inference_impl_source
    assert "AttentionPrefillPyTorchMemEffDisabled" in memeff_source
    assert "MODEL_STACK_DISABLE_ATTENTION_PREFILL_PYTORCH_MEMEFF" in memeff_source
    assert "MODEL_STACK_PYTORCH_MEMEFF_PREFILL_KERNEL" in memeff_source
    assert "PyTorchMemEffAttention::AttentionKernel<" in memeff_impl_source
    assert "PyTorchMemEffKernel64x128" in memeff_impl_source
    assert "PyTorchMemEffAttentionForwardKernel<Kernel>" in memeff_impl_source
    assert "params.num_batches = static_cast<int32_t>(desc.batch * desc.q_heads);" in memeff_impl_source
    assert "params.num_heads = 1;" in memeff_impl_source
    assert "params.q_strideH = 0;" in memeff_impl_source
    assert "params.o_strideM = static_cast<int32_t>(out.stride(2));" in memeff_impl_source
    assert "AttentionPrefillCutlassDisabled" in cutlass_source
    assert "MODEL_STACK_DISABLE_ATTENTION_PREFILL_CUTLASS" in cutlass_source
    assert "using Attention = AttentionKernel<" in cutlass_source
    assert "attention_kernel_batched_impl<Attention>" in cutlass_source
    assert "params.num_batches = static_cast<int32_t>(desc.batch * desc.q_heads);" in cutlass_source
    assert "params.num_heads = 1;" in cutlass_source
    assert "params.q_strideH = 0;" in cutlass_source
    assert "params.o_strideM = static_cast<int32_t>(out.stride(2));" in cutlass_source
    assert "MODEL_STACK_CUTLASS_PREFILL_QUERIES_PER_BLOCK" in cutlass_source
    assert "LaunchCutlassAttentionPrefillKernel<scalar_t, HeadDim, 128, 64>" in cutlass_source
    assert "MODEL_STACK_CUTLASS_PATH" in setup_source
    assert "MODEL_STACK_WITH_CUTLASS_FMHA" in setup_source
    assert "MODEL_STACK_PYTORCH_SOURCE_PATH" in setup_source
    assert "MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA" in setup_source
    assert "runtime/csrc/backend/attention/cuda_attention_sm80_inference_prefill.cu" in setup_source
    assert "runtime/csrc/backend/attention/cuda_attention_pytorch_memeff_prefill.cu" in setup_source


def test_attention_policy_prefers_smaller_prefill_row_threads_for_fixed_head_dims() -> None:
    source = _read("runtime/csrc/policy/attention_policy.h")

    assert "if (desc.phase == t10::desc::AttentionPhase::kPrefill)" in source
    assert "if (desc.head_dim <= 64)" in source
    assert "if (desc.kv_len >= 512)" in source
    assert "return 256;" in source
    assert "if (desc.kv_len >= 256)" in source
    assert "return 64;" in source
    assert "if (desc.head_dim <= 128)" in source
    assert "return 128;" in source
    assert "inline bool SupportsAttentionSplitKv" in source
    assert "inline int SelectAttentionSplitKvSplits" in source
    assert "inline bool AttentionHasDeadTopLeftCausalKvTail" in source
    assert "inline int64_t AttentionEffectiveKvLen" in source
    assert "const int batch_nheads_mblocks =" in source
    assert "if (batch_nheads_mblocks >= static_cast<int>(0.8f * effective_sms))" in source
    assert "if (efficiency >= 0.85f * max_efficiency)" in source
    assert "!AttentionHasDeadTopLeftCausalKvTail(desc)" in source


def test_attention_plan_info_exposes_split_kv_recommendation() -> None:
    native_source = _read("runtime/csrc/model_stack_native.cpp")
    ops_source = _read("runtime/ops.py")
    cuda_source = _read("runtime/csrc/backend/cuda_attention.cu")
    sm80_source = _read("runtime/csrc/backend/attention/cuda_attention_sm80_inference_prefill.cu")
    memeff_source = _read("runtime/csrc/backend/attention/cuda_attention_pytorch_memeff_prefill.cu")
    cutlass_source = _read("runtime/csrc/backend/attention/cuda_attention_cutlass_prefill.cuh")

    assert 'info["effective_kv_len"]' in native_source
    assert 'info["trimmed_causal_tail"]' in native_source
    assert 'info["split_kv_eligible"]' in native_source
    assert 'info["split_kv_block_n"]' in native_source
    assert 'info["split_kv_num_m_blocks"]' in native_source
    assert 'info["split_kv_num_n_blocks"]' in native_source
    assert 'info["split_kv_effective_sms"]' in native_source
    assert 'info["split_kv_splits"]' in native_source
    assert "SelectAttentionSplitKvSplits(desc, effective_sms)" in native_source
    assert "const int64_t effective_kv_len = t10::policy::AttentionEffectiveKvLen(desc);" in cuda_source
    assert "initial_plan.kernel == t10::policy::AttentionKernelKind::kPrefillHdim64" in cuda_source
    assert "k_contig = k_contig.narrow(2, 0, effective_kv_len);" in cuda_source
    assert "v_contig = v_contig.narrow(2, 0, effective_kv_len);" in cuda_source
    assert "k_contig = k_contig.narrow(2, 0, effective_kv_len).contiguous();" in cuda_source
    assert "v_contig = v_contig.narrow(2, 0, effective_kv_len).contiguous();" in cuda_source
    assert '"effective_kv_len": int(k.shape[2])' in ops_source
    assert '"trimmed_causal_tail": bool(is_causal and attn_mask is None and q.shape[2] < k.shape[2])' in ops_source
    assert '"split_kv_eligible": False' in ops_source
    assert '"split_kv_splits": 1' in ops_source
    assert "params.k_strideB = static_cast<int64_t>(k_contig.stride(1));" in sm80_source
    assert "params.v_strideB = static_cast<int64_t>(v_contig.stride(1));" in sm80_source
    assert "params.q_strideB = static_cast<int64_t>(q_contig.stride(1));" in memeff_source
    assert "params.k_strideB = static_cast<int64_t>(k_contig.stride(1));" in memeff_source
    assert "params.k_strideB = static_cast<int64_t>(k_contig.stride(1));" in cutlass_source
