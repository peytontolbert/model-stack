#pragma once

#include <torch/extension.h>

#include "../cuda_device_arch.cuh"
#include "../../descriptors/attention_desc.h"

#include <cstdlib>
#include <type_traits>

#ifdef MODEL_STACK_WITH_CUTLASS_FMHA
#include "kernel_forward.h"
#include <cutlass/arch/arch.h>
#endif

namespace t10::cuda::attention {

inline bool DeviceIsSm80OrLater(const torch::Tensor& reference) {
  int major = 0;
  return t10::cuda::DeviceComputeCapability(reference, &major, nullptr) && major >= 8;
}

inline bool AttentionPrefillCutlassDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_CUTLASS");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline int AttentionPrefillCutlassQueriesPerBlockOverride() {
  const char* env = std::getenv("MODEL_STACK_CUTLASS_PREFILL_QUERIES_PER_BLOCK");
  if (env == nullptr || env[0] == '\0') {
    return 0;
  }
  return std::atoi(env);
}

#ifdef MODEL_STACK_WITH_CUTLASS_FMHA

template <typename scalar_t>
struct CutlassAttentionScalar;

template <>
struct CutlassAttentionScalar<at::Half> {
  using type = cutlass::half_t;
};

template <>
struct CutlassAttentionScalar<at::BFloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename scalar_t, int HeadDim, int QueriesPerBlock, int KeysPerBlock>
inline bool LaunchCutlassAttentionPrefillKernel(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  if constexpr (
      !std::is_same_v<scalar_t, at::Half> &&
      !std::is_same_v<scalar_t, at::BFloat16>) {
    return false;
  } else if constexpr (HeadDim != 64) {
    return false;
  } else {
    if (AttentionPrefillCutlassDisabled()) {
      return false;
    }
    if (!DeviceIsSm80OrLater(q_contig) || t10::cuda::DeviceIsSm90OrLater(q_contig)) {
      return false;
    }
    if (desc.mask_kind != t10::desc::AttentionMaskKind::kNone) {
      return false;
    }
    if (desc.head_mode != t10::desc::AttentionHeadMode::kMHA || desc.q_heads != desc.kv_heads) {
      return false;
    }
    if (desc.q_layout != t10::desc::AttentionLayoutKind::kBHSD ||
        desc.kv_layout != t10::desc::AttentionLayoutKind::kBHSD) {
      return false;
    }
    if (desc.q_len < 64 || desc.kv_len < 128) {
      return false;
    }

    using cutlass_scalar_t = typename CutlassAttentionScalar<scalar_t>::type;
    using Attention = AttentionKernel<
        cutlass_scalar_t,
        cutlass::arch::Sm80,
        true,
        QueriesPerBlock,
        KeysPerBlock,
        64,
        false,
        false>;

    typename Attention::Params params;
    params.query_ptr = reinterpret_cast<cutlass_scalar_t*>(q_contig.data_ptr<scalar_t>());
    params.key_ptr = reinterpret_cast<cutlass_scalar_t*>(k_contig.data_ptr<scalar_t>());
    params.value_ptr = reinterpret_cast<cutlass_scalar_t*>(v_contig.data_ptr<scalar_t>());
    params.output_ptr = reinterpret_cast<cutlass_scalar_t*>(out.data_ptr<scalar_t>());
    params.output_accum_ptr = nullptr;
    params.logsumexp_ptr = nullptr;
    params.scale = scale_value;
    params.head_dim = HeadDim;
    params.head_dim_value = HeadDim;
    params.num_queries = static_cast<int32_t>(desc.q_len);
    params.num_keys = static_cast<int32_t>(desc.kv_len);
    params.num_keys_absolute = static_cast<int32_t>(desc.kv_len);
    // Flatten batch and heads so the CUTLASS BMHD epilogue matches our
    // underlying BHSD contiguous storage without a post-kernel copy.
    params.num_batches = static_cast<int32_t>(desc.batch * desc.q_heads);
    params.num_heads = 1;
    params.custom_mask_type = desc.causal ? Attention::CausalFromTopLeft : Attention::NoCustomMask;

    // Q/K/V arrive as BHSD contiguous. With batch and heads flattened together,
    // each logical batch is a single contiguous [seq, dim] matrix.
    params.q_strideM = static_cast<int32_t>(q_contig.stride(2));
    params.k_strideM = static_cast<int32_t>(k_contig.stride(2));
    params.v_strideM = static_cast<int32_t>(v_contig.stride(2));
    params.q_strideH = 0;
    params.k_strideH = 0;
    params.v_strideH = 0;
    params.q_strideB = static_cast<int64_t>(q_contig.stride(1));
    params.k_strideB = static_cast<int64_t>(k_contig.stride(1));
    params.v_strideB = static_cast<int64_t>(v_contig.stride(1));
    params.o_strideM = static_cast<int32_t>(out.stride(2));

    if (!Attention::check_supported(params)) {
      return false;
    }

    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    const int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      (void)cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    kernel_fn<<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(params);
    return true;
  }
}

template <typename scalar_t, int HeadDim>
inline bool TryLaunchCutlassAttentionPrefill(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  const int qpb_override = AttentionPrefillCutlassQueriesPerBlockOverride();
  if (qpb_override == 128) {
    return LaunchCutlassAttentionPrefillKernel<scalar_t, HeadDim, 128, 64>(
        q_contig, k_contig, v_contig, out, desc, scale_value, stream);
  }
  if (qpb_override == 64) {
    return LaunchCutlassAttentionPrefillKernel<scalar_t, HeadDim, 64, 64>(
        q_contig, k_contig, v_contig, out, desc, scale_value, stream);
  }
  if (desc.q_len >= 256 && desc.kv_len >= 256) {
    if (LaunchCutlassAttentionPrefillKernel<scalar_t, HeadDim, 128, 64>(
            q_contig, k_contig, v_contig, out, desc, scale_value, stream)) {
      return true;
    }
  }
  return LaunchCutlassAttentionPrefillKernel<scalar_t, HeadDim, 64, 64>(
      q_contig, k_contig, v_contig, out, desc, scale_value, stream);
}

#else

template <typename scalar_t, int HeadDim>
inline bool TryLaunchCutlassAttentionPrefill(
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const t10::desc::AttentionDesc&,
    float,
    cudaStream_t) {
  return false;
}

#endif

}  // namespace t10::cuda::attention
