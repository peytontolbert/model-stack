#pragma once

#include <torch/extension.h>

#include "../cuda_device_arch.cuh"
#include "../../policy/attention_policy.h"
#include "../../descriptors/attention_desc.h"

#include <cstdint>
#include <type_traits>

namespace t10::cuda::attention {

// Narrow long-context SM80 flash-style lane. This is intentionally limited to
// the equal-length causal BHSD MHA case where the current 64x64_rf lane is
// still slightly behind torch SDPA on Ampere-class GPUs.
inline bool DeviceIsSm80OrLaterForModelStackSm80Flash(const torch::Tensor& reference) {
  int major = 0;
  return t10::cuda::DeviceComputeCapability(reference, &major, nullptr) && major >= 8;
}

inline bool AttentionPrefillModelStackSm80FlashDisabled() {
  return t10::policy::AttentionSm80FlashPrefillDisabled();
}

inline int64_t AttentionPrefillModelStackSm80FlashMinSeq() {
  return t10::policy::AttentionSm80FlashPrefillMinSeq();
}

bool TryLaunchModelStackSm80FlashAttentionPrefillF16(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream);

bool TryLaunchModelStackSm80FlashAttentionPrefillBF16(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream);

template <typename scalar_t, int HeadDim>
inline bool TryLaunchModelStackSm80FlashAttentionPrefill(
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
    if (AttentionPrefillModelStackSm80FlashDisabled()) {
      return false;
    }
    if (!DeviceIsSm80OrLaterForModelStackSm80Flash(q_contig) ||
        t10::cuda::DeviceIsSm90OrLater(q_contig)) {
      return false;
    }
    if (!q_contig.is_contiguous() || !k_contig.is_contiguous() || !v_contig.is_contiguous()) {
      return false;
    }
    if (!t10::policy::SupportsAttentionSm80FlashPrefill(desc)) {
      return false;
    }
    if (desc.q_heads != desc.kv_heads) {
      return false;
    }
    if (!t10::policy::PreferAttentionSm80FlashPrefill(desc)) {
      return false;
    }
    if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return TryLaunchModelStackSm80FlashAttentionPrefillF16(
          q_contig,
          k_contig,
          v_contig,
          out,
          desc,
          scale_value,
          stream);
    } else {
      return TryLaunchModelStackSm80FlashAttentionPrefillBF16(
          q_contig,
          k_contig,
          v_contig,
          out,
          desc,
          scale_value,
          stream);
    }
  }
}

}  // namespace t10::cuda::attention
