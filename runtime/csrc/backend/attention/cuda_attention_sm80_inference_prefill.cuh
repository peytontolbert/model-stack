#pragma once

#include <torch/extension.h>

#include "../cuda_device_arch.cuh"
#include "../../descriptors/attention_desc.h"

#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace t10::cuda::attention {

// Narrow SM80 fast lane for inference-only causal prefill. This deliberately
// excludes masks, bias, dropout, varlen, and non-BHSD layouts.
inline bool DeviceIsSm80OrLaterForModelStackSm80Inference(const torch::Tensor& reference) {
  int major = 0;
  return t10::cuda::DeviceComputeCapability(reference, &major, nullptr) && major >= 8;
}

inline bool AttentionPrefillModelStackSm80InferenceDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_INFERENCE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

enum class ModelStackSm80InferencePrefillKernelKind {
  kAuto,
  k64x64Rf,
};

inline ModelStackSm80InferencePrefillKernelKind AttentionPrefillModelStackSm80InferenceKernelOverride() {
  const char* env = std::getenv("MODEL_STACK_SM80_INFERENCE_PREFILL_KERNEL");
  if (env == nullptr || env[0] == '\0') {
    return ModelStackSm80InferencePrefillKernelKind::kAuto;
  }
  if (std::strcmp(env, "64x64_rf") == 0) {
    return ModelStackSm80InferencePrefillKernelKind::k64x64Rf;
  }
  return ModelStackSm80InferencePrefillKernelKind::kAuto;
}

bool TryLaunchModelStackSm80InferenceAttentionPrefillF16(
    ModelStackSm80InferencePrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream);

bool TryLaunchModelStackSm80InferenceAttentionPrefillBF16(
    ModelStackSm80InferencePrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream);

template <typename scalar_t, int HeadDim>
inline bool TryLaunchModelStackSm80InferenceAttentionPrefill(
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
    if (AttentionPrefillModelStackSm80InferenceDisabled()) {
      return false;
    }
    if (!DeviceIsSm80OrLaterForModelStackSm80Inference(q_contig) ||
        t10::cuda::DeviceIsSm90OrLater(q_contig)) {
      return false;
    }
    if (desc.phase != t10::desc::AttentionPhase::kPrefill || !desc.causal) {
      return false;
    }
    if (desc.mask_kind != t10::desc::AttentionMaskKind::kNone) {
      return false;
    }
    if (desc.head_mode != t10::desc::AttentionHeadMode::kMHA ||
        desc.q_heads != desc.kv_heads) {
      return false;
    }
    if (desc.q_layout != t10::desc::AttentionLayoutKind::kBHSD ||
        desc.kv_layout != t10::desc::AttentionLayoutKind::kBHSD) {
      return false;
    }
    const auto kind = AttentionPrefillModelStackSm80InferenceKernelOverride();
    if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return TryLaunchModelStackSm80InferenceAttentionPrefillF16(
          kind, q_contig, k_contig, v_contig, out, desc, scale_value, stream);
    } else {
      return TryLaunchModelStackSm80InferenceAttentionPrefillBF16(
          kind, q_contig, k_contig, v_contig, out, desc, scale_value, stream);
    }
  }
}

}  // namespace t10::cuda::attention
