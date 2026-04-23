#pragma once

#include <torch/extension.h>

#include "../cuda_device_arch.cuh"
#include "../../descriptors/attention_desc.h"

#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace t10::cuda::attention {

inline bool DeviceIsSm80OrLaterForPyTorchMemEff(const torch::Tensor& reference) {
  int major = 0;
  return t10::cuda::DeviceComputeCapability(reference, &major, nullptr) && major >= 8;
}

inline bool AttentionPrefillPyTorchMemEffDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_PYTORCH_MEMEFF");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

enum class PyTorchMemEffPrefillKernelKind {
  kAuto,
  k64x64Rf,
  k64x128Rf,
  k32x128Gmem,
};

inline PyTorchMemEffPrefillKernelKind AttentionPrefillPyTorchMemEffKernelOverride() {
  const char* env = std::getenv("MODEL_STACK_PYTORCH_MEMEFF_PREFILL_KERNEL");
  if (env == nullptr || env[0] == '\0') {
    return PyTorchMemEffPrefillKernelKind::kAuto;
  }
  if (std::strcmp(env, "64x64_rf") == 0) {
    return PyTorchMemEffPrefillKernelKind::k64x64Rf;
  }
  if (std::strcmp(env, "64x128_rf") == 0) {
    return PyTorchMemEffPrefillKernelKind::k64x128Rf;
  }
  if (std::strcmp(env, "32x128_gmem") == 0) {
    return PyTorchMemEffPrefillKernelKind::k32x128Gmem;
  }
  return PyTorchMemEffPrefillKernelKind::kAuto;
}

bool TryLaunchPyTorchMemEffAttentionPrefillF16(
    PyTorchMemEffPrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream);

bool TryLaunchPyTorchMemEffAttentionPrefillBF16(
    PyTorchMemEffPrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream);

template <typename scalar_t, int HeadDim>
inline bool TryLaunchPyTorchMemEffAttentionPrefill(
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
    if (AttentionPrefillPyTorchMemEffDisabled()) {
      return false;
    }
    if (!DeviceIsSm80OrLaterForPyTorchMemEff(q_contig) ||
        t10::cuda::DeviceIsSm90OrLater(q_contig)) {
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
    const auto kind = AttentionPrefillPyTorchMemEffKernelOverride();
    if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return TryLaunchPyTorchMemEffAttentionPrefillF16(
          kind, q_contig, k_contig, v_contig, out, desc, scale_value, stream);
    } else {
      return TryLaunchPyTorchMemEffAttentionPrefillBF16(
          kind, q_contig, k_contig, v_contig, out, desc, scale_value, stream);
    }
  }
}

}  // namespace t10::cuda::attention
