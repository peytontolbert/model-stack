#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <tuple>
#include <algorithm>
#include <cstdlib>

#include "../cuda_device_arch.cuh"
#include "bitnet_formats.h"

namespace t10::bitnet {

constexpr int kDecodeThreads = 256;
constexpr int kPrefillThreadsX = 32;
constexpr int kPrefillThreadsY = 8;
constexpr int kPrefillTileRows = kPrefillThreadsY;
constexpr int kPrefillGenericTileK = 64;
constexpr int kPrefillSm80TileK = 128;
constexpr int kPrefillSm80ColsPerThread = 2;

enum class KernelKind {
  kDecodePersistent,
  kPrefillTiled,
  kPrefillSplitK,
};

struct Plan {
  KernelKind kind = KernelKind::kPrefillTiled;
  int rows_bucket = 1;
  int tile_k = kPrefillGenericTileK;
  int outputs_per_block = kPrefillThreadsX;
  int cols_per_thread = 1;
  int split_k_slices = 1;
  int persistent_ctas = 1;
};

inline bool IsSupportedLinearDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

inline bool DeviceIsSm80OrLater(const torch::Tensor& reference) {
  int major = 0;
  return t10::cuda::DeviceComputeCapability(reference, &major, nullptr) && major >= 8;
}

inline int DeviceMultiProcessorCount(const torch::Tensor& reference) {
  if (!reference.is_cuda()) {
    return 0;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, reference.get_device()) != cudaSuccess) {
    return 0;
  }
  return std::max(1, static_cast<int>(prop.multiProcessorCount));
}

inline bool SupportsScaleGranularity(const LayoutInfo& layout) {
  return layout.scale_granularity == 0 || layout.scale_granularity == 1 || layout.scale_granularity == 2;
}

inline bool BitNetPersistentDecodeDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_BITNET_PERSISTENT_DECODE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool BitNetSplitKDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_BITNET_SPLITK");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline int BitNetDecodeCtaMultiplier() {
  const char* env = std::getenv("MODEL_STACK_BITNET_DECODE_CTA_MULTIPLIER");
  if (env == nullptr || env[0] == '\0') {
    return 2;
  }
  return std::max(1, static_cast<int>(std::strtol(env, nullptr, 10)));
}

inline int BitNetSplitKMaxSlices() {
  const char* env = std::getenv("MODEL_STACK_BITNET_SPLITK_MAX_SLICES");
  if (env == nullptr || env[0] == '\0') {
    return 8;
  }
  return std::max(1, static_cast<int>(std::strtol(env, nullptr, 10)));
}

inline const char* KernelKindName(KernelKind kind) {
  switch (kind) {
    case KernelKind::kDecodePersistent:
      return "decode_persistent";
    case KernelKind::kPrefillTiled:
      return "prefill_tiled";
    case KernelKind::kPrefillSplitK:
      return "prefill_splitk";
  }
  return "unknown";
}

inline Plan ResolvePlan(const torch::Tensor& x_2d, const LayoutInfo& layout) {
  const int64_t rows = x_2d.size(0);
  const int64_t out_features = layout.logical_out_features;
  const int64_t in_features = layout.logical_in_features;
  const bool sm80 = DeviceIsSm80OrLater(x_2d);
  const int sm_count = DeviceMultiProcessorCount(x_2d);

  Plan plan;
  if (rows > 0 && rows <= 8 && !BitNetPersistentDecodeDisabled()) {
    plan.kind = KernelKind::kDecodePersistent;
    plan.rows_bucket = rows <= 1 ? 1 : (rows <= 2 ? 2 : (rows <= 4 ? 4 : 8));
    plan.tile_k = sm80 && in_features >= kPrefillSm80TileK ? kPrefillSm80TileK : kPrefillGenericTileK;
    plan.outputs_per_block = plan.rows_bucket <= 2 ? 128 : 64;
    const int64_t total_tiles = (out_features + plan.outputs_per_block - 1) / plan.outputs_per_block;
    const int target_ctas = std::max(1, sm_count) * BitNetDecodeCtaMultiplier();
    plan.persistent_ctas = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(total_tiles, target_ctas)));
    return plan;
  }

  if (!BitNetSplitKDisabled() && rows >= 8 && in_features >= 4096) {
    const int64_t workspace_elements = rows * out_features;
    const int max_slices = BitNetSplitKMaxSlices();
    const int suggested = std::max(2, static_cast<int>(in_features / 4096));
    const int slices = std::min(max_slices, suggested);
    if (slices > 1 && workspace_elements > 0 && workspace_elements <= 4'194'304) {
      plan.kind = KernelKind::kPrefillSplitK;
      plan.tile_k = sm80 && in_features >= kPrefillSm80TileK ? kPrefillSm80TileK : kPrefillGenericTileK;
      plan.cols_per_thread = sm80 ? kPrefillSm80ColsPerThread : 1;
      plan.outputs_per_block = kPrefillThreadsX * plan.cols_per_thread;
      plan.split_k_slices = slices;
      return plan;
    }
  }

  plan.kind = KernelKind::kPrefillTiled;
  plan.tile_k = sm80 && in_features >= kPrefillSm80TileK ? kPrefillSm80TileK : kPrefillGenericTileK;
  plan.cols_per_thread = sm80 ? kPrefillSm80ColsPerThread : 1;
  plan.outputs_per_block = kPrefillThreadsX * plan.cols_per_thread;
  return plan;
}

__device__ inline int DecodeSignedTernaryCode(uint8_t packed_value, int64_t in_idx) {
  const int code = static_cast<int>((packed_value >> ((in_idx & 0x03) * 2)) & 0x03);
  return code == 0 ? -1 : (code == 2 ? 1 : 0);
}

__device__ inline float ResolveRowScaleDevice(
    int64_t out_idx,
    const float* scale_values,
    const int32_t* segment_offsets,
    const LayoutInfo& layout) {
  if (layout.scale_granularity == 0) {
    return scale_values[0];
  }
  if (layout.scale_granularity == 1) {
    for (int64_t seg_idx = 0; seg_idx < layout.segment_count; ++seg_idx) {
      if (out_idx >= segment_offsets[seg_idx] && out_idx < segment_offsets[seg_idx + 1]) {
        return scale_values[seg_idx];
      }
    }
    return 0.0f;
  }
  if (layout.scale_granularity == 2) {
    const int64_t group_idx = out_idx / layout.scale_group_size;
    return scale_values[group_idx];
  }
  return 0.0f;
}

void LaunchBitNetDecodeKernel(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetPrefillKernel(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetPrefillSplitKKernel(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CudaPackBitNetWeightForward(
    const torch::Tensor& weight);

torch::Tensor CudaBitNetLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);

bool HasCudaBitNetLinearKernel();

}  // namespace t10::bitnet
