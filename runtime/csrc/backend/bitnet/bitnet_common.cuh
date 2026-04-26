#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <sm_61_intrinsics.h>

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
  const auto* prop = t10::cuda::CachedDeviceProperties(reference);
  if (prop == nullptr) {
    return 0;
  }
  return std::max(1, static_cast<int>(prop->multiProcessorCount));
}

inline bool SupportsScaleGranularity(const LayoutInfo& layout) {
  return layout.scale_granularity == 0 || layout.scale_granularity == 1 || layout.scale_granularity == 2;
}

inline int BitNetQuantMax(int64_t bits) {
  return (static_cast<int>(1) << (static_cast<int>(bits) - 1)) - 1;
}

template <typename scalar_t>
__device__ inline float BitNetInputValueAfterPreScaleDevice(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    int64_t index,
    int64_t col) {
  float value = static_cast<float>(x[index]);
  if (pre_scale != nullptr) {
    value /= static_cast<float>(pre_scale[col]);
    value = static_cast<float>(static_cast<scalar_t>(value));
  }
  return value;
}

__device__ inline float ResolveInputScaleDevice(
    int64_t row,
    const float* __restrict__ input_scale,
    int64_t input_scale_rows) {
  return fmaxf(input_scale[input_scale_rows == 1 ? 0 : row], 1.0e-8f);
}

__device__ inline int8_t BitNetQuantizeStaticInputCodeDevice(
    float value,
    float scale,
    int qmax) {
  const float scaled = value / scale;
  const float rounded = nearbyintf(scaled);
  const float clamped = fminf(static_cast<float>(qmax), fmaxf(-static_cast<float>(qmax), rounded));
  return static_cast<int8_t>(clamped);
}

template <typename scalar_t>
__device__ inline scalar_t BitNetFakeQuantizedStaticInputValueDevice(
    float value,
    float scale,
    int qmax) {
  const float code = static_cast<float>(BitNetQuantizeStaticInputCodeDevice(value, scale, qmax));
  return static_cast<scalar_t>(code * scale);
}

template <typename scalar_t>
__device__ inline int8_t BitNetQuantizeStaticInputCodeDevice(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ input_scale,
    int64_t input_scale_rows,
    int qmax,
    int64_t row,
    int64_t col,
    int64_t index) {
  const float value = BitNetInputValueAfterPreScaleDevice(x, pre_scale, index, col);
  const float scale = ResolveInputScaleDevice(row, input_scale, input_scale_rows);
  return BitNetQuantizeStaticInputCodeDevice(value, scale, qmax);
}

template <typename scalar_t>
__device__ inline scalar_t BitNetFakeQuantizedStaticInputValueDevice(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ input_scale,
    int64_t input_scale_rows,
    int qmax,
    int64_t row,
    int64_t col,
    int64_t index) {
  const float scale = ResolveInputScaleDevice(row, input_scale, input_scale_rows);
  const float code = static_cast<float>(
      BitNetQuantizeStaticInputCodeDevice(
          x,
          pre_scale,
          input_scale,
          input_scale_rows,
          qmax,
          row,
          col,
          index));
  return static_cast<scalar_t>(code * scale);
}

__device__ inline int32_t BitNetPackInt8x4Device(const int8_t* __restrict__ values) {
  return static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint8_t>(values[0])) |
                              (static_cast<uint32_t>(static_cast<uint8_t>(values[1])) << 8) |
                              (static_cast<uint32_t>(static_cast<uint8_t>(values[2])) << 16) |
                              (static_cast<uint32_t>(static_cast<uint8_t>(values[3])) << 24));
}

__device__ inline int BitNetDotInt8Chunk4Device(
    const int8_t* __restrict__ lhs,
    const int8_t* __restrict__ rhs,
    int acc) {
  const int32_t lhs_packed = BitNetPackInt8x4Device(lhs);
  const int32_t rhs_packed = BitNetPackInt8x4Device(rhs);
#if __CUDA_ARCH__ >= 610
  return __dp4a(lhs_packed, rhs_packed, acc);
#else
  acc += static_cast<int>(lhs[0]) * static_cast<int>(rhs[0]);
  acc += static_cast<int>(lhs[1]) * static_cast<int>(rhs[1]);
  acc += static_cast<int>(lhs[2]) * static_cast<int>(rhs[2]);
  acc += static_cast<int>(lhs[3]) * static_cast<int>(rhs[3]);
  return acc;
#endif
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

__device__ inline int32_t DecodeSignedTernaryPackInt8x4Device(uint8_t packed_value, int valid_values = 4) {
  int8_t values[4] = {0, 0, 0, 0};
  #pragma unroll
  for (int offset = 0; offset < 4; ++offset) {
    if (offset >= valid_values) {
      break;
    }
    values[offset] = static_cast<int8_t>(DecodeSignedTernaryCode(packed_value, offset));
  }
  return BitNetPackInt8x4Device(values);
}

__device__ inline int BitNetDotPackedInt8Chunk4Device(
    int32_t lhs_packed,
    int32_t rhs_packed,
    int acc) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(lhs_packed, rhs_packed, acc);
#else
  union PackedInt8x4 {
    int32_t packed;
    int8_t values[4];
  };
  PackedInt8x4 lhs{lhs_packed};
  PackedInt8x4 rhs{rhs_packed};
  acc += static_cast<int>(lhs.values[0]) * static_cast<int>(rhs.values[0]);
  acc += static_cast<int>(lhs.values[1]) * static_cast<int>(rhs.values[1]);
  acc += static_cast<int>(lhs.values[2]) * static_cast<int>(rhs.values[2]);
  acc += static_cast<int>(lhs.values[3]) * static_cast<int>(rhs.values[3]);
  return acc;
#endif
}

template <typename scalar_t>
__device__ inline float BitNetDotFloatPackedTernaryChunk4Device(
    const scalar_t* __restrict__ lhs,
    uint8_t packed_value,
    int valid_values,
    float acc) {
  #pragma unroll
  for (int offset = 0; offset < 4; ++offset) {
    if (offset >= valid_values) {
      break;
    }
    acc += static_cast<float>(lhs[offset]) * static_cast<float>(DecodeSignedTernaryCode(packed_value, offset));
  }
  return acc;
}

template <typename scalar_t>
__device__ inline float BitNetDotFloatPackedTernaryWord16Device(
    const scalar_t* __restrict__ lhs,
    uint32_t packed_word,
    int valid_values,
    float acc) {
  #pragma unroll
  for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {
    const int consumed = byte_idx * 4;
    if (consumed >= valid_values) {
      break;
    }
    const int valid_chunk = valid_values - consumed >= 4 ? 4 : (valid_values - consumed);
    const uint8_t packed_value = static_cast<uint8_t>((packed_word >> (byte_idx * 8)) & 0xffu);
    acc = BitNetDotFloatPackedTernaryChunk4Device(lhs + consumed, packed_value, valid_chunk, acc);
  }
  return acc;
}

__device__ inline uint32_t LoadPackedTernaryWord16Device(
    const uint8_t* __restrict__ packed_row,
    int word_idx) {
  return reinterpret_cast<const uint32_t*>(packed_row)[word_idx];
}

__device__ inline uint32_t LoadComputePackedTernaryWord16Device(
    const int32_t* __restrict__ compute_packed_words,
    int64_t compute_word_cols,
    int64_t compute_tile_n,
    int64_t global_col,
    int64_t global_word_idx) {
  const int64_t tile_group = global_col / compute_tile_n;
  const int64_t tile_col = global_col - tile_group * compute_tile_n;
  return static_cast<uint32_t>(
      compute_packed_words[(tile_group * compute_word_cols + global_word_idx) * compute_tile_n + tile_col]);
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

__device__ inline float ResolveComputePackedRowScaleDevice(
    int64_t out_idx,
    const float* compute_row_scales,
    int64_t compute_tile_n) {
  const int64_t tile_group = out_idx / compute_tile_n;
  const int64_t tile_col = out_idx - tile_group * compute_tile_n;
  return compute_row_scales[tile_group * compute_tile_n + tile_col];
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

void LaunchBitNetDecodeKernelComputePacked(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetDecodeKernelComputePackedRmsNorm(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const c10::optional<torch::Tensor>& rms_weight,
    double eps,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetDecodeKernelComputePackedAddRmsNorm(
    torch::Tensor& combined_2d,
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& update_2d,
    const c10::optional<torch::Tensor>& rms_weight,
    double residual_scale,
    double eps,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetDecodeKernelBitplaneRow1(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetDecodeKernelBitplaneRow1RmsNorm(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const c10::optional<torch::Tensor>& rms_weight,
    double eps,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetDecodeKernelBitplaneRow1AddRmsNorm(
    torch::Tensor& combined_2d,
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& update_2d,
    const c10::optional<torch::Tensor>& rms_weight,
    double residual_scale,
    double eps,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetPrefillKernelComputePacked(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetDecodeKernelStaticInput(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const c10::optional<torch::Tensor>& pre_scale,
    const torch::Tensor& input_scale,
    int64_t act_quant_bits,
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

void LaunchBitNetPrefillKernelStaticInput(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const c10::optional<torch::Tensor>& pre_scale,
    const torch::Tensor& input_scale,
    int64_t act_quant_bits,
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

void LaunchBitNetPrefillSplitKKernelComputePacked(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan);

void LaunchBitNetPrefillSplitKKernelStaticInput(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const c10::optional<torch::Tensor>& pre_scale,
    const torch::Tensor& input_scale,
    int64_t act_quant_bits,
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

torch::Tensor CudaBitNetLinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale,
    const c10::optional<torch::ScalarType>& out_dtype);

torch::Tensor CudaBitNetTransformInputForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale);

torch::Tensor CudaBitNetCalibrateInputScaleForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    int64_t act_quant_bits);

std::tuple<torch::Tensor, torch::Tensor> CudaBitNetQuantizeGatedActivationInt8CodesForward(
    const torch::Tensor& x,
    const std::string& activation,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale);

bool HasCudaBitNetLinearKernel();
bool HasCudaBitNetInputFrontendKernel();

}  // namespace t10::bitnet
