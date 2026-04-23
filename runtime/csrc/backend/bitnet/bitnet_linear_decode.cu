#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include "bitnet_common.cuh"
#include "bitnet_epilogue.cuh"

namespace t10::bitnet {
namespace {

__device__ inline float WarpReduceSumDevice(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void bitnet_linear_decode_scalar_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ scale_values,
    const int32_t* __restrict__ segment_offsets,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t rows,
    int64_t packed_cols) {
  const int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= layout.logical_out_features) {
    return;
  }
  const float row_scale = ResolveRowScaleDevice(out_idx, scale_values, segment_offsets, layout);
  const uint8_t* weight_row = packed_weight + (out_idx * packed_cols);
  for (int64_t row = 0; row < rows; ++row) {
    const scalar_t* x_row = x + (row * layout.logical_in_features);
    float acc = 0.0f;
    for (int64_t in_idx = 0; in_idx < layout.logical_in_features; ++in_idx) {
      const uint8_t packed_value = weight_row[in_idx >> 2];
      const int q = DecodeSignedTernaryCode(packed_value, in_idx);
      acc += static_cast<float>(x_row[in_idx]) * static_cast<float>(q);
    }
    acc *= row_scale;
    if (bias != nullptr) {
      acc += static_cast<float>(bias[out_idx]);
    }
    out[row * layout.logical_out_features + out_idx] = CastOutput<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void bitnet_linear_decode_scalar_static_input_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ input_scale,
    int64_t input_scale_rows,
    int qmax,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ scale_values,
    const int32_t* __restrict__ segment_offsets,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t rows,
    int64_t packed_cols) {
  const int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= layout.logical_out_features) {
    return;
  }
  const float row_scale = ResolveRowScaleDevice(out_idx, scale_values, segment_offsets, layout);
  const uint8_t* weight_row = packed_weight + (out_idx * packed_cols);
  for (int64_t row = 0; row < rows; ++row) {
    const float scale = ResolveInputScaleDevice(row, input_scale, input_scale_rows);
    int acc = 0;
    for (int64_t in_idx = 0; in_idx < layout.logical_in_features; in_idx += 4) {
      int8_t x_chunk[4] = {0, 0, 0, 0};
      int8_t w_chunk[4] = {0, 0, 0, 0};
      const uint8_t packed_value = weight_row[in_idx >> 2];
      #pragma unroll
      for (int offset = 0; offset < 4; ++offset) {
        const int64_t global_k = in_idx + offset;
        if (global_k >= layout.logical_in_features) {
          break;
        }
        w_chunk[offset] = static_cast<int8_t>(DecodeSignedTernaryCode(packed_value, global_k));
        const float input_value = BitNetInputValueAfterPreScaleDevice(
            x,
            pre_scale,
            row * layout.logical_in_features + global_k,
            global_k);
        x_chunk[offset] = BitNetQuantizeStaticInputCodeDevice(input_value, scale, qmax);
      }
      acc = BitNetDotInt8Chunk4Device(x_chunk, w_chunk, acc);
    }
    float value = static_cast<float>(acc) * scale * row_scale;
    if (bias != nullptr) {
      value += static_cast<float>(bias[out_idx]);
    }
    out[row * layout.logical_out_features + out_idx] = CastOutput<scalar_t>(value);
  }
}

template <typename scalar_t, int TileK>
__global__ __launch_bounds__(32, 8) void bitnet_linear_decode_row1_static_input_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ input_scale,
    int64_t input_scale_rows,
    int qmax,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ scale_values,
    const int32_t* __restrict__ segment_offsets,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t packed_cols) {
  constexpr int kWarpSize = 32;
  constexpr int kPackedTileK = (TileK + 3) / 4;

  __shared__ int32_t x_tile_packed[kPackedTileK];

  const int lane = static_cast<int>(threadIdx.x);
  const int64_t global_col = static_cast<int64_t>(blockIdx.x) * kWarpSize + lane;
  const float input_scale_value = ResolveInputScaleDevice(0, input_scale, input_scale_rows);

  int acc = 0;
  for (int64_t k0 = 0; k0 < layout.logical_in_features; k0 += TileK) {
    for (int packed_k = lane; packed_k < kPackedTileK; packed_k += kWarpSize) {
      const int tile_k0 = packed_k * 4;
      const int64_t global_k0 = k0 + tile_k0;
      int8_t x_chunk[4] = {0, 0, 0, 0};
      if (global_k0 < layout.logical_in_features) {
        const int valid_values = static_cast<int>(
            (layout.logical_in_features - global_k0) >= 4 ? 4 : (layout.logical_in_features - global_k0));
        #pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
          if (offset >= valid_values) {
            break;
          }
          const int64_t global_k = global_k0 + offset;
          const float input_value = BitNetInputValueAfterPreScaleDevice(
              x,
              pre_scale,
              global_k,
              global_k);
          x_chunk[offset] = BitNetQuantizeStaticInputCodeDevice(input_value, input_scale_value, qmax);
        }
      }
      x_tile_packed[packed_k] = BitNetPackInt8x4Device(x_chunk);
    }
    __syncthreads();

    if (global_col < layout.logical_out_features) {
      const uint8_t* weight_row = packed_weight + (global_col * packed_cols) + (k0 >> 2);
      #pragma unroll
      for (int packed_k = 0; packed_k < kPackedTileK; ++packed_k) {
        const int64_t global_k0 = k0 + packed_k * 4;
        if (global_k0 >= layout.logical_in_features) {
          break;
        }
        const int valid_values = static_cast<int>(
            (layout.logical_in_features - global_k0) >= 4 ? 4 : (layout.logical_in_features - global_k0));
        acc = BitNetDotPackedInt8Chunk4Device(
            x_tile_packed[packed_k],
            DecodeSignedTernaryPackInt8x4Device(weight_row[packed_k], valid_values),
            acc);
      }
    }
    __syncthreads();
  }

  if (global_col >= layout.logical_out_features) {
    return;
  }
  float value = static_cast<float>(acc) * input_scale_value *
      ResolveRowScaleDevice(global_col, scale_values, segment_offsets, layout);
  if (bias != nullptr) {
    value += static_cast<float>(bias[global_col]);
  }
  out[global_col] = CastOutput<scalar_t>(value);
}

template <typename scalar_t, int WarpsPerBlock>
__global__ __launch_bounds__(32 * WarpsPerBlock, 2) void bitnet_linear_decode_row1_bitplane_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ decode_nz_masks,
    const int32_t* __restrict__ decode_sign_masks,
    const float* __restrict__ decode_row_scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t decode_chunk_cols,
    int64_t decode_tile_n) {
  constexpr int kWarpSize = 32;

  const int lane = static_cast<int>(threadIdx.x);
  const int warp_idx = static_cast<int>(threadIdx.y);
  const int64_t out_idx = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_idx;
  const bool output_active = out_idx < layout.logical_out_features;
  const int64_t tile_group = output_active ? out_idx / decode_tile_n : 0;
  const int64_t tile_col = output_active ? out_idx - tile_group * decode_tile_n : 0;
  const int64_t mask_base = output_active ? ((tile_group * decode_chunk_cols) * decode_tile_n + tile_col) : 0;
  const uint32_t lane_bit = 1u << lane;

  float acc = 0.0f;
  for (int64_t chunk_idx = 0; chunk_idx < decode_chunk_cols; ++chunk_idx) {
    if (output_active) {
      const int64_t global_k = chunk_idx * kWarpSize + lane;
      const float x_value =
          global_k < layout.logical_in_features
              ? static_cast<float>(x[global_k])
              : 0.0f;
      const int64_t mask_idx = mask_base + chunk_idx * decode_tile_n;
      const uint32_t nz_mask = static_cast<uint32_t>(decode_nz_masks[mask_idx]);
      const uint32_t sign_mask = static_cast<uint32_t>(decode_sign_masks[mask_idx]);

      float contrib = 0.0f;
      if ((nz_mask & lane_bit) != 0u) {
        contrib = x_value;
        if ((sign_mask & lane_bit) == 0u) {
          contrib = -contrib;
        }
      }
      contrib = WarpReduceSumDevice(contrib);
      if (lane == 0) {
        acc += contrib;
      }
    }
  }

  if (output_active && lane == 0) {
    float value = acc * ResolveComputePackedRowScaleDevice(out_idx, decode_row_scales, decode_tile_n);
    if (bias != nullptr) {
      value += static_cast<float>(bias[out_idx]);
    }
    out[out_idx] = CastOutput<scalar_t>(value);
  }
}

template <typename scalar_t, int WarpsPerBlock>
__global__ __launch_bounds__(32 * WarpsPerBlock, 2) void bitnet_linear_decode_row1_bitplane_rms_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ rms_weight,
    float eps,
    const int32_t* __restrict__ decode_nz_masks,
    const int32_t* __restrict__ decode_sign_masks,
    const float* __restrict__ decode_row_scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t decode_chunk_cols,
    int64_t decode_tile_n) {
  constexpr int kWarpSize = 32;
  constexpr int kThreads = kWarpSize * WarpsPerBlock;

  __shared__ float warp_sums[WarpsPerBlock];
  __shared__ float inv_rms_shared;

  const int lane = static_cast<int>(threadIdx.x);
  const int warp_idx = static_cast<int>(threadIdx.y);
  const int thread_linear = warp_idx * kWarpSize + lane;
  const int64_t out_idx = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_idx;
  const bool output_active = out_idx < layout.logical_out_features;
  const int64_t tile_group = output_active ? out_idx / decode_tile_n : 0;
  const int64_t tile_col = output_active ? out_idx - tile_group * decode_tile_n : 0;
  const int64_t mask_base = output_active ? ((tile_group * decode_chunk_cols) * decode_tile_n + tile_col) : 0;
  const uint32_t lane_bit = 1u << lane;

  float thread_sum = 0.0f;
  for (int64_t col = thread_linear; col < layout.logical_in_features; col += kThreads) {
    const float value = static_cast<float>(x[col]);
    thread_sum += value * value;
  }
  thread_sum = WarpReduceSumDevice(thread_sum);
  if (lane == 0) {
    warp_sums[warp_idx] = thread_sum;
  }
  __syncthreads();

  if (warp_idx == 0) {
    float block_sum = lane < WarpsPerBlock ? warp_sums[lane] : 0.0f;
    block_sum = WarpReduceSumDevice(block_sum);
    if (lane == 0) {
      inv_rms_shared = rsqrtf((block_sum / static_cast<float>(layout.logical_in_features)) + eps);
    }
  }
  __syncthreads();
  const float inv_rms = inv_rms_shared;

  float acc = 0.0f;
  for (int64_t chunk_idx = 0; chunk_idx < decode_chunk_cols; ++chunk_idx) {
    if (output_active) {
      const int64_t global_k = chunk_idx * kWarpSize + lane;
      float contrib = 0.0f;
      if (global_k < layout.logical_in_features) {
        contrib = static_cast<float>(x[global_k]) * inv_rms;
        if (rms_weight != nullptr) {
          contrib *= static_cast<float>(rms_weight[global_k]);
        }
        const int64_t mask_idx = mask_base + chunk_idx * decode_tile_n;
        const uint32_t nz_mask = static_cast<uint32_t>(decode_nz_masks[mask_idx]);
        if ((nz_mask & lane_bit) == 0u) {
          contrib = 0.0f;
        } else {
          const uint32_t sign_mask = static_cast<uint32_t>(decode_sign_masks[mask_idx]);
          if ((sign_mask & lane_bit) == 0u) {
            contrib = -contrib;
          }
        }
      } else {
        const int64_t mask_idx = mask_base + chunk_idx * decode_tile_n;
        const uint32_t nz_mask = static_cast<uint32_t>(decode_nz_masks[mask_idx]);
        if ((nz_mask & lane_bit) != 0u) {
          const uint32_t sign_mask = static_cast<uint32_t>(decode_sign_masks[mask_idx]);
          contrib = (sign_mask & lane_bit) != 0u ? 0.0f : -0.0f;
        }
      }
      contrib = WarpReduceSumDevice(contrib);
      if (lane == 0) {
        acc += contrib;
      }
    }
  }

  if (output_active && lane == 0) {
    float value = acc * ResolveComputePackedRowScaleDevice(out_idx, decode_row_scales, decode_tile_n);
    if (bias != nullptr) {
      value += static_cast<float>(bias[out_idx]);
    }
    out[out_idx] = CastOutput<scalar_t>(value);
  }
}

template <typename scalar_t, int WarpsPerBlock>
__global__ __launch_bounds__(32 * WarpsPerBlock, 2) void bitnet_linear_decode_row1_bitplane_add_rms_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ rms_weight,
    float residual_scale,
    float eps,
    const int32_t* __restrict__ decode_nz_masks,
    const int32_t* __restrict__ decode_sign_masks,
    const float* __restrict__ decode_row_scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ combined,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t decode_chunk_cols,
    int64_t decode_tile_n) {
  constexpr int kWarpSize = 32;
  constexpr int kThreads = kWarpSize * WarpsPerBlock;

  __shared__ float warp_sums[WarpsPerBlock];
  __shared__ float inv_rms_shared;

  const int lane = static_cast<int>(threadIdx.x);
  const int warp_idx = static_cast<int>(threadIdx.y);
  const int thread_linear = warp_idx * kWarpSize + lane;
  const int64_t out_idx = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_idx;
  const bool output_active = out_idx < layout.logical_out_features;
  const int64_t tile_group = output_active ? out_idx / decode_tile_n : 0;
  const int64_t tile_col = output_active ? out_idx - tile_group * decode_tile_n : 0;
  const int64_t mask_base = output_active ? ((tile_group * decode_chunk_cols) * decode_tile_n + tile_col) : 0;
  const uint32_t lane_bit = 1u << lane;

  float thread_sum = 0.0f;
  for (int64_t col = thread_linear; col < layout.logical_in_features; col += kThreads) {
    const float value =
        static_cast<float>(x[col]) + (residual_scale * static_cast<float>(update[col]));
    const scalar_t combined_value = static_cast<scalar_t>(value);
    combined[col] = combined_value;
    const float rounded = static_cast<float>(combined_value);
    thread_sum += rounded * rounded;
  }
  thread_sum = WarpReduceSumDevice(thread_sum);
  if (lane == 0) {
    warp_sums[warp_idx] = thread_sum;
  }
  __syncthreads();

  if (warp_idx == 0) {
    float block_sum = lane < WarpsPerBlock ? warp_sums[lane] : 0.0f;
    block_sum = WarpReduceSumDevice(block_sum);
    if (lane == 0) {
      inv_rms_shared = rsqrtf((block_sum / static_cast<float>(layout.logical_in_features)) + eps);
    }
  }
  __syncthreads();
  const float inv_rms = inv_rms_shared;

  float acc = 0.0f;
  for (int64_t chunk_idx = 0; chunk_idx < decode_chunk_cols; ++chunk_idx) {
    if (output_active) {
      float contrib = 0.0f;
      const int64_t global_k = chunk_idx * kWarpSize + lane;
      if (global_k < layout.logical_in_features) {
        contrib = static_cast<float>(combined[global_k]) * inv_rms;
        if (rms_weight != nullptr) {
          contrib *= static_cast<float>(rms_weight[global_k]);
        }
        const int64_t mask_idx = mask_base + chunk_idx * decode_tile_n;
        const uint32_t nz_mask = static_cast<uint32_t>(decode_nz_masks[mask_idx]);
        if ((nz_mask & lane_bit) == 0u) {
          contrib = 0.0f;
        } else {
          const uint32_t sign_mask = static_cast<uint32_t>(decode_sign_masks[mask_idx]);
          if ((sign_mask & lane_bit) == 0u) {
            contrib = -contrib;
          }
        }
      } else {
        const int64_t mask_idx = mask_base + chunk_idx * decode_tile_n;
        const uint32_t nz_mask = static_cast<uint32_t>(decode_nz_masks[mask_idx]);
        if ((nz_mask & lane_bit) != 0u) {
          const uint32_t sign_mask = static_cast<uint32_t>(decode_sign_masks[mask_idx]);
          contrib = (sign_mask & lane_bit) != 0u ? 0.0f : -0.0f;
        }
      }
      contrib = WarpReduceSumDevice(contrib);
      if (lane == 0) {
        acc += contrib;
      }
    }
  }

  if (output_active && lane == 0) {
    float value = acc * ResolveComputePackedRowScaleDevice(out_idx, decode_row_scales, decode_tile_n);
    if (bias != nullptr) {
      value += static_cast<float>(bias[out_idx]);
    }
    out[out_idx] = CastOutput<scalar_t>(value);
  }
}

template <typename scalar_t, int MaxRows, int TileK, int OutputsPerBlock>
__global__ __launch_bounds__(32 * MaxRows, 2) void bitnet_linear_decode_persistent_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ scale_values,
    const int32_t* __restrict__ segment_offsets,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t rows,
    int64_t packed_cols,
    int64_t total_tiles) {
  constexpr int kWarpSize = 32;
  constexpr int kColsPerLane = OutputsPerBlock / kWarpSize;
  constexpr int kThreads = kWarpSize * MaxRows;
  constexpr int kPackedTileK = (TileK + 3) / 4;
  constexpr int kPackedTileKWords = (kPackedTileK + 3) / 4;

  __shared__ scalar_t x_tile[MaxRows][TileK];
  __shared__ uint32_t w_tile_words[OutputsPerBlock][kPackedTileKWords];
  __shared__ float row_scales[OutputsPerBlock];

  const int lane = static_cast<int>(threadIdx.x);
  const int row_slot = static_cast<int>(threadIdx.y);
  const int thread_linear = row_slot * kWarpSize + lane;
  const bool row_active = row_slot < rows;

  for (int64_t tile_idx = static_cast<int64_t>(blockIdx.x); tile_idx < total_tiles; tile_idx += gridDim.x) {
    const int64_t col_base = tile_idx * OutputsPerBlock;
    float accum[kColsPerLane];
    #pragma unroll
    for (int idx = 0; idx < kColsPerLane; ++idx) {
      accum[idx] = 0.0f;
    }

    for (int idx = thread_linear; idx < OutputsPerBlock; idx += kThreads) {
      const int64_t global_col = col_base + idx;
      row_scales[idx] =
          global_col < layout.logical_out_features
              ? ResolveRowScaleDevice(global_col, scale_values, segment_offsets, layout)
              : 0.0f;
    }
    __syncthreads();

    for (int64_t k0 = 0; k0 < layout.logical_in_features; k0 += TileK) {
      for (int index = thread_linear; index < MaxRows * TileK; index += kThreads) {
        const int tile_row = index / TileK;
        const int tile_k = index % TileK;
        const int64_t global_k = k0 + tile_k;
        x_tile[tile_row][tile_k] =
            (tile_row < rows && global_k < layout.logical_in_features)
                ? x[tile_row * layout.logical_in_features + global_k]
                : static_cast<scalar_t>(0);
      }
      for (int index = thread_linear; index < OutputsPerBlock * kPackedTileKWords; index += kThreads) {
        const int tile_col = index / kPackedTileKWords;
        const int word_idx = index % kPackedTileKWords;
        const int tile_k0 = word_idx * 16;
        const int64_t global_col = col_base + tile_col;
        const int64_t global_k0 = k0 + tile_k0;
        uint32_t packed_word = 0;
        if (global_col < layout.logical_out_features && global_k0 < layout.logical_in_features) {
          const uint8_t* weight_row = packed_weight + (global_col * packed_cols) + (k0 >> 2);
          packed_word = LoadPackedTernaryWord16Device(weight_row, word_idx);
        }
        w_tile_words[tile_col][word_idx] = packed_word;
      }
      __syncthreads();

      if (row_active) {
        #pragma unroll
        for (int col_iter = 0; col_iter < kColsPerLane; ++col_iter) {
          const int tile_col = lane + col_iter * kWarpSize;
          const int64_t global_col = col_base + tile_col;
          if (global_col >= layout.logical_out_features) {
            continue;
          }
          float acc = accum[col_iter];
          #pragma unroll
          for (int word_idx = 0; word_idx < kPackedTileKWords; ++word_idx) {
            const int tile_k0 = word_idx * 16;
            const int64_t global_k0 = k0 + tile_k0;
            if (global_k0 >= layout.logical_in_features) {
              break;
            }
            const int valid_values = static_cast<int>(
                (layout.logical_in_features - global_k0) >= 16 ? 16 : (layout.logical_in_features - global_k0));
            acc = BitNetDotFloatPackedTernaryWord16Device(
                &x_tile[row_slot][tile_k0],
                w_tile_words[tile_col][word_idx],
                valid_values,
                acc);
          }
          accum[col_iter] = acc;
        }
      }
      __syncthreads();
    }

    if (row_active) {
      #pragma unroll
      for (int col_iter = 0; col_iter < kColsPerLane; ++col_iter) {
        const int tile_col = lane + col_iter * kWarpSize;
        const int64_t global_col = col_base + tile_col;
        if (global_col >= layout.logical_out_features) {
          continue;
        }
        float value = accum[col_iter] * row_scales[tile_col];
        if (bias != nullptr) {
          value += static_cast<float>(bias[global_col]);
        }
        out[row_slot * layout.logical_out_features + global_col] = CastOutput<scalar_t>(value);
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, int MaxRows, int TileK, int OutputsPerBlock>
__global__ __launch_bounds__(32 * MaxRows, 2) void bitnet_linear_decode_persistent_compute_packed_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ compute_packed_words,
    const float* __restrict__ compute_row_scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t rows,
    int64_t compute_word_cols,
    int64_t compute_tile_n,
    int64_t total_tiles) {
  constexpr int kWarpSize = 32;
  constexpr int kColsPerLane = OutputsPerBlock / kWarpSize;
  constexpr int kThreads = kWarpSize * MaxRows;
  constexpr int kPackedTileKWords = (TileK + 15) / 16;

  __shared__ scalar_t x_tile[MaxRows][TileK];
  __shared__ uint32_t w_tile_words[OutputsPerBlock][kPackedTileKWords];
  __shared__ float row_scales[OutputsPerBlock];

  const int lane = static_cast<int>(threadIdx.x);
  const int row_slot = static_cast<int>(threadIdx.y);
  const int thread_linear = row_slot * kWarpSize + lane;
  const bool row_active = row_slot < rows;

  for (int64_t tile_idx = static_cast<int64_t>(blockIdx.x); tile_idx < total_tiles; tile_idx += gridDim.x) {
    const int64_t col_base = tile_idx * OutputsPerBlock;
    float accum[kColsPerLane];
    #pragma unroll
    for (int idx = 0; idx < kColsPerLane; ++idx) {
      accum[idx] = 0.0f;
    }

    for (int idx = thread_linear; idx < OutputsPerBlock; idx += kThreads) {
      const int64_t global_col = col_base + idx;
      float scale = 0.0f;
      if (global_col < layout.logical_out_features) {
        const int64_t tile_group = global_col / compute_tile_n;
        const int64_t tile_col = global_col - tile_group * compute_tile_n;
        scale = compute_row_scales[tile_group * compute_tile_n + tile_col];
      }
      row_scales[idx] = scale;
    }
    __syncthreads();

    for (int64_t k0 = 0; k0 < layout.logical_in_features; k0 += TileK) {
      for (int index = thread_linear; index < MaxRows * TileK; index += kThreads) {
        const int tile_row = index / TileK;
        const int tile_k = index % TileK;
        const int64_t global_k = k0 + tile_k;
        x_tile[tile_row][tile_k] =
            (tile_row < rows && global_k < layout.logical_in_features)
                ? x[tile_row * layout.logical_in_features + global_k]
                : static_cast<scalar_t>(0);
      }
      for (int index = thread_linear; index < OutputsPerBlock * kPackedTileKWords; index += kThreads) {
        const int tile_col = index / kPackedTileKWords;
        const int word_idx = index % kPackedTileKWords;
        const int64_t global_col = col_base + tile_col;
        const int64_t global_word_idx = (k0 >> 4) + word_idx;
        uint32_t packed_word = 0;
        if (global_col < layout.logical_out_features && global_word_idx < compute_word_cols) {
          const int64_t tile_group = global_col / compute_tile_n;
          const int64_t compute_tile_col = global_col - tile_group * compute_tile_n;
          packed_word = static_cast<uint32_t>(
              compute_packed_words[(tile_group * compute_word_cols + global_word_idx) * compute_tile_n + compute_tile_col]);
        }
        w_tile_words[tile_col][word_idx] = packed_word;
      }
      __syncthreads();

      if (row_active) {
        #pragma unroll
        for (int col_iter = 0; col_iter < kColsPerLane; ++col_iter) {
          const int tile_col = lane + col_iter * kWarpSize;
          const int64_t global_col = col_base + tile_col;
          if (global_col >= layout.logical_out_features) {
            continue;
          }
          float acc = accum[col_iter];
          #pragma unroll
          for (int word_idx = 0; word_idx < kPackedTileKWords; ++word_idx) {
            const int tile_k0 = word_idx * 16;
            const int64_t global_k0 = k0 + tile_k0;
            if (global_k0 >= layout.logical_in_features) {
              break;
            }
            const int valid_values = static_cast<int>(
                (layout.logical_in_features - global_k0) >= 16 ? 16 : (layout.logical_in_features - global_k0));
            acc = BitNetDotFloatPackedTernaryWord16Device(
                &x_tile[row_slot][tile_k0],
                w_tile_words[tile_col][word_idx],
                valid_values,
                acc);
          }
          accum[col_iter] = acc;
        }
      }
      __syncthreads();
    }

    if (row_active) {
      #pragma unroll
      for (int col_iter = 0; col_iter < kColsPerLane; ++col_iter) {
        const int tile_col = lane + col_iter * kWarpSize;
        const int64_t global_col = col_base + tile_col;
        if (global_col >= layout.logical_out_features) {
          continue;
        }
        float value = accum[col_iter] * row_scales[tile_col];
        if (bias != nullptr) {
          value += static_cast<float>(bias[global_col]);
        }
        out[row_slot * layout.logical_out_features + global_col] = CastOutput<scalar_t>(value);
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, int MaxRows, int TileK, int OutputsPerBlock>
__global__ __launch_bounds__(32 * MaxRows, 2) void bitnet_linear_decode_persistent_static_input_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ input_scale,
    int64_t input_scale_rows,
    int qmax,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ scale_values,
    const int32_t* __restrict__ segment_offsets,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t rows,
    int64_t packed_cols,
    int64_t total_tiles) {
  constexpr int kWarpSize = 32;
  constexpr int kColsPerLane = OutputsPerBlock / kWarpSize;
  constexpr int kThreads = kWarpSize * MaxRows;
  constexpr int kPackedTileK = (TileK + 3) / 4;

  __shared__ __align__(16) int32_t x_tile_packed[MaxRows][kPackedTileK];
  __shared__ __align__(16) int32_t w_tile_packed[OutputsPerBlock][kPackedTileK];
  __shared__ float row_scales[OutputsPerBlock];
  __shared__ float input_scales[MaxRows];

  const int lane = static_cast<int>(threadIdx.x);
  const int row_slot = static_cast<int>(threadIdx.y);
  const int thread_linear = row_slot * kWarpSize + lane;
  const bool row_active = row_slot < rows;

  for (int idx = thread_linear; idx < MaxRows; idx += kThreads) {
    input_scales[idx] = idx < rows ? ResolveInputScaleDevice(idx, input_scale, input_scale_rows) : 1.0f;
  }
  __syncthreads();

  for (int64_t tile_idx = static_cast<int64_t>(blockIdx.x); tile_idx < total_tiles; tile_idx += gridDim.x) {
    const int64_t col_base = tile_idx * OutputsPerBlock;
    int accum[kColsPerLane];
    #pragma unroll
    for (int idx = 0; idx < kColsPerLane; ++idx) {
      accum[idx] = 0;
    }

    for (int idx = thread_linear; idx < OutputsPerBlock; idx += kThreads) {
      const int64_t global_col = col_base + idx;
      row_scales[idx] =
          global_col < layout.logical_out_features
              ? ResolveRowScaleDevice(global_col, scale_values, segment_offsets, layout)
              : 0.0f;
    }
    __syncthreads();

    for (int64_t k0 = 0; k0 < layout.logical_in_features; k0 += TileK) {
      for (int index = thread_linear; index < MaxRows * kPackedTileK; index += kThreads) {
        const int tile_row = index / kPackedTileK;
        const int packed_k = index % kPackedTileK;
        const int tile_k0 = packed_k * 4;
        const int64_t global_k0 = k0 + tile_k0;
        int8_t x_chunk[4] = {0, 0, 0, 0};
        if (tile_row < rows && global_k0 < layout.logical_in_features) {
          const int valid_values = static_cast<int>(
              (layout.logical_in_features - global_k0) >= 4 ? 4 : (layout.logical_in_features - global_k0));
          #pragma unroll
          for (int offset = 0; offset < 4; ++offset) {
            if (offset >= valid_values) {
              break;
            }
            const int64_t global_k = global_k0 + offset;
            x_chunk[offset] = BitNetQuantizeStaticInputCodeDevice(
                BitNetInputValueAfterPreScaleDevice(
                    x,
                    pre_scale,
                    tile_row * layout.logical_in_features + global_k,
                    global_k),
                input_scales[tile_row],
                qmax);
          }
        }
        x_tile_packed[tile_row][packed_k] = BitNetPackInt8x4Device(x_chunk);
      }
      for (int index = thread_linear; index < OutputsPerBlock * kPackedTileK; index += kThreads) {
        const int tile_col = index / kPackedTileK;
        const int packed_k = index % kPackedTileK;
        const int tile_k0 = packed_k * 4;
        const int64_t global_col = col_base + tile_col;
        const int64_t global_k0 = k0 + tile_k0;
        int32_t packed_chunk = 0;
        if (global_col < layout.logical_out_features && global_k0 < layout.logical_in_features) {
          const uint8_t packed_value = packed_weight[global_col * packed_cols + (global_k0 >> 2)];
          const int valid_values = static_cast<int>(
              (layout.logical_in_features - global_k0) >= 4 ? 4 : (layout.logical_in_features - global_k0));
          packed_chunk = DecodeSignedTernaryPackInt8x4Device(packed_value, valid_values);
        }
        w_tile_packed[tile_col][packed_k] = packed_chunk;
      }
      __syncthreads();

      if (row_active) {
        #pragma unroll
        for (int col_iter = 0; col_iter < kColsPerLane; ++col_iter) {
          const int tile_col = lane + col_iter * kWarpSize;
          const int64_t global_col = col_base + tile_col;
          if (global_col >= layout.logical_out_features) {
            continue;
          }
          int acc = accum[col_iter];
          #pragma unroll
          for (int packed_k = 0; packed_k < kPackedTileK; ++packed_k) {
            const int64_t global_k0 = k0 + packed_k * 4;
            if (global_k0 >= layout.logical_in_features) {
              break;
            }
            acc = BitNetDotPackedInt8Chunk4Device(
                x_tile_packed[row_slot][packed_k],
                w_tile_packed[tile_col][packed_k],
                acc);
          }
          accum[col_iter] = acc;
        }
      }
      __syncthreads();
    }

    if (row_active) {
      const float input_scale = input_scales[row_slot];
      #pragma unroll
      for (int col_iter = 0; col_iter < kColsPerLane; ++col_iter) {
        const int tile_col = lane + col_iter * kWarpSize;
        const int64_t global_col = col_base + tile_col;
        if (global_col >= layout.logical_out_features) {
          continue;
        }
        float value = static_cast<float>(accum[col_iter]) * input_scale * row_scales[tile_col];
        if (bias != nullptr) {
          value += static_cast<float>(bias[global_col]);
        }
        out[row_slot * layout.logical_out_features + global_col] = CastOutput<scalar_t>(value);
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, int MaxRows>
void launch_decode_row1_static_input(
    const Plan& plan,
    const torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const c10::optional<torch::Tensor>& pre_scale,
    const torch::Tensor& input_scale,
    int64_t act_quant_bits,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& segment_offsets,
    const scalar_t* bias_ptr,
    const LayoutInfo& layout,
    cudaStream_t stream) {
  static_assert(MaxRows == 1, "row1 launch only supports a single decode row");
  const dim3 blocks(static_cast<unsigned int>((layout.logical_out_features + 31) / 32));
  const dim3 threads(32);
  const scalar_t* pre_scale_ptr = nullptr;
  if (pre_scale.has_value() && pre_scale.value().defined()) {
    pre_scale_ptr = pre_scale.value().data_ptr<scalar_t>();
  }
  if (plan.tile_k >= kPrefillSm80TileK) {
    bitnet_linear_decode_row1_static_input_kernel<scalar_t, kPrefillSm80TileK>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            pre_scale_ptr,
            input_scale.data_ptr<float>(),
            input_scale.numel(),
            BitNetQuantMax(act_quant_bits),
            packed_weight.data_ptr<uint8_t>(),
            scale_values.data_ptr<float>(),
            segment_offsets.data_ptr<int32_t>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            packed_weight.size(1));
    return;
  }
  bitnet_linear_decode_row1_static_input_kernel<scalar_t, kPrefillGenericTileK>
      <<<blocks, threads, 0, stream>>>(
          x_2d.data_ptr<scalar_t>(),
          pre_scale_ptr,
          input_scale.data_ptr<float>(),
          input_scale.numel(),
          BitNetQuantMax(act_quant_bits),
          packed_weight.data_ptr<uint8_t>(),
          scale_values.data_ptr<float>(),
          segment_offsets.data_ptr<int32_t>(),
          bias_ptr,
          out_2d.data_ptr<scalar_t>(),
          layout,
          packed_weight.size(1));
}

template <typename scalar_t, int WarpsPerBlock>
void launch_decode_row1_bitplane(
    const torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const scalar_t* bias_ptr,
    const LayoutInfo& layout,
    cudaStream_t stream) {
  static_assert(WarpsPerBlock > 0, "row1 bitplane launch requires at least one warp");
  const dim3 blocks(static_cast<unsigned int>((layout.logical_out_features + WarpsPerBlock - 1) / WarpsPerBlock));
  const dim3 threads(32, WarpsPerBlock);
  bitnet_linear_decode_row1_bitplane_kernel<scalar_t, WarpsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          x_2d.data_ptr<scalar_t>(),
          decode_nz_masks.data_ptr<int32_t>(),
          decode_sign_masks.data_ptr<int32_t>(),
          decode_row_scales.data_ptr<float>(),
          bias_ptr,
          out_2d.data_ptr<scalar_t>(),
          layout,
          decode_nz_masks.size(1),
          decode_nz_masks.size(2));
}

template <typename scalar_t, int WarpsPerBlock>
void launch_decode_row1_bitplane_rms_norm(
    const torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const c10::optional<torch::Tensor>& rms_weight,
    double eps,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const scalar_t* bias_ptr,
    const LayoutInfo& layout,
    cudaStream_t stream) {
  static_assert(WarpsPerBlock > 0, "row1 fused RMSNorm+bitplane launch requires at least one warp");
  const dim3 blocks(static_cast<unsigned int>((layout.logical_out_features + WarpsPerBlock - 1) / WarpsPerBlock));
  const dim3 threads(32, WarpsPerBlock);
  const scalar_t* rms_weight_ptr = nullptr;
  if (rms_weight.has_value() && rms_weight.value().defined()) {
    rms_weight_ptr = rms_weight.value().data_ptr<scalar_t>();
  }
  bitnet_linear_decode_row1_bitplane_rms_norm_kernel<scalar_t, WarpsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          x_2d.data_ptr<scalar_t>(),
          rms_weight_ptr,
          static_cast<float>(eps),
          decode_nz_masks.data_ptr<int32_t>(),
          decode_sign_masks.data_ptr<int32_t>(),
          decode_row_scales.data_ptr<float>(),
          bias_ptr,
          out_2d.data_ptr<scalar_t>(),
          layout,
          decode_nz_masks.size(1),
          decode_nz_masks.size(2));
}

template <typename scalar_t, int WarpsPerBlock>
void launch_decode_row1_bitplane_add_rms_norm(
    const torch::Tensor& combined_2d,
    const torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& update_2d,
    const c10::optional<torch::Tensor>& rms_weight,
    double residual_scale,
    double eps,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const scalar_t* bias_ptr,
    const LayoutInfo& layout,
    cudaStream_t stream) {
  static_assert(WarpsPerBlock > 0, "row1 fused AddRmsNorm+bitplane launch requires at least one warp");
  const dim3 blocks(static_cast<unsigned int>((layout.logical_out_features + WarpsPerBlock - 1) / WarpsPerBlock));
  const dim3 threads(32, WarpsPerBlock);
  const scalar_t* rms_weight_ptr = nullptr;
  if (rms_weight.has_value() && rms_weight.value().defined()) {
    rms_weight_ptr = rms_weight.value().data_ptr<scalar_t>();
  }
  bitnet_linear_decode_row1_bitplane_add_rms_norm_kernel<scalar_t, WarpsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          x_2d.data_ptr<scalar_t>(),
          update_2d.data_ptr<scalar_t>(),
          rms_weight_ptr,
          static_cast<float>(residual_scale),
          static_cast<float>(eps),
          decode_nz_masks.data_ptr<int32_t>(),
          decode_sign_masks.data_ptr<int32_t>(),
          decode_row_scales.data_ptr<float>(),
          bias_ptr,
          combined_2d.data_ptr<scalar_t>(),
          out_2d.data_ptr<scalar_t>(),
          layout,
          decode_nz_masks.size(1),
          decode_nz_masks.size(2));
}

template <typename scalar_t, int MaxRows>
void launch_decode_bucket(
    const Plan& plan,
    const torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& segment_offsets,
    const scalar_t* bias_ptr,
    const LayoutInfo& layout,
    cudaStream_t stream) {
  const int64_t rows = x_2d.size(0);
  const int64_t packed_cols = packed_weight.size(1);
  const int64_t total_tiles = (layout.logical_out_features + plan.outputs_per_block - 1) / plan.outputs_per_block;
  const dim3 threads(32, MaxRows);
  const dim3 blocks(static_cast<unsigned int>(std::max<int64_t>(1, plan.persistent_ctas)));

  if (plan.tile_k >= kPrefillSm80TileK && plan.outputs_per_block >= 128) {
    bitnet_linear_decode_persistent_kernel<scalar_t, MaxRows, kPrefillSm80TileK, 128>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            packed_weight.data_ptr<uint8_t>(),
            scale_values.data_ptr<float>(),
            segment_offsets.data_ptr<int32_t>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            packed_cols,
            total_tiles);
    return;
  }
  if (plan.tile_k >= kPrefillSm80TileK) {
    bitnet_linear_decode_persistent_kernel<scalar_t, MaxRows, kPrefillSm80TileK, 64>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            packed_weight.data_ptr<uint8_t>(),
            scale_values.data_ptr<float>(),
            segment_offsets.data_ptr<int32_t>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            packed_cols,
            total_tiles);
    return;
  }
  if (plan.outputs_per_block >= 128) {
    bitnet_linear_decode_persistent_kernel<scalar_t, MaxRows, kPrefillGenericTileK, 128>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            packed_weight.data_ptr<uint8_t>(),
            scale_values.data_ptr<float>(),
            segment_offsets.data_ptr<int32_t>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            packed_cols,
            total_tiles);
    return;
  }
  bitnet_linear_decode_persistent_kernel<scalar_t, MaxRows, kPrefillGenericTileK, 64>
      <<<blocks, threads, 0, stream>>>(
          x_2d.data_ptr<scalar_t>(),
          packed_weight.data_ptr<uint8_t>(),
          scale_values.data_ptr<float>(),
          segment_offsets.data_ptr<int32_t>(),
          bias_ptr,
          out_2d.data_ptr<scalar_t>(),
          layout,
          rows,
          packed_cols,
          total_tiles);
}

template <typename scalar_t, int MaxRows>
void launch_decode_bucket_compute_packed(
    const Plan& plan,
    const torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const scalar_t* bias_ptr,
    const LayoutInfo& layout,
    cudaStream_t stream) {
  const int64_t rows = x_2d.size(0);
  const int64_t compute_word_cols = compute_packed_words.size(1);
  const int64_t compute_tile_n = compute_packed_words.size(2);
  const int64_t total_tiles = (layout.logical_out_features + plan.outputs_per_block - 1) / plan.outputs_per_block;
  const dim3 threads(32, MaxRows);
  const dim3 blocks(static_cast<unsigned int>(std::max<int64_t>(1, plan.persistent_ctas)));

  if (plan.tile_k >= kPrefillSm80TileK && plan.outputs_per_block >= 128) {
    bitnet_linear_decode_persistent_compute_packed_kernel<scalar_t, MaxRows, kPrefillSm80TileK, 128>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            compute_packed_words.data_ptr<int32_t>(),
            compute_row_scales.data_ptr<float>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            compute_word_cols,
            compute_tile_n,
            total_tiles);
    return;
  }
  if (plan.tile_k >= kPrefillSm80TileK) {
    bitnet_linear_decode_persistent_compute_packed_kernel<scalar_t, MaxRows, kPrefillSm80TileK, 64>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            compute_packed_words.data_ptr<int32_t>(),
            compute_row_scales.data_ptr<float>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            compute_word_cols,
            compute_tile_n,
            total_tiles);
    return;
  }
  if (plan.outputs_per_block >= 128) {
    bitnet_linear_decode_persistent_compute_packed_kernel<scalar_t, MaxRows, kPrefillGenericTileK, 128>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            compute_packed_words.data_ptr<int32_t>(),
            compute_row_scales.data_ptr<float>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            compute_word_cols,
            compute_tile_n,
            total_tiles);
    return;
  }
  bitnet_linear_decode_persistent_compute_packed_kernel<scalar_t, MaxRows, kPrefillGenericTileK, 64>
      <<<blocks, threads, 0, stream>>>(
          x_2d.data_ptr<scalar_t>(),
          compute_packed_words.data_ptr<int32_t>(),
          compute_row_scales.data_ptr<float>(),
          bias_ptr,
          out_2d.data_ptr<scalar_t>(),
          layout,
          rows,
          compute_word_cols,
          compute_tile_n,
          total_tiles);
}

template <typename scalar_t, int MaxRows>
void launch_decode_bucket_static_input(
    const Plan& plan,
    const torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const c10::optional<torch::Tensor>& pre_scale,
    const torch::Tensor& input_scale,
    int64_t act_quant_bits,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& segment_offsets,
    const scalar_t* bias_ptr,
    const LayoutInfo& layout,
    cudaStream_t stream) {
  const int64_t rows = x_2d.size(0);
  const int64_t packed_cols = packed_weight.size(1);
  const int64_t total_tiles = (layout.logical_out_features + plan.outputs_per_block - 1) / plan.outputs_per_block;
  const dim3 threads(32, MaxRows);
  const dim3 blocks(static_cast<unsigned int>(std::max<int64_t>(1, plan.persistent_ctas)));
  const scalar_t* pre_scale_ptr = nullptr;
  if (pre_scale.has_value() && pre_scale.value().defined()) {
    pre_scale_ptr = pre_scale.value().data_ptr<scalar_t>();
  }

  if (plan.tile_k >= kPrefillSm80TileK && plan.outputs_per_block >= 128) {
    bitnet_linear_decode_persistent_static_input_kernel<scalar_t, MaxRows, kPrefillSm80TileK, 128>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            pre_scale_ptr,
            input_scale.data_ptr<float>(),
            input_scale.numel(),
            BitNetQuantMax(act_quant_bits),
            packed_weight.data_ptr<uint8_t>(),
            scale_values.data_ptr<float>(),
            segment_offsets.data_ptr<int32_t>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            packed_cols,
            total_tiles);
    return;
  }
  if (plan.tile_k >= kPrefillSm80TileK) {
    bitnet_linear_decode_persistent_static_input_kernel<scalar_t, MaxRows, kPrefillSm80TileK, 64>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            pre_scale_ptr,
            input_scale.data_ptr<float>(),
            input_scale.numel(),
            BitNetQuantMax(act_quant_bits),
            packed_weight.data_ptr<uint8_t>(),
            scale_values.data_ptr<float>(),
            segment_offsets.data_ptr<int32_t>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            packed_cols,
            total_tiles);
    return;
  }
  if (plan.outputs_per_block >= 128) {
    bitnet_linear_decode_persistent_static_input_kernel<scalar_t, MaxRows, kPrefillGenericTileK, 128>
        <<<blocks, threads, 0, stream>>>(
            x_2d.data_ptr<scalar_t>(),
            pre_scale_ptr,
            input_scale.data_ptr<float>(),
            input_scale.numel(),
            BitNetQuantMax(act_quant_bits),
            packed_weight.data_ptr<uint8_t>(),
            scale_values.data_ptr<float>(),
            segment_offsets.data_ptr<int32_t>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            packed_cols,
            total_tiles);
    return;
  }
  bitnet_linear_decode_persistent_static_input_kernel<scalar_t, MaxRows, kPrefillGenericTileK, 64>
      <<<blocks, threads, 0, stream>>>(
          x_2d.data_ptr<scalar_t>(),
          pre_scale_ptr,
          input_scale.data_ptr<float>(),
          input_scale.numel(),
          BitNetQuantMax(act_quant_bits),
          packed_weight.data_ptr<uint8_t>(),
          scale_values.data_ptr<float>(),
          segment_offsets.data_ptr<int32_t>(),
          bias_ptr,
          out_2d.data_ptr<scalar_t>(),
          layout,
          rows,
          packed_cols,
          total_tiles);
}

}  // namespace

void LaunchBitNetDecodeKernel(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan) {
  c10::cuda::CUDAGuard device_guard(x_2d.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.device().index());
  const auto rows = x_2d.size(0);
  const auto packed_cols = packed_weight.size(1);
  const dim3 scalar_blocks(static_cast<unsigned int>((layout.logical_out_features + kDecodeThreads - 1) / kDecodeThreads));
  const dim3 scalar_threads(kDecodeThreads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_decode_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        if (plan.kind == KernelKind::kDecodePersistent) {
          switch (plan.rows_bucket) {
            case 1:
              launch_decode_bucket<scalar_t, 1>(
                  plan, out_2d, x_2d, packed_weight, scale_values, segment_offsets, bias_ptr, layout, stream.stream());
              break;
            case 2:
              launch_decode_bucket<scalar_t, 2>(
                  plan, out_2d, x_2d, packed_weight, scale_values, segment_offsets, bias_ptr, layout, stream.stream());
              break;
            case 4:
              launch_decode_bucket<scalar_t, 4>(
                  plan, out_2d, x_2d, packed_weight, scale_values, segment_offsets, bias_ptr, layout, stream.stream());
              break;
            default:
              launch_decode_bucket<scalar_t, 8>(
                  plan, out_2d, x_2d, packed_weight, scale_values, segment_offsets, bias_ptr, layout, stream.stream());
              break;
          }
          return;
        }
        bitnet_linear_decode_scalar_kernel<scalar_t><<<scalar_blocks, scalar_threads, 0, stream.stream()>>>(
            x_2d.data_ptr<scalar_t>(),
            packed_weight.data_ptr<uint8_t>(),
            scale_values.data_ptr<float>(),
            segment_offsets.data_ptr<int32_t>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            packed_cols);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void LaunchBitNetDecodeKernelComputePacked(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan) {
  TORCH_CHECK(
      plan.kind == KernelKind::kDecodePersistent,
      "LaunchBitNetDecodeKernelComputePacked only supports decode-persistent plans");
  c10::cuda::CUDAGuard device_guard(x_2d.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.device().index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_decode_compute_packed_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        switch (plan.rows_bucket) {
          case 1:
            launch_decode_bucket_compute_packed<scalar_t, 1>(
                plan,
                out_2d,
                x_2d,
                compute_packed_words,
                compute_row_scales,
                bias_ptr,
                layout,
                stream.stream());
            break;
          case 2:
            launch_decode_bucket_compute_packed<scalar_t, 2>(
                plan,
                out_2d,
                x_2d,
                compute_packed_words,
                compute_row_scales,
                bias_ptr,
                layout,
                stream.stream());
            break;
          case 4:
            launch_decode_bucket_compute_packed<scalar_t, 4>(
                plan,
                out_2d,
                x_2d,
                compute_packed_words,
                compute_row_scales,
                bias_ptr,
                layout,
                stream.stream());
            break;
          default:
            launch_decode_bucket_compute_packed<scalar_t, 8>(
                plan,
                out_2d,
                x_2d,
                compute_packed_words,
                compute_row_scales,
                bias_ptr,
                layout,
                stream.stream());
            break;
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void LaunchBitNetDecodeKernelBitplaneRow1(
    torch::Tensor& out_2d,
    const torch::Tensor& x_2d,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const LayoutInfo& layout,
    const Plan& plan) {
  TORCH_CHECK(
      plan.kind == KernelKind::kDecodePersistent,
      "LaunchBitNetDecodeKernelBitplaneRow1 only supports decode-persistent plans");
  TORCH_CHECK(x_2d.size(0) == 1, "LaunchBitNetDecodeKernelBitplaneRow1 requires a single decode row");
  c10::cuda::CUDAGuard device_guard(x_2d.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.device().index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_decode_bitplane_row1_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        if (layout.logical_out_features >= 256) {
          launch_decode_row1_bitplane<scalar_t, 8>(
              out_2d,
              x_2d,
              decode_nz_masks,
              decode_sign_masks,
              decode_row_scales,
              bias_ptr,
              layout,
              stream.stream());
          return;
        }
        launch_decode_row1_bitplane<scalar_t, 4>(
            out_2d,
            x_2d,
            decode_nz_masks,
            decode_sign_masks,
            decode_row_scales,
            bias_ptr,
            layout,
            stream.stream());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    const Plan& plan) {
  TORCH_CHECK(
      plan.kind == KernelKind::kDecodePersistent,
      "LaunchBitNetDecodeKernelBitplaneRow1RmsNorm only supports decode-persistent plans");
  TORCH_CHECK(x_2d.size(0) == 1, "LaunchBitNetDecodeKernelBitplaneRow1RmsNorm requires a single decode row");
  c10::cuda::CUDAGuard device_guard(x_2d.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.device().index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_decode_bitplane_row1_rms_norm_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        if (layout.logical_out_features >= 256) {
          launch_decode_row1_bitplane_rms_norm<scalar_t, 8>(
              out_2d,
              x_2d,
              rms_weight,
              eps,
              decode_nz_masks,
              decode_sign_masks,
              decode_row_scales,
              bias_ptr,
              layout,
              stream.stream());
          return;
        }
        launch_decode_row1_bitplane_rms_norm<scalar_t, 4>(
            out_2d,
            x_2d,
            rms_weight,
            eps,
            decode_nz_masks,
            decode_sign_masks,
            decode_row_scales,
            bias_ptr,
            layout,
            stream.stream());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    const Plan& plan) {
  TORCH_CHECK(
      plan.kind == KernelKind::kDecodePersistent,
      "LaunchBitNetDecodeKernelBitplaneRow1AddRmsNorm only supports decode-persistent plans");
  TORCH_CHECK(
      x_2d.size(0) == 1 && update_2d.size(0) == 1,
      "LaunchBitNetDecodeKernelBitplaneRow1AddRmsNorm requires a single decode row");
  c10::cuda::CUDAGuard device_guard(x_2d.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.device().index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_decode_bitplane_row1_add_rms_norm_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        if (layout.logical_out_features >= 256) {
          launch_decode_row1_bitplane_add_rms_norm<scalar_t, 8>(
              combined_2d,
              out_2d,
              x_2d,
              update_2d,
              rms_weight,
              residual_scale,
              eps,
              decode_nz_masks,
              decode_sign_masks,
              decode_row_scales,
              bias_ptr,
              layout,
              stream.stream());
          return;
        }
        launch_decode_row1_bitplane_add_rms_norm<scalar_t, 4>(
            combined_2d,
            out_2d,
            x_2d,
            update_2d,
            rms_weight,
            residual_scale,
            eps,
            decode_nz_masks,
            decode_sign_masks,
            decode_row_scales,
            bias_ptr,
            layout,
            stream.stream());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    const Plan& plan) {
  c10::cuda::CUDAGuard device_guard(x_2d.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.device().index());
  const auto rows = x_2d.size(0);
  const auto packed_cols = packed_weight.size(1);
  const dim3 scalar_blocks(static_cast<unsigned int>((layout.logical_out_features + kDecodeThreads - 1) / kDecodeThreads));
  const dim3 scalar_threads(kDecodeThreads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_decode_static_input_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        const scalar_t* pre_scale_ptr = nullptr;
        if (pre_scale.has_value() && pre_scale.value().defined()) {
          pre_scale_ptr = pre_scale.value().data_ptr<scalar_t>();
        }
        if (plan.kind == KernelKind::kDecodePersistent) {
          if (rows == 1) {
            launch_decode_row1_static_input<scalar_t, 1>(
                plan,
                out_2d,
                x_2d,
                pre_scale,
                input_scale,
                act_quant_bits,
                packed_weight,
                scale_values,
                segment_offsets,
                bias_ptr,
                layout,
                stream.stream());
            return;
          }
          switch (plan.rows_bucket) {
            case 1:
              launch_decode_bucket_static_input<scalar_t, 1>(
                  plan,
                  out_2d,
                  x_2d,
                  pre_scale,
                  input_scale,
                  act_quant_bits,
                  packed_weight,
                  scale_values,
                  segment_offsets,
                  bias_ptr,
                  layout,
                  stream.stream());
              break;
            case 2:
              launch_decode_bucket_static_input<scalar_t, 2>(
                  plan,
                  out_2d,
                  x_2d,
                  pre_scale,
                  input_scale,
                  act_quant_bits,
                  packed_weight,
                  scale_values,
                  segment_offsets,
                  bias_ptr,
                  layout,
                  stream.stream());
              break;
            case 4:
              launch_decode_bucket_static_input<scalar_t, 4>(
                  plan,
                  out_2d,
                  x_2d,
                  pre_scale,
                  input_scale,
                  act_quant_bits,
                  packed_weight,
                  scale_values,
                  segment_offsets,
                  bias_ptr,
                  layout,
                  stream.stream());
              break;
            default:
              launch_decode_bucket_static_input<scalar_t, 8>(
                  plan,
                  out_2d,
                  x_2d,
                  pre_scale,
                  input_scale,
                  act_quant_bits,
                  packed_weight,
                  scale_values,
                  segment_offsets,
                  bias_ptr,
                  layout,
                  stream.stream());
              break;
          }
          return;
        }
        bitnet_linear_decode_scalar_static_input_kernel<scalar_t><<<scalar_blocks, scalar_threads, 0, stream.stream()>>>(
            x_2d.data_ptr<scalar_t>(),
            pre_scale_ptr,
            input_scale.data_ptr<float>(),
            input_scale.numel(),
            BitNetQuantMax(act_quant_bits),
            packed_weight.data_ptr<uint8_t>(),
            scale_values.data_ptr<float>(),
            segment_offsets.data_ptr<int32_t>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows,
            packed_cols);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace t10::bitnet
