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

template <typename scalar_t, int TileK, int ColsPerThread>
__global__ __launch_bounds__(kPrefillThreadsX * kPrefillThreadsY, 2) void bitnet_linear_prefill_tiled_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ scale_values,
    const int32_t* __restrict__ segment_offsets,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t rows,
    int64_t packed_cols) {
  constexpr int TileCols = kPrefillThreadsX * ColsPerThread;
  constexpr int Threads = kPrefillThreadsX * kPrefillThreadsY;
  constexpr int PackedTileK = (TileK + 3) / 4;
  __shared__ scalar_t x_tile[kPrefillTileRows][TileK];
  __shared__ uint8_t w_tile_packed[TileCols][PackedTileK];
  __shared__ float row_scales[TileCols];

  const int local_col = static_cast<int>(threadIdx.x);
  const int local_row = static_cast<int>(threadIdx.y);
  const int thread_linear = local_row * kPrefillThreadsX + local_col;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + local_row;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TileCols + local_col;

  float accum[ColsPerThread];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    accum[idx] = 0.0f;
  }

  for (int idx = thread_linear; idx < TileCols; idx += Threads) {
    const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + idx;
    row_scales[idx] =
        global_col < layout.logical_out_features
            ? ResolveRowScaleDevice(global_col, scale_values, segment_offsets, layout)
            : 0.0f;
  }
  __syncthreads();

  for (int64_t k0 = 0; k0 < layout.logical_in_features; k0 += TileK) {
    for (int index = thread_linear; index < kPrefillTileRows * TileK; index += Threads) {
      const int tile_row = index / TileK;
      const int tile_k = index % TileK;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + tile_row;
      const int64_t global_k = k0 + tile_k;
      x_tile[tile_row][tile_k] =
          (global_row < rows && global_k < layout.logical_in_features)
              ? x[global_row * layout.logical_in_features + global_k]
              : static_cast<scalar_t>(0);
    }
    for (int index = thread_linear; index < TileCols * PackedTileK; index += Threads) {
      const int tile_col = index / PackedTileK;
      const int packed_k = index % PackedTileK;
      const int tile_k0 = packed_k * 4;
      const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
      const int64_t global_k0 = k0 + tile_k0;
      uint8_t packed_value = static_cast<uint8_t>(0x55);
      if (global_col < layout.logical_out_features && global_k0 < layout.logical_in_features) {
        packed_value = packed_weight[global_col * packed_cols + (global_k0 >> 2)];
      }
      w_tile_packed[tile_col][packed_k] = packed_value;
    }
    __syncthreads();

    if (row < rows) {
      #pragma unroll
      for (int idx = 0; idx < ColsPerThread; ++idx) {
        const int tile_col = local_col + idx * kPrefillThreadsX;
        const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
        if (global_col >= layout.logical_out_features) {
          continue;
        }
        float acc = accum[idx];
        #pragma unroll
        for (int packed_k = 0; packed_k < PackedTileK; ++packed_k) {
          const int tile_k0 = packed_k * 4;
          const int64_t global_k0 = k0 + tile_k0;
          if (global_k0 >= layout.logical_in_features) {
            break;
          }
          const int valid_values = static_cast<int>(
              (layout.logical_in_features - global_k0) >= 4 ? 4 : (layout.logical_in_features - global_k0));
          acc = BitNetDotFloatPackedTernaryChunk4Device(
              &x_tile[local_row][tile_k0],
              w_tile_packed[tile_col][packed_k],
              valid_values,
              acc);
        }
        accum[idx] = acc;
      }
    }
    __syncthreads();
  }

  if (row >= rows) {
    return;
  }
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t global_col = col0 + idx * kPrefillThreadsX;
    if (global_col >= layout.logical_out_features) {
      continue;
    }
    const int tile_col = local_col + idx * kPrefillThreadsX;
    float value = accum[idx] * row_scales[tile_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[row * layout.logical_out_features + global_col] = CastOutput<scalar_t>(value);
  }
}

template <typename scalar_t, int TileK, int ColsPerThread>
__global__ __launch_bounds__(kPrefillThreadsX * kPrefillThreadsY, 2) void bitnet_linear_prefill_tiled_static_input_kernel(
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
  constexpr int TileCols = kPrefillThreadsX * ColsPerThread;
  constexpr int Threads = kPrefillThreadsX * kPrefillThreadsY;
  constexpr int PackedTileK = (TileK + 3) / 4;
  __shared__ __align__(16) int32_t x_tile_packed[kPrefillTileRows][PackedTileK];
  __shared__ __align__(16) int32_t w_tile_packed[TileCols][PackedTileK];
  __shared__ float row_scales[TileCols];
  __shared__ float input_scales[kPrefillTileRows];

  const int local_col = static_cast<int>(threadIdx.x);
  const int local_row = static_cast<int>(threadIdx.y);
  const int thread_linear = local_row * kPrefillThreadsX + local_col;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + local_row;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TileCols + local_col;

  int accum[ColsPerThread];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    accum[idx] = 0;
  }

  for (int idx = thread_linear; idx < TileCols; idx += Threads) {
    const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + idx;
    row_scales[idx] =
        global_col < layout.logical_out_features
            ? ResolveRowScaleDevice(global_col, scale_values, segment_offsets, layout)
            : 0.0f;
  }
  for (int idx = thread_linear; idx < kPrefillTileRows; idx += Threads) {
    const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + idx;
    input_scales[idx] = global_row < rows ? ResolveInputScaleDevice(global_row, input_scale, input_scale_rows) : 1.0f;
  }
  __syncthreads();

  for (int64_t k0 = 0; k0 < layout.logical_in_features; k0 += TileK) {
    for (int index = thread_linear; index < kPrefillTileRows * PackedTileK; index += Threads) {
      const int tile_row = index / PackedTileK;
      const int packed_k = index % PackedTileK;
      const int tile_k0 = packed_k * 4;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + tile_row;
      const int64_t global_k0 = k0 + tile_k0;
      int8_t x_chunk[4] = {0, 0, 0, 0};
      if (global_row < rows && global_k0 < layout.logical_in_features) {
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
                  global_row * layout.logical_in_features + global_k,
                  global_k),
              input_scales[tile_row],
              qmax);
        }
      }
      x_tile_packed[tile_row][packed_k] = BitNetPackInt8x4Device(x_chunk);
    }
    for (int index = thread_linear; index < TileCols * PackedTileK; index += Threads) {
      const int tile_col = index / PackedTileK;
      const int packed_k = index % PackedTileK;
      const int tile_k0 = packed_k * 4;
      const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
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

    if (row < rows) {
      #pragma unroll
      for (int idx = 0; idx < ColsPerThread; ++idx) {
        const int tile_col = local_col + idx * kPrefillThreadsX;
        const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
        if (global_col >= layout.logical_out_features) {
          continue;
        }
        int acc = accum[idx];
        #pragma unroll
        for (int packed_k = 0; packed_k < PackedTileK; ++packed_k) {
          const int64_t global_k0 = k0 + packed_k * 4;
          if (global_k0 >= layout.logical_in_features) {
            break;
          }
          acc = BitNetDotPackedInt8Chunk4Device(
              x_tile_packed[local_row][packed_k],
              w_tile_packed[tile_col][packed_k],
              acc);
        }
        accum[idx] = acc;
      }
    }
    __syncthreads();
  }

  if (row >= rows) {
    return;
  }
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t global_col = col0 + idx * kPrefillThreadsX;
    if (global_col >= layout.logical_out_features) {
      continue;
    }
    const int tile_col = local_col + idx * kPrefillThreadsX;
    float value = static_cast<float>(accum[idx]) * input_scales[local_row] * row_scales[tile_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[row * layout.logical_out_features + global_col] = CastOutput<scalar_t>(value);
  }
}

template <typename scalar_t, int TileK, int ColsPerThread>
__global__ __launch_bounds__(kPrefillThreadsX * kPrefillThreadsY, 2) void bitnet_linear_prefill_splitk_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ scale_values,
    const int32_t* __restrict__ segment_offsets,
    float* __restrict__ workspace,
    LayoutInfo layout,
    int64_t rows,
    int64_t packed_cols) {
  constexpr int TileCols = kPrefillThreadsX * ColsPerThread;
  constexpr int Threads = kPrefillThreadsX * kPrefillThreadsY;
  constexpr int PackedTileK = (TileK + 3) / 4;
  __shared__ scalar_t x_tile[kPrefillTileRows][TileK];
  __shared__ uint8_t w_tile_packed[TileCols][PackedTileK];
  __shared__ float row_scales[TileCols];

  const int local_col = static_cast<int>(threadIdx.x);
  const int local_row = static_cast<int>(threadIdx.y);
  const int thread_linear = local_row * kPrefillThreadsX + local_col;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + local_row;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TileCols + local_col;

  const int split_k_slices = gridDim.z;
  const int64_t slice_span = ((layout.logical_in_features + split_k_slices - 1) / split_k_slices + TileK - 1) / TileK * TileK;
  const int64_t slice_begin_raw = static_cast<int64_t>(blockIdx.z) * slice_span;
  const int64_t slice_begin = slice_begin_raw < layout.logical_in_features ? slice_begin_raw : layout.logical_in_features;
  const int64_t slice_end_raw = slice_begin + slice_span;
  const int64_t slice_end = slice_end_raw < layout.logical_in_features ? slice_end_raw : layout.logical_in_features;

  float accum[ColsPerThread];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    accum[idx] = 0.0f;
  }

  for (int idx = thread_linear; idx < TileCols; idx += Threads) {
    const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + idx;
    row_scales[idx] =
        global_col < layout.logical_out_features
            ? ResolveRowScaleDevice(global_col, scale_values, segment_offsets, layout)
            : 0.0f;
  }
  __syncthreads();

  for (int64_t k0 = slice_begin; k0 < slice_end; k0 += TileK) {
    for (int index = thread_linear; index < kPrefillTileRows * TileK; index += Threads) {
      const int tile_row = index / TileK;
      const int tile_k = index % TileK;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + tile_row;
      const int64_t global_k = k0 + tile_k;
      x_tile[tile_row][tile_k] =
          (global_row < rows && global_k < slice_end && global_k < layout.logical_in_features)
              ? x[global_row * layout.logical_in_features + global_k]
              : static_cast<scalar_t>(0);
    }
    for (int index = thread_linear; index < TileCols * PackedTileK; index += Threads) {
      const int tile_col = index / PackedTileK;
      const int packed_k = index % PackedTileK;
      const int tile_k0 = packed_k * 4;
      const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
      const int64_t global_k0 = k0 + tile_k0;
      uint8_t packed_value = static_cast<uint8_t>(0x55);
      if (global_col < layout.logical_out_features && global_k0 < slice_end && global_k0 < layout.logical_in_features) {
        packed_value = packed_weight[global_col * packed_cols + (global_k0 >> 2)];
      }
      w_tile_packed[tile_col][packed_k] = packed_value;
    }
    __syncthreads();

    if (row < rows) {
      #pragma unroll
      for (int idx = 0; idx < ColsPerThread; ++idx) {
        const int tile_col = local_col + idx * kPrefillThreadsX;
        const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
        if (global_col >= layout.logical_out_features) {
          continue;
        }
        float acc = accum[idx];
        #pragma unroll
        for (int packed_k = 0; packed_k < PackedTileK; ++packed_k) {
          const int tile_k0 = packed_k * 4;
          const int64_t global_k0 = k0 + tile_k0;
          if (global_k0 >= slice_end || global_k0 >= layout.logical_in_features) {
            break;
          }
          const int64_t slice_limit = slice_end < layout.logical_in_features ? slice_end : layout.logical_in_features;
          const int valid_values = static_cast<int>(
              (slice_limit - global_k0) >= 4 ? 4 : (slice_limit - global_k0));
          acc = BitNetDotFloatPackedTernaryChunk4Device(
              &x_tile[local_row][tile_k0],
              w_tile_packed[tile_col][packed_k],
              valid_values,
              acc);
        }
        accum[idx] = acc;
      }
    }
    __syncthreads();
  }

  if (row >= rows) {
    return;
  }
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t global_col = col0 + idx * kPrefillThreadsX;
    if (global_col >= layout.logical_out_features) {
      continue;
    }
    const int tile_col = local_col + idx * kPrefillThreadsX;
    atomicAdd(
        workspace + (row * layout.logical_out_features + global_col),
        accum[idx] * row_scales[tile_col]);
  }
}

template <typename scalar_t, int TileK, int ColsPerThread>
__global__ __launch_bounds__(kPrefillThreadsX * kPrefillThreadsY, 2) void bitnet_linear_prefill_splitk_static_input_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ input_scale,
    int64_t input_scale_rows,
    int qmax,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ scale_values,
    const int32_t* __restrict__ segment_offsets,
    float* __restrict__ workspace,
    LayoutInfo layout,
    int64_t rows,
    int64_t packed_cols) {
  constexpr int TileCols = kPrefillThreadsX * ColsPerThread;
  constexpr int Threads = kPrefillThreadsX * kPrefillThreadsY;
  constexpr int PackedTileK = (TileK + 3) / 4;
  __shared__ __align__(16) int32_t x_tile_packed[kPrefillTileRows][PackedTileK];
  __shared__ __align__(16) int32_t w_tile_packed[TileCols][PackedTileK];
  __shared__ float input_scales[kPrefillTileRows];
  __shared__ float row_scales[TileCols];

  const int local_col = static_cast<int>(threadIdx.x);
  const int local_row = static_cast<int>(threadIdx.y);
  const int thread_linear = local_row * kPrefillThreadsX + local_col;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + local_row;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TileCols + local_col;

  const int split_k_slices = gridDim.z;
  const int64_t slice_span = ((layout.logical_in_features + split_k_slices - 1) / split_k_slices + TileK - 1) / TileK * TileK;
  const int64_t slice_begin_raw = static_cast<int64_t>(blockIdx.z) * slice_span;
  const int64_t slice_begin = slice_begin_raw < layout.logical_in_features ? slice_begin_raw : layout.logical_in_features;
  const int64_t slice_end_raw = slice_begin + slice_span;
  const int64_t slice_end = slice_end_raw < layout.logical_in_features ? slice_end_raw : layout.logical_in_features;

  int accum[ColsPerThread];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    accum[idx] = 0;
  }

  for (int idx = thread_linear; idx < TileCols; idx += Threads) {
    const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + idx;
    row_scales[idx] =
        global_col < layout.logical_out_features
            ? ResolveRowScaleDevice(global_col, scale_values, segment_offsets, layout)
            : 0.0f;
  }
  for (int idx = thread_linear; idx < kPrefillTileRows; idx += Threads) {
    const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + idx;
    input_scales[idx] = global_row < rows ? ResolveInputScaleDevice(global_row, input_scale, input_scale_rows) : 1.0f;
  }
  __syncthreads();

  for (int64_t k0 = slice_begin; k0 < slice_end; k0 += TileK) {
    for (int index = thread_linear; index < kPrefillTileRows * PackedTileK; index += Threads) {
      const int tile_row = index / PackedTileK;
      const int packed_k = index % PackedTileK;
      const int tile_k0 = packed_k * 4;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kPrefillTileRows + tile_row;
      const int64_t global_k0 = k0 + tile_k0;
      int8_t x_chunk[4] = {0, 0, 0, 0};
      if (global_row < rows && global_k0 < slice_end && global_k0 < layout.logical_in_features) {
        const int64_t slice_limit = slice_end < layout.logical_in_features ? slice_end : layout.logical_in_features;
        const int64_t remaining = slice_limit - global_k0;
        const int valid_values = static_cast<int>(remaining >= 4 ? 4 : remaining);
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
                  global_row * layout.logical_in_features + global_k,
                  global_k),
              input_scales[tile_row],
              qmax);
        }
      }
      x_tile_packed[tile_row][packed_k] = BitNetPackInt8x4Device(x_chunk);
    }
    for (int index = thread_linear; index < TileCols * PackedTileK; index += Threads) {
      const int tile_col = index / PackedTileK;
      const int packed_k = index % PackedTileK;
      const int tile_k0 = packed_k * 4;
      const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
      const int64_t global_k0 = k0 + tile_k0;
      int32_t packed_chunk = 0;
      if (global_col < layout.logical_out_features && global_k0 < slice_end && global_k0 < layout.logical_in_features) {
        const uint8_t packed_value = packed_weight[global_col * packed_cols + (global_k0 >> 2)];
        const int64_t slice_limit = slice_end < layout.logical_in_features ? slice_end : layout.logical_in_features;
        const int64_t remaining = slice_limit - global_k0;
        const int valid_values = static_cast<int>(remaining >= 4 ? 4 : remaining);
        packed_chunk = DecodeSignedTernaryPackInt8x4Device(packed_value, valid_values);
      }
      w_tile_packed[tile_col][packed_k] = packed_chunk;
    }
    __syncthreads();

    if (row < rows) {
      #pragma unroll
      for (int idx = 0; idx < ColsPerThread; ++idx) {
        const int tile_col = local_col + idx * kPrefillThreadsX;
        const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
        if (global_col >= layout.logical_out_features) {
          continue;
        }
        int acc = accum[idx];
        #pragma unroll
        for (int packed_k = 0; packed_k < PackedTileK; ++packed_k) {
          const int64_t global_k0 = k0 + packed_k * 4;
          if (global_k0 >= slice_end || global_k0 >= layout.logical_in_features) {
            break;
          }
          acc = BitNetDotPackedInt8Chunk4Device(
              x_tile_packed[local_row][packed_k],
              w_tile_packed[tile_col][packed_k],
              acc);
        }
        accum[idx] = acc;
      }
    }
    __syncthreads();
  }

  if (row >= rows) {
    return;
  }
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t global_col = col0 + idx * kPrefillThreadsX;
    if (global_col >= layout.logical_out_features) {
      continue;
    }
    const int tile_col = local_col + idx * kPrefillThreadsX;
    atomicAdd(
        workspace + (row * layout.logical_out_features + global_col),
        static_cast<float>(accum[idx]) * input_scales[local_row] * row_scales[tile_col]);
  }
}

template <typename scalar_t>
__global__ void bitnet_linear_splitk_finalize_kernel(
    const float* __restrict__ workspace,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    LayoutInfo layout,
    int64_t rows) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * layout.logical_out_features;
  if (idx >= total) {
    return;
  }
  const int64_t col = idx % layout.logical_out_features;
  float value = workspace[idx];
  if (bias != nullptr) {
    value += static_cast<float>(bias[col]);
  }
  out[idx] = CastOutput<scalar_t>(value);
}

inline bool UseTiledPrefillKernel(int64_t rows, int64_t out_features, int64_t in_features) {
  return rows > 1 && out_features > 0 && in_features >= 32 && (rows * out_features) >= 256;
}

}  // namespace

void LaunchBitNetPrefillKernel(
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
  const auto out_features = layout.logical_out_features;
  const auto in_features = layout.logical_in_features;

  if (!UseTiledPrefillKernel(rows, out_features, in_features)) {
    LaunchBitNetDecodeKernel(out_2d, x_2d, packed_weight, scale_values, segment_offsets, bias, layout, plan);
    return;
  }

  const dim3 threads(kPrefillThreadsX, kPrefillThreadsY);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_prefill_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        if (plan.tile_k >= kPrefillSm80TileK && plan.cols_per_thread >= kPrefillSm80ColsPerThread) {
          const dim3 blocks(
              static_cast<unsigned int>((out_features + (kPrefillThreadsX * kPrefillSm80ColsPerThread) - 1) /
                                        (kPrefillThreadsX * kPrefillSm80ColsPerThread)),
              static_cast<unsigned int>((rows + kPrefillTileRows - 1) / kPrefillTileRows));
          bitnet_linear_prefill_tiled_kernel<scalar_t, kPrefillSm80TileK, kPrefillSm80ColsPerThread>
              <<<blocks, threads, 0, stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  packed_weight.data_ptr<uint8_t>(),
                  scale_values.data_ptr<float>(),
                  segment_offsets.data_ptr<int32_t>(),
                  bias_ptr,
                  out_2d.data_ptr<scalar_t>(),
                  layout,
                  rows,
                  packed_weight.size(1));
        } else {
          const dim3 blocks(
              static_cast<unsigned int>((out_features + kPrefillThreadsX - 1) / kPrefillThreadsX),
              static_cast<unsigned int>((rows + kPrefillTileRows - 1) / kPrefillTileRows));
          bitnet_linear_prefill_tiled_kernel<scalar_t, kPrefillGenericTileK, 1>
              <<<blocks, threads, 0, stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  packed_weight.data_ptr<uint8_t>(),
                  scale_values.data_ptr<float>(),
                  segment_offsets.data_ptr<int32_t>(),
                  bias_ptr,
                  out_2d.data_ptr<scalar_t>(),
                  layout,
                  rows,
                  packed_weight.size(1));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    const Plan& plan) {
  c10::cuda::CUDAGuard device_guard(x_2d.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.device().index());
  const auto rows = x_2d.size(0);
  const auto out_features = layout.logical_out_features;
  const auto in_features = layout.logical_in_features;

  if (!UseTiledPrefillKernel(rows, out_features, in_features)) {
    LaunchBitNetDecodeKernelStaticInput(
        out_2d,
        x_2d,
        pre_scale,
        input_scale,
        act_quant_bits,
        packed_weight,
        scale_values,
        segment_offsets,
        bias,
        layout,
        plan);
    return;
  }

  const dim3 threads(kPrefillThreadsX, kPrefillThreadsY);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_prefill_static_input_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        const scalar_t* pre_scale_ptr = nullptr;
        if (pre_scale.has_value() && pre_scale.value().defined()) {
          pre_scale_ptr = pre_scale.value().data_ptr<scalar_t>();
        }
        if (plan.tile_k >= kPrefillSm80TileK && plan.cols_per_thread >= kPrefillSm80ColsPerThread) {
          const dim3 blocks(
              static_cast<unsigned int>((out_features + (kPrefillThreadsX * kPrefillSm80ColsPerThread) - 1) /
                                        (kPrefillThreadsX * kPrefillSm80ColsPerThread)),
              static_cast<unsigned int>((rows + kPrefillTileRows - 1) / kPrefillTileRows));
          bitnet_linear_prefill_tiled_static_input_kernel<scalar_t, kPrefillSm80TileK, kPrefillSm80ColsPerThread>
              <<<blocks, threads, 0, stream.stream()>>>(
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
                  packed_weight.size(1));
        } else {
          const dim3 blocks(
              static_cast<unsigned int>((out_features + kPrefillThreadsX - 1) / kPrefillThreadsX),
              static_cast<unsigned int>((rows + kPrefillTileRows - 1) / kPrefillTileRows));
          bitnet_linear_prefill_tiled_static_input_kernel<scalar_t, kPrefillGenericTileK, 1>
              <<<blocks, threads, 0, stream.stream()>>>(
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
                  packed_weight.size(1));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void LaunchBitNetPrefillSplitKKernel(
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
  const auto out_features = layout.logical_out_features;
  const auto total = rows * out_features;
  auto workspace = torch::zeros({rows, out_features}, torch::TensorOptions().device(x_2d.device()).dtype(torch::kFloat32));

  const dim3 threads(kPrefillThreadsX, kPrefillThreadsY);
  const dim3 blocks(
      static_cast<unsigned int>((out_features + plan.outputs_per_block - 1) / plan.outputs_per_block),
      static_cast<unsigned int>((rows + kPrefillTileRows - 1) / kPrefillTileRows),
      static_cast<unsigned int>(std::max(1, plan.split_k_slices)));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_prefill_splitk_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        if (plan.tile_k >= kPrefillSm80TileK && plan.cols_per_thread >= kPrefillSm80ColsPerThread) {
          bitnet_linear_prefill_splitk_kernel<scalar_t, kPrefillSm80TileK, kPrefillSm80ColsPerThread>
              <<<blocks, threads, 0, stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  packed_weight.data_ptr<uint8_t>(),
                  scale_values.data_ptr<float>(),
                  segment_offsets.data_ptr<int32_t>(),
                  workspace.data_ptr<float>(),
                  layout,
                  rows,
                  packed_weight.size(1));
        } else {
          bitnet_linear_prefill_splitk_kernel<scalar_t, kPrefillGenericTileK, 1>
              <<<blocks, threads, 0, stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  packed_weight.data_ptr<uint8_t>(),
                  scale_values.data_ptr<float>(),
                  segment_offsets.data_ptr<int32_t>(),
                  workspace.data_ptr<float>(),
                  layout,
                  rows,
                  packed_weight.size(1));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const dim3 finalize_blocks(static_cast<unsigned int>((total + kDecodeThreads - 1) / kDecodeThreads));
        const dim3 finalize_threads(kDecodeThreads);
        bitnet_linear_splitk_finalize_kernel<scalar_t><<<finalize_blocks, finalize_threads, 0, stream.stream()>>>(
            workspace.data_ptr<float>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    const Plan& plan) {
  c10::cuda::CUDAGuard device_guard(x_2d.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.device().index());
  const auto rows = x_2d.size(0);
  const auto out_features = layout.logical_out_features;
  const auto total = rows * out_features;
  auto workspace = torch::zeros({rows, out_features}, torch::TensorOptions().device(x_2d.device()).dtype(torch::kFloat32));

  const dim3 threads(kPrefillThreadsX, kPrefillThreadsY);
  const dim3 blocks(
      static_cast<unsigned int>((out_features + plan.outputs_per_block - 1) / plan.outputs_per_block),
      static_cast<unsigned int>((rows + kPrefillTileRows - 1) / kPrefillTileRows),
      static_cast<unsigned int>(std::max(1, plan.split_k_slices)));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_2d.scalar_type(),
      "bitnet_linear_prefill_splitk_static_input_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value() && bias.value().defined()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        const scalar_t* pre_scale_ptr = nullptr;
        if (pre_scale.has_value() && pre_scale.value().defined()) {
          pre_scale_ptr = pre_scale.value().data_ptr<scalar_t>();
        }
        if (plan.tile_k >= kPrefillSm80TileK && plan.cols_per_thread >= kPrefillSm80ColsPerThread) {
          bitnet_linear_prefill_splitk_static_input_kernel<scalar_t, kPrefillSm80TileK, kPrefillSm80ColsPerThread>
              <<<blocks, threads, 0, stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  pre_scale_ptr,
                  input_scale.data_ptr<float>(),
                  input_scale.numel(),
                  BitNetQuantMax(act_quant_bits),
                  packed_weight.data_ptr<uint8_t>(),
                  scale_values.data_ptr<float>(),
                  segment_offsets.data_ptr<int32_t>(),
                  workspace.data_ptr<float>(),
                  layout,
                  rows,
                  packed_weight.size(1));
        } else {
          bitnet_linear_prefill_splitk_static_input_kernel<scalar_t, kPrefillGenericTileK, 1>
              <<<blocks, threads, 0, stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  pre_scale_ptr,
                  input_scale.data_ptr<float>(),
                  input_scale.numel(),
                  BitNetQuantMax(act_quant_bits),
                  packed_weight.data_ptr<uint8_t>(),
                  scale_values.data_ptr<float>(),
                  segment_offsets.data_ptr<int32_t>(),
                  workspace.data_ptr<float>(),
                  layout,
                  rows,
                  packed_weight.size(1));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const dim3 finalize_blocks(static_cast<unsigned int>((total + kDecodeThreads - 1) / kDecodeThreads));
        const dim3 finalize_threads(kDecodeThreads);
        bitnet_linear_splitk_finalize_kernel<scalar_t><<<finalize_blocks, finalize_threads, 0, stream.stream()>>>(
            workspace.data_ptr<float>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            layout,
            rows);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace t10::bitnet
