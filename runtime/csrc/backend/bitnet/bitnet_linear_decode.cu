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

  __shared__ scalar_t x_tile[MaxRows][TileK];
  __shared__ int8_t w_tile[OutputsPerBlock][TileK];
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
      for (int index = thread_linear; index < OutputsPerBlock * kPackedTileK; index += kThreads) {
        const int tile_col = index / kPackedTileK;
        const int packed_k = index % kPackedTileK;
        const int tile_k0 = packed_k * 4;
        const int64_t global_col = col_base + tile_col;
        const int64_t global_k0 = k0 + tile_k0;
        uint8_t packed_value = static_cast<uint8_t>(0x55);
        if (global_col < layout.logical_out_features && global_k0 < layout.logical_in_features) {
          packed_value = packed_weight[global_col * packed_cols + (global_k0 >> 2)];
        }
        #pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
          const int tile_k = tile_k0 + offset;
          if (tile_k < TileK) {
            w_tile[tile_col][tile_k] =
                (global_col < layout.logical_out_features && (global_k0 + offset) < layout.logical_in_features)
                    ? static_cast<int8_t>(DecodeSignedTernaryCode(packed_value, global_k0 + offset))
                    : static_cast<int8_t>(0);
          }
        }
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
          for (int kk = 0; kk < TileK; ++kk) {
            if (k0 + kk >= layout.logical_in_features) {
              break;
            }
            acc += static_cast<float>(x_tile[row_slot][kk]) * static_cast<float>(w_tile[tile_col][kk]);
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
  auto stream = c10::cuda::getDefaultCUDAStream(x_2d.device().index());
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

}  // namespace t10::bitnet
