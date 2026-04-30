#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <mma.h>

#include "cuda_device_arch.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

using namespace nvcuda;

constexpr int kThreads = 256;
constexpr int kThreadCols = 32;
constexpr int kThreadRows = 8;
constexpr int kTileRows = kThreadRows;
constexpr int kGenericTileK = 64;
constexpr int kSm90TileK = 128;
constexpr int kSm90ColsPerThread = 4;
constexpr int kImmaThreads = 32;
constexpr int kImmaTileM = 8;
constexpr int kImmaTileN = 8;
constexpr int kImmaTileK = 32;

bool Int4ImmaEnabled() {
  const char* disable_env = std::getenv("MODEL_STACK_DISABLE_INT4_IMMA_ACT_QUANT");
  if (disable_env != nullptr && disable_env[0] != '\0' && disable_env[0] != '0') {
    return false;
  }
  const char* enable_env = std::getenv("MODEL_STACK_ENABLE_INT4_IMMA_ACT_QUANT");
  if (enable_env != nullptr && enable_env[0] != '\0' && enable_env[0] != '0') {
    return true;
  }
  return true;
}

int64_t Int4ImmaMinOps() {
  const char* env = std::getenv("MODEL_STACK_INT4_IMMA_MIN_OPS");
  if (env == nullptr || env[0] == '\0') {
    return 4'000'000;
  }
  return std::max<int64_t>(1, std::strtoll(env, nullptr, 10));
}

bool IsSupportedInt4LinearDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

template <typename scalar_t>
__global__ void int4_linear_forward_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t packed_cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * out_features;
  if (idx >= total) {
    return;
  }
  const int64_t row = idx / out_features;
  const int64_t col = idx % out_features;

  const scalar_t* x_row = x + (row * in_features);
  const uint8_t* w_row = packed_weight + (col * packed_cols);

  float acc = 0.0f;
  for (int64_t k = 0; k < in_features; ++k) {
    const uint8_t packed = w_row[k >> 1];
    const uint8_t nibble = (k & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    const int q = static_cast<int>(nibble) - 8;
    acc += static_cast<float>(x_row[k]) * static_cast<float>(q);
  }

  acc *= inv_scale[col];
  if (bias != nullptr) {
    acc += static_cast<float>(bias[col]);
  }
  out[row * out_features + col] = static_cast<scalar_t>(acc);
}

__device__ inline int8_t DecodePackedInt4(uint8_t packed, int64_t k) {
  const uint8_t nibble = (k & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
  return static_cast<int8_t>(static_cast<int>(nibble) - 8);
}

__device__ inline uint32_t EncodeSignedInt4Nibble(int value) {
  return static_cast<uint32_t>(value & 0x0F);
}

__device__ inline uint32_t PackSignedInt4Word(const int values[8]) {
  uint32_t packed = 0;
  #pragma unroll
  for (int idx = 0; idx < 8; ++idx) {
    packed |= (EncodeSignedInt4Nibble(values[idx]) << (idx * 4));
  }
  return packed;
}

__device__ inline int QuantizeFloatToSignedInt4(float value, float scale) {
  if (!(scale > 0.0f)) {
    return 0;
  }
  const float scaled = value / scale;
  const float rounded = nearbyintf(scaled);
  const float clamped = fminf(7.0f, fmaxf(-8.0f, rounded));
  return static_cast<int>(clamped);
}

template <typename scalar_t, int TileK, int ColsPerThread>
__global__ __launch_bounds__(kThreads, 2) void int4_linear_forward_tiled_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t packed_cols) {
  constexpr int TileCols = kThreadCols * ColsPerThread;
  __shared__ scalar_t x_tile[kTileRows][TileK];
  __shared__ int8_t w_tile[TileCols][TileK];

  const int local_col = static_cast<int>(threadIdx.x);
  const int local_row = static_cast<int>(threadIdx.y);
  const int thread_linear = local_row * kThreadCols + local_col;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kTileRows + local_row;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TileCols + local_col;

  float accum[ColsPerThread];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    accum[idx] = 0.0f;
  }

  for (int64_t k0 = 0; k0 < in_features; k0 += TileK) {
    for (int index = thread_linear; index < kTileRows * TileK; index += kThreads) {
      const int tile_row = index / TileK;
      const int tile_k = index % TileK;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kTileRows + tile_row;
      const int64_t global_k = k0 + tile_k;
      x_tile[tile_row][tile_k] =
          (global_row < rows && global_k < in_features) ? x[global_row * in_features + global_k] : static_cast<scalar_t>(0);
    }
    for (int index = thread_linear; index < TileCols * TileK; index += kThreads) {
      const int tile_col = index / TileK;
      const int tile_k = index % TileK;
      const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
      const int64_t global_k = k0 + tile_k;
      if (global_col < out_features && global_k < in_features) {
        const uint8_t packed = packed_weight[global_col * packed_cols + (global_k >> 1)];
        w_tile[tile_col][tile_k] = DecodePackedInt4(packed, global_k);
      } else {
        w_tile[tile_col][tile_k] = static_cast<int8_t>(0);
      }
    }
    __syncthreads();

    if (row < rows) {
      #pragma unroll
      for (int idx = 0; idx < ColsPerThread; ++idx) {
        const int tile_col = local_col + idx * kThreadCols;
        const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
        if (global_col >= out_features) {
          continue;
        }
        float acc = accum[idx];
        #pragma unroll
        for (int kk = 0; kk < TileK; ++kk) {
          if (k0 + kk >= in_features) {
            break;
          }
          acc += static_cast<float>(x_tile[local_row][kk]) * static_cast<float>(w_tile[tile_col][kk]);
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
    const int64_t global_col = col0 + idx * kThreadCols;
    if (global_col >= out_features) {
      continue;
    }
    float value = accum[idx] * inv_scale[global_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[row * out_features + global_col] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, int TileK, int ColsPerThread>
__global__ __launch_bounds__(kThreads, 2) void int4_linear_forward_sm90_vectorized_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t packed_cols) {
  constexpr int TileCols = kThreadCols * ColsPerThread;
  constexpr int PackedTileK = (TileK + 1) / 2;
  __shared__ scalar_t x_tile[kTileRows][TileK];
  __shared__ int8_t w_tile[TileCols][TileK];

  const int local_col = static_cast<int>(threadIdx.x);
  const int local_row = static_cast<int>(threadIdx.y);
  const int thread_linear = local_row * kThreadCols + local_col;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kTileRows + local_row;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TileCols + local_col;

  float accum[ColsPerThread];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    accum[idx] = 0.0f;
  }

  for (int64_t k0 = 0; k0 < in_features; k0 += TileK) {
    for (int index = thread_linear; index < kTileRows * TileK; index += kThreads) {
      const int tile_row = index / TileK;
      const int tile_k = index % TileK;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kTileRows + tile_row;
      const int64_t global_k = k0 + tile_k;
      x_tile[tile_row][tile_k] =
          (global_row < rows && global_k < in_features) ? x[global_row * in_features + global_k] : static_cast<scalar_t>(0);
    }
    for (int index = thread_linear; index < TileCols * PackedTileK; index += kThreads) {
      const int tile_col = index / PackedTileK;
      const int packed_k = index % PackedTileK;
      const int tile_k0 = packed_k * 2;
      const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
      const int64_t global_k0 = k0 + tile_k0;
      int8_t low = static_cast<int8_t>(0);
      int8_t high = static_cast<int8_t>(0);
      if (global_col < out_features && global_k0 < in_features) {
        const uint8_t packed = packed_weight[global_col * packed_cols + (global_k0 >> 1)];
        low = static_cast<int8_t>(static_cast<int>(packed & 0x0F) - 8);
        if (global_k0 + 1 < in_features) {
          high = static_cast<int8_t>(static_cast<int>((packed >> 4) & 0x0F) - 8);
        }
      }
      w_tile[tile_col][tile_k0] = low;
      if (tile_k0 + 1 < TileK) {
        w_tile[tile_col][tile_k0 + 1] = high;
      }
    }
    __syncthreads();

    if (row < rows) {
      #pragma unroll
      for (int idx = 0; idx < ColsPerThread; ++idx) {
        const int tile_col = local_col + idx * kThreadCols;
        const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
        if (global_col >= out_features) {
          continue;
        }
        float acc = accum[idx];
        #pragma unroll
        for (int kk = 0; kk < TileK; ++kk) {
          if (k0 + kk >= in_features) {
            break;
          }
          acc += static_cast<float>(x_tile[local_row][kk]) * static_cast<float>(w_tile[tile_col][kk]);
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
    const int64_t global_col = col0 + idx * kThreadCols;
    if (global_col >= out_features) {
      continue;
    }
    float value = accum[idx] * inv_scale[global_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[row * out_features + global_col] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void int4_linear_forward_sm90_imma_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t packed_cols) {
  using s4_t = wmma::experimental::precision::s4;

  const int lane = static_cast<int>(threadIdx.x);
  const int64_t row_start = static_cast<int64_t>(blockIdx.y) * kImmaTileM;
  const int64_t col_start = static_cast<int64_t>(blockIdx.x) * kImmaTileN;

  __shared__ int packed_x_tile[kImmaTileM * (kImmaTileK / 8)];
  __shared__ int packed_w_tile[kImmaTileN * (kImmaTileK / 8)];
  __shared__ int partial_accum[kImmaTileM * kImmaTileN];
  __shared__ float row_chunk_scale[kImmaTileM];
  __shared__ float out_accum[kImmaTileM * kImmaTileN];

  for (int idx = lane; idx < kImmaTileM * kImmaTileN; idx += kImmaThreads) {
    out_accum[idx] = 0.0f;
  }
  __syncthreads();

  for (int64_t k0 = 0; k0 < in_features; k0 += kImmaTileK) {
    if (lane < kImmaTileM) {
      const int64_t global_row = row_start + lane;
      float max_abs = 0.0f;
      if (global_row < rows) {
        for (int kk = 0; kk < kImmaTileK; ++kk) {
          const int64_t global_k = k0 + kk;
          if (global_k >= in_features) {
            break;
          }
          max_abs = fmaxf(max_abs, fabsf(static_cast<float>(x[global_row * in_features + global_k])));
        }
      }
      row_chunk_scale[lane] = max_abs > 0.0f ? (max_abs / 7.0f) : 1.0f;
    }
    __syncthreads();

    if (lane < kImmaThreads) {
      const int local_row = lane / 4;
      const int word = lane % 4;
      const int element_offset = word * 8;
      const int64_t global_row = row_start + local_row;
      int q_values[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (global_row < rows) {
        const float scale = row_chunk_scale[local_row];
        #pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
          const int64_t global_k = k0 + element_offset + idx;
          if (global_k < in_features) {
            q_values[idx] = QuantizeFloatToSignedInt4(
                static_cast<float>(x[global_row * in_features + global_k]),
                scale);
          }
        }
      }
      packed_x_tile[local_row * 4 + word] = static_cast<int>(PackSignedInt4Word(q_values));

      const int local_col = local_row;
      const int64_t global_col = col_start + local_col;
      uint32_t packed_word = 0;
      #pragma unroll
      for (int idx = 0; idx < 8; ++idx) {
        const int64_t global_k = k0 + element_offset + idx;
        if (global_col < out_features && global_k < in_features) {
          const uint8_t packed = packed_weight[global_col * packed_cols + (global_k >> 1)];
          const uint8_t stored_nibble = (global_k & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
          const uint8_t tc_nibble = static_cast<uint8_t>((stored_nibble + 8) & 0x0F);
          packed_word |= (static_cast<uint32_t>(tc_nibble) << (idx * 4));
        }
      }
      packed_w_tile[local_col * 4 + word] = static_cast<int>(packed_word);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kImmaTileM, kImmaTileN, kImmaTileK, s4_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kImmaTileM, kImmaTileN, kImmaTileK, s4_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kImmaTileM, kImmaTileN, kImmaTileK, int> acc_frag;
    wmma::fill_fragment(acc_frag, 0);
    wmma::load_matrix_sync(a_frag, packed_x_tile, kImmaTileK);
    wmma::load_matrix_sync(b_frag, packed_w_tile, kImmaTileK);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    wmma::store_matrix_sync(partial_accum, acc_frag, kImmaTileN, wmma::mem_row_major);
    __syncthreads();

    for (int idx = lane; idx < kImmaTileM * kImmaTileN; idx += kImmaThreads) {
      const int local_row = idx / kImmaTileN;
      out_accum[idx] += static_cast<float>(partial_accum[idx]) * row_chunk_scale[local_row];
    }
    __syncthreads();
  }

  for (int idx = lane; idx < kImmaTileM * kImmaTileN; idx += kImmaThreads) {
    const int local_row = idx / kImmaTileN;
    const int local_col = idx % kImmaTileN;
    const int64_t global_row = row_start + local_row;
    const int64_t global_col = col_start + local_col;
    if (global_row >= rows || global_col >= out_features) {
      continue;
    }
    float value = out_accum[idx] * inv_scale[global_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[global_row * out_features + global_col] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, int TileOut>
__global__ __launch_bounds__(kThreads, 2) void int4_linear_grad_input_tiled_kernel(
    const scalar_t* __restrict__ grad_out,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ inv_scale,
    scalar_t* __restrict__ grad_input,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t packed_cols) {
  __shared__ scalar_t grad_tile[kThreadRows][TileOut];
  __shared__ float scaled_w_tile[kThreadCols][TileOut];

  const int local_col = static_cast<int>(threadIdx.x);
  const int local_row = static_cast<int>(threadIdx.y);
  const int thread_linear = local_row * kThreadCols + local_col;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kThreadRows + local_row;
  const int64_t in_col = static_cast<int64_t>(blockIdx.x) * kThreadCols + local_col;

  float acc = 0.0f;
  for (int64_t out0 = 0; out0 < out_features; out0 += TileOut) {
    for (int index = thread_linear; index < kThreadRows * TileOut; index += kThreads) {
      const int tile_row = index / TileOut;
      const int tile_out = index % TileOut;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kThreadRows + tile_row;
      const int64_t global_out = out0 + tile_out;
      grad_tile[tile_row][tile_out] =
          (global_row < rows && global_out < out_features)
              ? grad_out[global_row * out_features + global_out]
              : static_cast<scalar_t>(0);
    }
    for (int index = thread_linear; index < kThreadCols * TileOut; index += kThreads) {
      const int tile_in = index / TileOut;
      const int tile_out = index % TileOut;
      const int64_t global_in = static_cast<int64_t>(blockIdx.x) * kThreadCols + tile_in;
      const int64_t global_out = out0 + tile_out;
      float value = 0.0f;
      if (global_in < in_features && global_out < out_features) {
        const uint8_t packed = packed_weight[global_out * packed_cols + (global_in >> 1)];
        const int q = static_cast<int>(DecodePackedInt4(packed, global_in));
        value = static_cast<float>(q) * inv_scale[global_out];
      }
      scaled_w_tile[tile_in][tile_out] = value;
    }
    __syncthreads();

    if (row < rows && in_col < in_features) {
      #pragma unroll
      for (int tile_out = 0; tile_out < TileOut; ++tile_out) {
        if (out0 + tile_out >= out_features) {
          break;
        }
        acc += static_cast<float>(grad_tile[local_row][tile_out]) * scaled_w_tile[local_col][tile_out];
      }
    }
    __syncthreads();
  }

  if (row < rows && in_col < in_features) {
    grad_input[row * in_features + in_col] = static_cast<scalar_t>(acc);
  }
}

inline bool UseTiledInt4Kernel(int64_t rows, int64_t out_features, int64_t in_features) {
  return rows > 0 && out_features > 0 && in_features >= 32 && (rows * out_features) >= 256;
}

inline bool UseSm90Int4ImmaKernel(
    const torch::Tensor& reference,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  if (!Int4ImmaEnabled()) {
    return false;
  }
  if (!t10::cuda::DeviceIsSm90OrLater(reference)) {
    return false;
  }
  if (rows < kImmaTileM || out_features < kImmaTileN || in_features < kImmaTileK) {
    return false;
  }
  return (rows * out_features * in_features) >= Int4ImmaMinOps();
}

}  // namespace

bool HasCudaInt4LinearKernel() {
  return true;
}

torch::Tensor CudaInt4LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda(), "CudaInt4LinearForward: x must be a CUDA tensor");
  TORCH_CHECK(packed_weight.is_cuda(), "CudaInt4LinearForward: packed_weight must be a CUDA tensor");
  TORCH_CHECK(inv_scale.is_cuda(), "CudaInt4LinearForward: inv_scale must be a CUDA tensor");
  TORCH_CHECK(IsSupportedInt4LinearDtype(x.scalar_type()), "CudaInt4LinearForward: unsupported x dtype");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8, "CudaInt4LinearForward: packed_weight must be uint8");
  TORCH_CHECK(inv_scale.scalar_type() == torch::kFloat32, "CudaInt4LinearForward: inv_scale must be float32");
  TORCH_CHECK(x.dim() >= 2, "CudaInt4LinearForward: x must have rank >= 2");
  TORCH_CHECK(packed_weight.dim() == 2, "CudaInt4LinearForward: packed_weight must be rank-2");
  TORCH_CHECK(inv_scale.dim() == 1, "CudaInt4LinearForward: inv_scale must be rank-1");
  TORCH_CHECK(inv_scale.size(0) == packed_weight.size(0), "CudaInt4LinearForward: inv_scale size mismatch");

  c10::cuda::CUDAGuard device_guard{x.device()};

  auto x_contig = x.contiguous();
  auto packed_contig = packed_weight.contiguous();
  auto scale_contig = inv_scale.contiguous();
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    auto bias_value = bias.value().to(x_contig.device(), x_contig.scalar_type()).contiguous();
    bias_cast = bias_value;
  }

  const auto in_features = x_contig.size(-1);
  const auto rows = x_contig.numel() / in_features;
  const auto out_features = packed_contig.size(0);
  const auto packed_cols = packed_contig.size(1);
  TORCH_CHECK(
      packed_cols == (in_features + 1) / 2,
      "CudaInt4LinearForward: packed weight column count mismatch");

  auto out_2d = torch::empty({rows, out_features}, x_contig.options());
  if (rows == 0 || out_features == 0) {
    std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
    out_sizes.back() = out_features;
    return out_2d.view(out_sizes);
  }
  const dim3 threads(kThreads);
  const dim3 blocks(static_cast<unsigned int>((rows * out_features + kThreads - 1) / kThreads));
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const bool use_tiled = UseTiledInt4Kernel(rows, out_features, in_features);
  const bool use_imma = UseSm90Int4ImmaKernel(x_contig, rows, out_features, in_features);
  const bool use_sm90 = !use_imma && use_tiled && t10::cuda::DeviceIsSm90OrLater(x_contig) && in_features >= kSm90TileK;
  const dim3 tile_threads(kThreadCols, kThreadRows);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_int4_linear_forward",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias_cast.has_value() && bias_cast.value().defined()) {
          bias_ptr = bias_cast.value().data_ptr<scalar_t>();
        }
        if (use_imma) {
          const dim3 imma_blocks(
              static_cast<unsigned int>((out_features + kImmaTileN - 1) / kImmaTileN),
              static_cast<unsigned int>((rows + kImmaTileM - 1) / kImmaTileM));
          int4_linear_forward_sm90_imma_kernel<scalar_t><<<imma_blocks, kImmaThreads, 0, stream.stream()>>>(
              x_contig.data_ptr<scalar_t>(),
              packed_contig.data_ptr<uint8_t>(),
              scale_contig.data_ptr<float>(),
              bias_ptr,
              out_2d.data_ptr<scalar_t>(),
              rows,
              out_features,
              in_features,
              packed_cols);
        } else if (use_sm90) {
          const dim3 tile_blocks(
              static_cast<unsigned int>(
                  (out_features + (kThreadCols * kSm90ColsPerThread) - 1) /
                  (kThreadCols * kSm90ColsPerThread)),
              static_cast<unsigned int>((rows + kTileRows - 1) / kTileRows));
          int4_linear_forward_sm90_vectorized_kernel<scalar_t, kSm90TileK, kSm90ColsPerThread>
              <<<tile_blocks, tile_threads, 0, stream.stream()>>>(
              x_contig.data_ptr<scalar_t>(),
              packed_contig.data_ptr<uint8_t>(),
              scale_contig.data_ptr<float>(),
              bias_ptr,
              out_2d.data_ptr<scalar_t>(),
              rows,
              out_features,
              in_features,
              packed_cols);
        } else if (use_tiled) {
          const dim3 tile_blocks(
              static_cast<unsigned int>((out_features + kThreadCols - 1) / kThreadCols),
              static_cast<unsigned int>((rows + kTileRows - 1) / kTileRows));
          int4_linear_forward_tiled_kernel<scalar_t, kGenericTileK, 1><<<tile_blocks, tile_threads, 0, stream.stream()>>>(
              x_contig.data_ptr<scalar_t>(),
              packed_contig.data_ptr<uint8_t>(),
              scale_contig.data_ptr<float>(),
              bias_ptr,
              out_2d.data_ptr<scalar_t>(),
              rows,
              out_features,
              in_features,
              packed_cols);
        } else {
          int4_linear_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
              x_contig.data_ptr<scalar_t>(),
              packed_contig.data_ptr<uint8_t>(),
              scale_contig.data_ptr<float>(),
              bias_ptr,
              out_2d.data_ptr<scalar_t>(),
              rows,
              out_features,
              in_features,
              packed_cols);
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = out_features;
  return out_2d.view(out_sizes);
}

torch::Tensor CudaInt4LinearGradInputForward(
    const torch::Tensor& grad_out,
    const torch::Tensor& packed_weight,
    const torch::Tensor& inv_scale,
    int64_t in_features) {
  TORCH_CHECK(grad_out.is_cuda(), "CudaInt4LinearGradInputForward: grad_out must be a CUDA tensor");
  TORCH_CHECK(packed_weight.is_cuda(), "CudaInt4LinearGradInputForward: packed_weight must be a CUDA tensor");
  TORCH_CHECK(inv_scale.is_cuda(), "CudaInt4LinearGradInputForward: inv_scale must be a CUDA tensor");
  TORCH_CHECK(IsSupportedInt4LinearDtype(grad_out.scalar_type()), "CudaInt4LinearGradInputForward: unsupported grad_out dtype");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8, "CudaInt4LinearGradInputForward: packed_weight must be uint8");
  TORCH_CHECK(inv_scale.scalar_type() == torch::kFloat32, "CudaInt4LinearGradInputForward: inv_scale must be float32");
  TORCH_CHECK(grad_out.dim() >= 2, "CudaInt4LinearGradInputForward: grad_out must have rank >= 2");
  TORCH_CHECK(packed_weight.dim() == 2, "CudaInt4LinearGradInputForward: packed_weight must be rank-2");
  TORCH_CHECK(inv_scale.dim() == 1, "CudaInt4LinearGradInputForward: inv_scale must be rank-1");
  TORCH_CHECK(in_features > 0, "CudaInt4LinearGradInputForward: in_features must be positive");

  c10::cuda::CUDAGuard device_guard{grad_out.device()};
  auto grad_contig = grad_out.contiguous();
  auto packed_contig = packed_weight.contiguous();
  auto scale_contig = inv_scale.contiguous();
  const auto out_features = grad_contig.size(-1);
  const auto rows = grad_contig.numel() / out_features;
  const auto packed_cols = packed_contig.size(1);
  TORCH_CHECK(packed_contig.size(0) == out_features, "CudaInt4LinearGradInputForward: output feature mismatch");
  TORCH_CHECK(scale_contig.size(0) == out_features, "CudaInt4LinearGradInputForward: inv_scale size mismatch");
  TORCH_CHECK(packed_cols == (in_features + 1) / 2, "CudaInt4LinearGradInputForward: packed weight column count mismatch");

  auto grad_input_2d = torch::empty({rows, in_features}, grad_contig.options());
  if (rows == 0 || out_features == 0) {
    std::vector<int64_t> out_sizes(grad_contig.sizes().begin(), grad_contig.sizes().end());
    out_sizes.back() = in_features;
    return grad_input_2d.view(out_sizes);
  }

  constexpr int kGradInputTileOut = 64;
  const dim3 threads(kThreadCols, kThreadRows);
  const dim3 blocks(
      static_cast<unsigned int>((in_features + kThreadCols - 1) / kThreadCols),
      static_cast<unsigned int>((rows + kThreadRows - 1) / kThreadRows));
  auto stream = c10::cuda::getCurrentCUDAStream(grad_out.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_contig.scalar_type(),
      "model_stack_cuda_int4_linear_grad_input",
      [&] {
        int4_linear_grad_input_tiled_kernel<scalar_t, kGradInputTileOut><<<blocks, threads, 0, stream.stream()>>>(
            grad_contig.data_ptr<scalar_t>(),
            packed_contig.data_ptr<uint8_t>(),
            scale_contig.data_ptr<float>(),
            grad_input_2d.data_ptr<scalar_t>(),
            rows,
            out_features,
            in_features,
            packed_cols);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(grad_contig.sizes().begin(), grad_contig.sizes().end());
  out_sizes.back() = in_features;
  return grad_input_2d.view(out_sizes);
}
