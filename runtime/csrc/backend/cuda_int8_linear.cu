#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <mma.h>
#include <sm_61_intrinsics.h>

#include "cuda_hopper_advanced.cuh"
#include "cuda_device_arch.cuh"

#include <cstdlib>
#include <cstdint>
#include <vector>

torch::Tensor CublasLtInt8LinearForward(
    const torch::Tensor& qx,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);

torch::Tensor CutlassInt8LinearFusedForward(
    const torch::Tensor& qx,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);

namespace {

using namespace nvcuda;

constexpr int kThreads = 256;
constexpr int kThreadCols = 32;
constexpr int kThreadRows = 8;
constexpr int kTileRows = kThreadRows;
constexpr int kRow1Threads = 32;
constexpr int kGenericTileK = 64;
constexpr int kSm90TileK = 128;
constexpr int64_t kCublasLtRow1MinOutFeatures = 512;
constexpr int kTensorCoreThreads = 32;
constexpr int kTensorCoreTileM = 16;
constexpr int kTensorCoreTileN = 16;
constexpr int kTensorCoreTileK = 16;
constexpr int kSm90TensorCoreColsPerWarp = 2;
constexpr int kWgmmaThreads = 128;
constexpr int kWgmmaTileM = 64;
constexpr int kWgmmaTileN = 8;
constexpr int kWgmmaTileK = 32;
constexpr int kWgmmaLayoutNoSwizzle = 0;

bool IsSupportedInt8LinearOutDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

bool Int8LinearTensorCoreDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_LINEAR_WMMA");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8LinearWgmmaDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_LINEAR_WGMMA");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8LinearWgmmaEnabled() {
  const char* env = std::getenv("MODEL_STACK_ENABLE_INT8_LINEAR_WGMMA");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8LinearCublasLtDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_LINEAR_CUBLASLT");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8LinearCutlassFusedEnabled() {
  const char* env = std::getenv("MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8LinearRow1Disabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_LINEAR_ROW1");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

int64_t Int8LinearCublasLtMinOps() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_MIN_OPS");
  if (env == nullptr || env[0] == '\0') {
    return 8'000'000;
  }
  return std::max<int64_t>(1, std::strtoll(env, nullptr, 10));
}

int64_t Int8LinearWgmmaMinOps() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_WGMMA_MIN_OPS");
  if (env == nullptr || env[0] == '\0') {
    return 131072;
  }
  return std::max<int64_t>(1, std::strtoll(env, nullptr, 10));
}

template <typename scalar_t>
__global__ void int8_linear_forward_kernel(
    const int8_t* __restrict__ qx,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * out_features;
  if (idx >= total) {
    return;
  }

  const int64_t row = idx / out_features;
  const int64_t col = idx % out_features;
  const int8_t* x_row = qx + (row * in_features);
  const int8_t* w_row = qweight + (col * in_features);

  int acc = 0;
  for (int64_t k = 0; k < in_features; ++k) {
    acc += static_cast<int>(x_row[k]) * static_cast<int>(w_row[k]);
  }

  float value = static_cast<float>(acc) * x_scale[row] * inv_scale[col];
  if (bias != nullptr) {
    value += static_cast<float>(bias[col]);
  }
  out[row * out_features + col] = static_cast<scalar_t>(value);
}

__device__ inline int DotInt8Chunk4(const int8_t* lhs, const int8_t* rhs, int acc) {
#if __CUDA_ARCH__ >= 610
  const int32_t lhs_packed = *reinterpret_cast<const int32_t*>(lhs);
  const int32_t rhs_packed = *reinterpret_cast<const int32_t*>(rhs);
  return __dp4a(lhs_packed, rhs_packed, acc);
#else
  acc += static_cast<int>(lhs[0]) * static_cast<int>(rhs[0]);
  acc += static_cast<int>(lhs[1]) * static_cast<int>(rhs[1]);
  acc += static_cast<int>(lhs[2]) * static_cast<int>(rhs[2]);
  acc += static_cast<int>(lhs[3]) * static_cast<int>(rhs[3]);
  return acc;
#endif
}

template <typename scalar_t, int TileK, int ColsPerThread>
__global__ __launch_bounds__(kRow1Threads, 8) void int8_linear_forward_row1_kernel(
    const int8_t* __restrict__ qx,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t out_features,
    int64_t in_features) {
  constexpr int TileCols = kRow1Threads * ColsPerThread;
  __shared__ int8_t x_tile[TileK];

  const int lane = static_cast<int>(threadIdx.x);
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TileCols + lane;
  int accum[ColsPerThread];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    accum[idx] = 0;
  }

  for (int64_t k0 = 0; k0 < in_features; k0 += TileK) {
    for (int tile_k = lane; tile_k < TileK; tile_k += kRow1Threads) {
      const int64_t global_k = k0 + tile_k;
      x_tile[tile_k] = global_k < in_features ? qx[global_k] : static_cast<int8_t>(0);
    }
    __syncthreads();

    #pragma unroll
    for (int idx = 0; idx < ColsPerThread; ++idx) {
      const int64_t global_col = col0 + idx * kRow1Threads;
      if (global_col >= out_features) {
        continue;
      }
      const int8_t* w_row = qweight + (global_col * in_features) + k0;
      int acc = accum[idx];
      int kk = 0;
      for (; kk + 3 < TileK && k0 + kk + 3 < in_features; kk += 4) {
        acc = DotInt8Chunk4(&x_tile[kk], &w_row[kk], acc);
      }
      for (; kk < TileK && k0 + kk < in_features; ++kk) {
        acc += static_cast<int>(x_tile[kk]) * static_cast<int>(w_row[kk]);
      }
      accum[idx] = acc;
    }
    __syncthreads();
  }

  const float row_scale = x_scale[0];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t global_col = col0 + idx * kRow1Threads;
    if (global_col >= out_features) {
      continue;
    }
    float value = static_cast<float>(accum[idx]) * row_scale * inv_scale[global_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[global_col] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(kWgmmaThreads, 1) void int8_linear_forward_sm90a_wgmma_kernel(
    const int8_t* __restrict__ qx,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  __shared__ __align__(16) uint8_t a_tile[kWgmmaTileM][kWgmmaTileK];
  __shared__ __align__(16) int8_t b_tile[kWgmmaTileK][kWgmmaTileN];
  __shared__ int weight_correction[kWgmmaTileN];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t row_start = static_cast<int64_t>(blockIdx.y) * kWgmmaTileM;
  const int64_t col_start = static_cast<int64_t>(blockIdx.x) * kWgmmaTileN;

  if (tid < kWgmmaTileN) {
    weight_correction[tid] = 0;
  }
  __syncthreads();

  int accum[4] = {0, 0, 0, 0};
  #pragma unroll
  for (int idx = 0; idx < 4; ++idx) {
    t10::cuda::WgmmaFenceOperand(accum[idx]);
  }
  t10::cuda::WgmmaFenceSyncAligned();

  for (int64_t k0 = 0; k0 < in_features; k0 += kWgmmaTileK) {
    for (int idx = tid; idx < kWgmmaTileM * kWgmmaTileK; idx += kWgmmaThreads) {
      const int tile_row = idx / kWgmmaTileK;
      const int tile_k = idx % kWgmmaTileK;
      const int64_t global_row = row_start + tile_row;
      const int64_t global_k = k0 + tile_k;
      uint8_t rebased = 0;
      if (global_row < rows && global_k < in_features) {
        rebased = static_cast<uint8_t>(static_cast<int>(qx[global_row * in_features + global_k]) + 128);
      }
      a_tile[tile_row][tile_k] = rebased;
    }
    for (int idx = tid; idx < kWgmmaTileK * kWgmmaTileN; idx += kWgmmaThreads) {
      const int tile_k = idx / kWgmmaTileN;
      const int tile_col = idx % kWgmmaTileN;
      const int64_t global_k = k0 + tile_k;
      const int64_t global_col = col_start + tile_col;
      b_tile[tile_k][tile_col] =
          (global_col < out_features && global_k < in_features)
              ? qweight[global_col * in_features + global_k]
              : static_cast<int8_t>(0);
    }
    __syncthreads();

    if (tid < kWgmmaTileN) {
      int sum = 0;
      #pragma unroll
      for (int kk = 0; kk < kWgmmaTileK; ++kk) {
        sum += static_cast<int>(b_tile[kk][tid]);
      }
      weight_correction[tid] += sum * 128;
    }
    __syncthreads();

    t10::cuda::AsyncProxyFenceSharedCta();
    const auto desc_a = static_cast<uint64_t>(t10::cuda::MakeWgmmaSmemDesc(
        a_tile,
        kWgmmaLayoutNoSwizzle,
        0,
        kWgmmaTileK * static_cast<int>(sizeof(uint8_t))));
    const auto desc_b = static_cast<uint64_t>(t10::cuda::MakeWgmmaSmemDesc(
        b_tile,
        kWgmmaLayoutNoSwizzle,
        0,
        kWgmmaTileN * static_cast<int>(sizeof(int8_t))));
    t10::cuda::WgmmaM64N8K32S32U8S8(desc_a, desc_b, accum);
    t10::cuda::WgmmaCommitGroupSyncAligned();
    t10::cuda::WgmmaWaitGroupSyncAligned<0>();
    __syncthreads();
  }

  const int64_t global_row = row_start + warp * 16 + (lane & 15);
  const int local_col_base = (lane >> 4) * 4;
  if (global_row >= rows) {
    return;
  }

  const float row_scale = x_scale[global_row];
  #pragma unroll
  for (int idx = 0; idx < 4; ++idx) {
    const int local_col = local_col_base + idx;
    const int64_t global_col = col_start + local_col;
    if (local_col >= kWgmmaTileN || global_col >= out_features) {
      continue;
    }
    const int corrected_acc = accum[idx] - weight_correction[local_col];
    float value = static_cast<float>(corrected_acc) * row_scale * inv_scale[global_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[global_row * out_features + global_col] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, int TileColsPerWarp>
__global__ void int8_linear_forward_sm90_tensorcore_kernel(
    const int8_t* __restrict__ qx,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t row_start = static_cast<int64_t>(blockIdx.y) * kTensorCoreTileM;
  const int64_t col_start = static_cast<int64_t>(blockIdx.x) * (kTensorCoreTileN * TileColsPerWarp);

  __shared__ __align__(16) int8_t x_tile[kTensorCoreTileM * kTensorCoreTileK];
  __shared__ __align__(16) int8_t w_tile[TileColsPerWarp][kTensorCoreTileN * kTensorCoreTileK];
  __shared__ __align__(16) int accum_tile[TileColsPerWarp][kTensorCoreTileM * kTensorCoreTileN];

  wmma::fragment<wmma::accumulator, kTensorCoreTileM, kTensorCoreTileN, kTensorCoreTileK, int>
      acc_frag[TileColsPerWarp];
  #pragma unroll
  for (int tile = 0; tile < TileColsPerWarp; ++tile) {
    wmma::fill_fragment(acc_frag[tile], 0);
  }

  for (int64_t k0 = 0; k0 < in_features; k0 += kTensorCoreTileK) {
    for (int idx = lane; idx < kTensorCoreTileM * kTensorCoreTileK; idx += kTensorCoreThreads) {
      const int tile_row = idx / kTensorCoreTileK;
      const int tile_k = idx % kTensorCoreTileK;
      const int64_t global_row = row_start + tile_row;
      const int64_t global_k = k0 + tile_k;
      x_tile[idx] =
          (global_row < rows && global_k < in_features)
              ? qx[global_row * in_features + global_k]
              : static_cast<int8_t>(0);
    }
    #pragma unroll
    for (int tile = 0; tile < TileColsPerWarp; ++tile) {
      for (int idx = lane; idx < kTensorCoreTileN * kTensorCoreTileK; idx += kTensorCoreThreads) {
        const int tile_col = idx / kTensorCoreTileK;
        const int tile_k = idx % kTensorCoreTileK;
        const int64_t global_col = col_start + tile * kTensorCoreTileN + tile_col;
        const int64_t global_k = k0 + tile_k;
        w_tile[tile][idx] =
            (global_col < out_features && global_k < in_features)
                ? qweight[global_col * in_features + global_k]
                : static_cast<int8_t>(0);
      }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kTensorCoreTileM, kTensorCoreTileN, kTensorCoreTileK, signed char, wmma::row_major>
        a_frag;
    wmma::load_matrix_sync(a_frag, reinterpret_cast<const signed char*>(x_tile), kTensorCoreTileK);
    #pragma unroll
    for (int tile = 0; tile < TileColsPerWarp; ++tile) {
      wmma::fragment<wmma::matrix_b, kTensorCoreTileM, kTensorCoreTileN, kTensorCoreTileK, signed char, wmma::col_major>
          b_frag;
      wmma::load_matrix_sync(
          b_frag,
          reinterpret_cast<const signed char*>(w_tile[tile]),
          kTensorCoreTileK);
      wmma::mma_sync(acc_frag[tile], a_frag, b_frag, acc_frag[tile]);
    }
    __syncthreads();
  }

  #pragma unroll
  for (int tile = 0; tile < TileColsPerWarp; ++tile) {
    wmma::store_matrix_sync(
        accum_tile[tile],
        acc_frag[tile],
        kTensorCoreTileN,
        wmma::mem_row_major);
  }
  __syncthreads();

  for (int idx = lane; idx < TileColsPerWarp * kTensorCoreTileM * kTensorCoreTileN; idx += kTensorCoreThreads) {
    const int tile = idx / (kTensorCoreTileM * kTensorCoreTileN);
    const int local = idx % (kTensorCoreTileM * kTensorCoreTileN);
    const int tile_row = local / kTensorCoreTileN;
    const int tile_col = local % kTensorCoreTileN;
    const int64_t global_row = row_start + tile_row;
    const int64_t global_col = col_start + tile * kTensorCoreTileN + tile_col;
    if (global_row >= rows || global_col >= out_features) {
      continue;
    }
    float value =
        static_cast<float>(accum_tile[tile][local]) * x_scale[global_row] * inv_scale[global_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[global_row * out_features + global_col] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, int TileK, int ColsPerThread>
__global__ __launch_bounds__(kThreads, 2) void int8_linear_forward_tiled_kernel(
    const int8_t* __restrict__ qx,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  constexpr int TileCols = kThreadCols * ColsPerThread;
  __shared__ int8_t x_tile[kTileRows][TileK];
  __shared__ int8_t w_tile[TileCols][TileK];

  const int local_col = static_cast<int>(threadIdx.x);
  const int local_row = static_cast<int>(threadIdx.y);
  const int thread_linear = local_row * kThreadCols + local_col;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kTileRows + local_row;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TileCols + local_col;

  int accum[ColsPerThread];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    accum[idx] = 0;
  }

  for (int64_t k0 = 0; k0 < in_features; k0 += TileK) {
    for (int index = thread_linear; index < kTileRows * TileK; index += kThreads) {
      const int tile_row = index / TileK;
      const int tile_k = index % TileK;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kTileRows + tile_row;
      const int64_t global_k = k0 + tile_k;
      x_tile[tile_row][tile_k] =
          (global_row < rows && global_k < in_features) ? qx[global_row * in_features + global_k] : static_cast<int8_t>(0);
    }
    for (int index = thread_linear; index < TileCols * TileK; index += kThreads) {
      const int tile_col = index / TileK;
      const int tile_k = index % TileK;
      const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TileCols + tile_col;
      const int64_t global_k = k0 + tile_k;
      w_tile[tile_col][tile_k] =
          (global_col < out_features && global_k < in_features)
              ? qweight[global_col * in_features + global_k]
              : static_cast<int8_t>(0);
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
        int acc = accum[idx];
        int kk = 0;
        for (; kk + 3 < TileK && k0 + kk + 3 < in_features; kk += 4) {
          acc = DotInt8Chunk4(&x_tile[local_row][kk], &w_tile[tile_col][kk], acc);
        }
        for (; kk < TileK && k0 + kk < in_features; ++kk) {
          acc += static_cast<int>(x_tile[local_row][kk]) * static_cast<int>(w_tile[tile_col][kk]);
        }
        accum[idx] = acc;
      }
    }
    __syncthreads();
  }

  if (row >= rows) {
    return;
  }
  const float row_scale = x_scale[row];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t global_col = col0 + idx * kThreadCols;
    if (global_col >= out_features) {
      continue;
    }
    float value = static_cast<float>(accum[idx]) * row_scale * inv_scale[global_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[row * out_features + global_col] = static_cast<scalar_t>(value);
  }
}

inline bool UseTiledInt8Kernel(int64_t rows, int64_t out_features, int64_t in_features) {
  return rows > 0 && out_features > 0 && in_features >= 32 && (rows * out_features) >= 256;
}

inline bool UseRow1Int8Kernel(int64_t rows, int64_t out_features, int64_t in_features) {
  if (Int8LinearRow1Disabled()) {
    return false;
  }
  return rows == 1 && out_features > 0 && in_features >= 32;
}

inline bool UseSm90Int8TensorCoreKernel(
    const torch::Tensor& reference,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  if (Int8LinearTensorCoreDisabled()) {
    return false;
  }
  if (!t10::cuda::DeviceIsSm90OrLater(reference)) {
    return false;
  }
  if (rows < kTensorCoreTileM || out_features < kTensorCoreTileN || in_features < kTensorCoreTileK) {
    return false;
  }
  return (rows * out_features * in_features) >= 131072;
}

inline bool UseSm90aInt8WgmmaKernel(
    const torch::Tensor& reference,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  if (!t10::cuda::BuildRequestsSm90aExperimental()) {
    return false;
  }
  if (!Int8LinearWgmmaEnabled() || Int8LinearWgmmaDisabled()) {
    return false;
  }
  if (!t10::cuda::DeviceIsSm90OrLater(reference)) {
    return false;
  }
  if (rows < kWgmmaTileM || out_features <= 0 || in_features < kWgmmaTileK) {
    return false;
  }
  if (out_features < kWgmmaTileN) {
    return false;
  }
  return (rows * out_features * in_features) >= Int8LinearWgmmaMinOps();
}

inline bool UseCublasLtInt8Backend(
    const torch::Tensor& reference,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  if (rows < kTensorCoreTileM || out_features < kTensorCoreTileN || in_features < kTensorCoreTileK) {
    if (!(rows == 1 && out_features >= kCublasLtRow1MinOutFeatures && in_features >= kTensorCoreTileK)) {
      return false;
    }
  }
  if ((in_features % kTensorCoreTileK) != 0) {
    return false;
  }
  if (rows == 1 && out_features >= kCublasLtRow1MinOutFeatures) {
    return true;
  }
  return (rows * out_features * in_features) >= Int8LinearCublasLtMinOps();
}

}  // namespace

bool HasCudaInt8LinearKernel() {
  return true;
}

torch::Tensor CudaInt8LinearForward(
    const torch::Tensor& qx,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(qx.is_cuda(), "CudaInt8LinearForward: qx must be a CUDA tensor");
  TORCH_CHECK(x_scale.is_cuda(), "CudaInt8LinearForward: x_scale must be a CUDA tensor");
  TORCH_CHECK(qweight.is_cuda(), "CudaInt8LinearForward: qweight must be a CUDA tensor");
  TORCH_CHECK(inv_scale.is_cuda(), "CudaInt8LinearForward: inv_scale must be a CUDA tensor");
  TORCH_CHECK(qx.scalar_type() == torch::kInt8, "CudaInt8LinearForward: qx must use int8 storage");
  TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32, "CudaInt8LinearForward: x_scale must use float32 storage");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt8, "CudaInt8LinearForward: qweight must use int8 storage");
  TORCH_CHECK(inv_scale.scalar_type() == torch::kFloat32, "CudaInt8LinearForward: inv_scale must use float32 storage");
  TORCH_CHECK(qx.dim() >= 2, "CudaInt8LinearForward: qx must have rank >= 2");
  TORCH_CHECK(qweight.dim() == 2, "CudaInt8LinearForward: qweight must be rank-2");
  TORCH_CHECK(inv_scale.dim() == 1, "CudaInt8LinearForward: inv_scale must be rank-1");

  const auto output_dtype = out_dtype.has_value() ? out_dtype.value() : torch::kFloat32;
  TORCH_CHECK(IsSupportedInt8LinearOutDtype(output_dtype), "CudaInt8LinearForward: unsupported output dtype");

  c10::cuda::CUDAGuard device_guard{qx.device()};

  auto qx_contig = qx.contiguous();
  auto x_scale_contig = x_scale.contiguous();
  auto qweight_contig = qweight.contiguous();
  auto inv_scale_contig = inv_scale.contiguous();
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    auto bias_value = bias.value().to(qx_contig.device(), output_dtype).contiguous();
    bias_cast = bias_value;
  }

  const auto in_features = qx_contig.size(-1);
  const auto rows = qx_contig.numel() / in_features;
  const auto out_features = qweight_contig.size(0);
  TORCH_CHECK(x_scale_contig.numel() == rows, "CudaInt8LinearForward: x_scale size mismatch");
  TORCH_CHECK(qweight_contig.size(1) == in_features, "CudaInt8LinearForward: qweight column count mismatch");
  TORCH_CHECK(inv_scale_contig.size(0) == out_features, "CudaInt8LinearForward: inv_scale size mismatch");

  auto out_2d = torch::empty(
      {rows, out_features},
      qx_contig.options().dtype(output_dtype));
  if (rows == 0 || out_features == 0) {
    std::vector<int64_t> out_sizes(qx_contig.sizes().begin(), qx_contig.sizes().end());
    out_sizes.back() = out_features;
    return out_2d.view(out_sizes);
  }

  const dim3 threads(kThreads);
  const dim3 blocks(static_cast<unsigned int>((rows * out_features + kThreads - 1) / kThreads));
  auto stream = c10::cuda::getCurrentCUDAStream(qx.get_device());
  const bool use_tiled = UseTiledInt8Kernel(rows, out_features, in_features);
  const bool use_row1 = UseRow1Int8Kernel(rows, out_features, in_features);
  const bool use_cublaslt =
      !Int8LinearCublasLtDisabled() && UseCublasLtInt8Backend(qx_contig, rows, out_features, in_features);
  const bool use_sm90a_wgmma =
      !use_row1 && !use_cublaslt && UseSm90aInt8WgmmaKernel(qx_contig, rows, out_features, in_features);
  const bool use_sm90_tensorcore =
      !use_sm90a_wgmma &&
      !use_row1 && !use_cublaslt && UseSm90Int8TensorCoreKernel(qx_contig, rows, out_features, in_features);
  const bool use_sm90 = !use_row1 && !use_sm90_tensorcore && use_tiled && t10::cuda::DeviceIsSm90OrLater(qx_contig) &&
      in_features >= kSm90TileK;
  const dim3 tile_threads(kThreadCols, kThreadRows);

  if (Int8LinearCutlassFusedEnabled() && !use_row1) {
    auto out = CutlassInt8LinearFusedForward(
        qx_contig,
        x_scale_contig,
        qweight_contig,
        inv_scale_contig,
        bias_cast,
        c10::optional<torch::ScalarType>(output_dtype));
    if (out.defined()) {
      return out;
    }
  }

  if (use_cublaslt) {
    auto out = CublasLtInt8LinearForward(
        qx_contig,
        x_scale_contig,
        qweight_contig,
        inv_scale_contig,
        bias_cast,
        c10::optional<torch::ScalarType>(output_dtype));
    if (out.defined()) {
      return out;
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output_dtype,
      "model_stack_cuda_int8_linear_forward",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias_cast.has_value() && bias_cast.value().defined()) {
          bias_ptr = bias_cast.value().data_ptr<scalar_t>();
        }
        if (use_row1) {
          constexpr int kRow1ColsPerThread = 4;
          const dim3 row1_blocks(
              static_cast<unsigned int>((out_features + (kRow1Threads * kRow1ColsPerThread) - 1) /
                                        (kRow1Threads * kRow1ColsPerThread)));
          if (in_features >= kSm90TileK) {
            int8_linear_forward_row1_kernel<scalar_t, kSm90TileK, kRow1ColsPerThread>
                <<<row1_blocks, kRow1Threads, 0, stream.stream()>>>(
                    qx_contig.data_ptr<int8_t>(),
                    x_scale_contig.data_ptr<float>(),
                    qweight_contig.data_ptr<int8_t>(),
                    inv_scale_contig.data_ptr<float>(),
                    bias_ptr,
                    out_2d.data_ptr<scalar_t>(),
                    out_features,
                    in_features);
          } else {
            int8_linear_forward_row1_kernel<scalar_t, kGenericTileK, kRow1ColsPerThread>
                <<<row1_blocks, kRow1Threads, 0, stream.stream()>>>(
                    qx_contig.data_ptr<int8_t>(),
                    x_scale_contig.data_ptr<float>(),
                    qweight_contig.data_ptr<int8_t>(),
                    inv_scale_contig.data_ptr<float>(),
                    bias_ptr,
                    out_2d.data_ptr<scalar_t>(),
                    out_features,
                    in_features);
          }
        } else if (use_sm90a_wgmma) {
          const dim3 wgmma_blocks(
              static_cast<unsigned int>((out_features + kWgmmaTileN - 1) / kWgmmaTileN),
              static_cast<unsigned int>((rows + kWgmmaTileM - 1) / kWgmmaTileM));
          int8_linear_forward_sm90a_wgmma_kernel<scalar_t><<<wgmma_blocks, kWgmmaThreads, 0, stream.stream()>>>(
              qx_contig.data_ptr<int8_t>(),
              x_scale_contig.data_ptr<float>(),
              qweight_contig.data_ptr<int8_t>(),
              inv_scale_contig.data_ptr<float>(),
              bias_ptr,
              out_2d.data_ptr<scalar_t>(),
              rows,
              out_features,
              in_features);
        } else if (use_sm90_tensorcore) {
          const dim3 tc_blocks(
              static_cast<unsigned int>(
                  (out_features + (kTensorCoreTileN * kSm90TensorCoreColsPerWarp) - 1) /
                  (kTensorCoreTileN * kSm90TensorCoreColsPerWarp)),
              static_cast<unsigned int>((rows + kTensorCoreTileM - 1) / kTensorCoreTileM));
          int8_linear_forward_sm90_tensorcore_kernel<scalar_t, kSm90TensorCoreColsPerWarp>
              <<<tc_blocks, kTensorCoreThreads, 0, stream.stream()>>>(
                  qx_contig.data_ptr<int8_t>(),
                  x_scale_contig.data_ptr<float>(),
                  qweight_contig.data_ptr<int8_t>(),
                  inv_scale_contig.data_ptr<float>(),
                  bias_ptr,
                  out_2d.data_ptr<scalar_t>(),
                  rows,
                  out_features,
                  in_features);
        } else if (use_sm90) {
          const dim3 tile_blocks(
              static_cast<unsigned int>((out_features + (kThreadCols * 2) - 1) / (kThreadCols * 2)),
              static_cast<unsigned int>((rows + kTileRows - 1) / kTileRows));
          int8_linear_forward_tiled_kernel<scalar_t, kSm90TileK, 2><<<tile_blocks, tile_threads, 0, stream.stream()>>>(
              qx_contig.data_ptr<int8_t>(),
              x_scale_contig.data_ptr<float>(),
              qweight_contig.data_ptr<int8_t>(),
              inv_scale_contig.data_ptr<float>(),
              bias_ptr,
              out_2d.data_ptr<scalar_t>(),
              rows,
              out_features,
              in_features);
        } else if (use_tiled) {
          const dim3 tile_blocks(
              static_cast<unsigned int>((out_features + kThreadCols - 1) / kThreadCols),
              static_cast<unsigned int>((rows + kTileRows - 1) / kTileRows));
          int8_linear_forward_tiled_kernel<scalar_t, kGenericTileK, 1><<<tile_blocks, tile_threads, 0, stream.stream()>>>(
              qx_contig.data_ptr<int8_t>(),
              x_scale_contig.data_ptr<float>(),
              qweight_contig.data_ptr<int8_t>(),
              inv_scale_contig.data_ptr<float>(),
              bias_ptr,
              out_2d.data_ptr<scalar_t>(),
              rows,
              out_features,
              in_features);
        } else {
          int8_linear_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
              qx_contig.data_ptr<int8_t>(),
              x_scale_contig.data_ptr<float>(),
              qweight_contig.data_ptr<int8_t>(),
              inv_scale_contig.data_ptr<float>(),
              bias_ptr,
              out_2d.data_ptr<scalar_t>(),
              rows,
              out_features,
              in_features);
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(qx_contig.sizes().begin(), qx_contig.sizes().end());
  out_sizes.back() = out_features;
  return out_2d.view(out_sizes);
}
