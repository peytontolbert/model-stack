#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <vector>

torch::Tensor CudaInt8LinearForward(
    const torch::Tensor& qx,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);
torch::Tensor CudaInt8AttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& q_scale,
    const torch::Tensor& k,
    const torch::Tensor& k_scale,
    const torch::Tensor& v,
    const torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    const c10::optional<torch::ScalarType>& out_dtype);

namespace {

constexpr int kQuantThreads = 256;
constexpr int kQuantWarpSize = 32;
constexpr int kQuantWarpThreads = 256;
constexpr int kQuantWarpRowsPerBlock = kQuantWarpThreads / kQuantWarpSize;
constexpr int kQuantWarp4RowsPerBlock = 4;
constexpr int kQuantWarp4Threads = kQuantWarp4RowsPerBlock * kQuantWarpSize;
constexpr int kQuantWarp16RowsPerBlock = 16;
constexpr int kQuantWarp16Threads = kQuantWarp16RowsPerBlock * kQuantWarpSize;
constexpr int kQuantWideWarpsPerRow = 2;
constexpr int kQuantWideGroupSize = kQuantWideWarpsPerRow * kQuantWarpSize;
constexpr int kQuantWideRowsPerBlock = kQuantThreads / kQuantWideGroupSize;

bool Int8QuantWarpRowEnabled(int64_t cols) {
  const char* disable = std::getenv("MODEL_STACK_DISABLE_INT8_QUANT_WARP_ROW");
  if (disable != nullptr && disable[0] != '\0' && disable[0] != '0') {
    return false;
  }
  return cols <= 4096;
}

bool Int8QuantWideRowEnabled(int64_t cols) {
  return Int8QuantWarpRowEnabled(cols) && cols > 2048;
}

int Int8QuantWarpRowsPerBlock() {
  const char* env = std::getenv("MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK");
  if (env == nullptr || env[0] == '\0') {
    return kQuantWarpRowsPerBlock;
  }
  const int value = std::atoi(env);
  if (value == kQuantWarp4RowsPerBlock || value == kQuantWarpRowsPerBlock || value == kQuantWarp16RowsPerBlock) {
    return value;
  }
  return kQuantWarpRowsPerBlock;
}

bool Int8QuantSharedCacheEnabled(int64_t cols, int rows_per_block, size_t element_size) {
  const char* disable = std::getenv("MODEL_STACK_DISABLE_INT8_QUANT_SHARED_CACHE");
  if (disable != nullptr && disable[0] != '\0' && disable[0] != '0') {
    return false;
  }
  constexpr size_t kMaxDefaultDynamicSharedBytes = 48 * 1024;
  const size_t bytes =
      static_cast<size_t>(rows_per_block) * static_cast<size_t>(cols) * element_size;
  return bytes <= kMaxDefaultDynamicSharedBytes;
}

bool Int8QuantVec4Enabled(int64_t cols) {
  const char* enable = std::getenv("MODEL_STACK_ENABLE_INT8_QUANT_VEC4");
  if (enable == nullptr || enable[0] == '\0' || enable[0] == '0') {
    return false;
  }
  return (cols % 4) == 0;
}

bool IsSupportedInt8FrontendDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

__device__ __forceinline__ int8_t QuantizeScaledToInt8(float scaled) {
  int q = __float2int_rn(scaled);
  q = q < -127 ? -127 : q;
  q = q > 127 ? 127 : q;
  return static_cast<int8_t>(q);
}

__device__ __forceinline__ uint32_t PackInt8x4(int8_t q0, int8_t q1, int8_t q2, int8_t q3) {
  return static_cast<uint32_t>(static_cast<uint8_t>(q0)) |
      (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
      (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
      (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t Relu2Cast(scalar_t value) {
  const float v = static_cast<float>(value);
  const float y = v > 0.0f ? v : 0.0f;
  return static_cast<scalar_t>(y * y);
}

template <typename scalar_t>
__global__ void quantize_activation_int8_rowwise_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  __shared__ float shared_max[kQuantThreads];
  float local_max = 0.0f;
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = fabsf(static_cast<float>(x[row * cols + col]));
    local_max = fmaxf(local_max, value);
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  float scale = 1.0f;
  if (provided_scale != nullptr) {
    scale = provided_scale[provided_rows == 1 ? 0 : row];
    if (!(scale > 0.0f)) {
      scale = 1.0f;
    }
  } else {
    scale = shared_max[0] > 0.0f ? (shared_max[0] / 127.0f) : 1.0f;
  }
  if (threadIdx.x == 0) {
    row_scale[row] = scale;
  }
  __syncthreads();

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float scaled = static_cast<float>(x[row * cols + col]) / scale;
    qx[row * cols + col] = QuantizeScaledToInt8(scaled);
  }
}

template <typename scalar_t>
__global__ void quantize_relu2_activation_int8_rowwise_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  __shared__ float shared_max[kQuantThreads];
  float local_max = 0.0f;
  const int64_t row_offset = row * cols;
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const scalar_t value_raw = Relu2Cast(x[row_offset + col]);
    local_max = fmaxf(local_max, fabsf(static_cast<float>(value_raw)));
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float scale = shared_max[0] > 0.0f ? (shared_max[0] / 127.0f) : 1.0f;
  if (threadIdx.x == 0) {
    row_scale[row] = scale;
  }
  __syncthreads();

  const float inv_scale = 1.0f / scale;
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float scaled = static_cast<float>(Relu2Cast(x[row_offset + col])) * inv_scale;
    qx[row_offset + col] = QuantizeScaledToInt8(scaled);
  }
}

template <typename scalar_t, int RowsPerBlock>
__global__ void quantize_activation_int8_rowwise_warp_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  const int warp_id = threadIdx.x / kQuantWarpSize;
  const int lane = threadIdx.x & (kQuantWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * RowsPerBlock + warp_id;
  if (row >= rows) {
    return;
  }

  float local_max = 0.0f;
  const int64_t row_offset = row * cols;
  for (int64_t col = lane; col < cols; col += kQuantWarpSize) {
    const float value = fabsf(static_cast<float>(x[row_offset + col]));
    local_max = fmaxf(local_max, value);
  }
  unsigned int mask = 0xffffffffu;
  for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
  }

  float scale = 1.0f;
  if (lane == 0) {
    if (provided_scale != nullptr) {
      scale = provided_scale[provided_rows == 1 ? 0 : row];
      if (!(scale > 0.0f)) {
        scale = 1.0f;
      }
    } else {
      scale = local_max > 0.0f ? (local_max / 127.0f) : 1.0f;
    }
    row_scale[row] = scale;
  }
  scale = __shfl_sync(mask, scale, 0);
  const float inv_scale = 1.0f / scale;

  for (int64_t col = lane; col < cols; col += kQuantWarpSize) {
    const float scaled = static_cast<float>(x[row_offset + col]) * inv_scale;
    qx[row_offset + col] = QuantizeScaledToInt8(scaled);
  }
}

template <typename scalar_t, int RowsPerBlock>
__global__ void quantize_relu2_activation_int8_rowwise_warp_cached_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  extern __shared__ __align__(16) unsigned char shared_values_raw[];
  auto* shared_values = reinterpret_cast<scalar_t*>(shared_values_raw);
  const int warp_id = threadIdx.x / kQuantWarpSize;
  const int lane = threadIdx.x & (kQuantWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * RowsPerBlock + warp_id;
  if (row >= rows) {
    return;
  }

  float local_max = 0.0f;
  const int64_t row_offset = row * cols;
  scalar_t* row_values = shared_values + static_cast<int64_t>(warp_id) * cols;
  for (int64_t col = lane; col < cols; col += kQuantWarpSize) {
    const scalar_t value_raw = Relu2Cast(x[row_offset + col]);
    row_values[col] = value_raw;
    local_max = fmaxf(local_max, fabsf(static_cast<float>(value_raw)));
  }
  unsigned int mask = 0xffffffffu;
  for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
  }

  float scale = 1.0f;
  if (lane == 0) {
    scale = local_max > 0.0f ? (local_max / 127.0f) : 1.0f;
    row_scale[row] = scale;
  }
  scale = __shfl_sync(mask, scale, 0);
  const float inv_scale = 1.0f / scale;

  for (int64_t col = lane; col < cols; col += kQuantWarpSize) {
    const float scaled = static_cast<float>(row_values[col]) * inv_scale;
    qx[row_offset + col] = QuantizeScaledToInt8(scaled);
  }
}

template <typename scalar_t, int RowsPerBlock>
__global__ void quantize_activation_int8_rowwise_warp_provided_scale_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  const int warp_id = threadIdx.x / kQuantWarpSize;
  const int lane = threadIdx.x & (kQuantWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * RowsPerBlock + warp_id;
  if (row >= rows) {
    return;
  }

  float scale = 1.0f;
  if (lane == 0) {
    scale = provided_scale[provided_rows == 1 ? 0 : row];
    if (!(scale > 0.0f)) {
      scale = 1.0f;
    }
    row_scale[row] = scale;
  }
  unsigned int mask = 0xffffffffu;
  scale = __shfl_sync(mask, scale, 0);
  const float inv_scale = 1.0f / scale;

  const int64_t row_offset = row * cols;
  for (int64_t col = lane; col < cols; col += kQuantWarpSize) {
    const float scaled = static_cast<float>(x[row_offset + col]) * inv_scale;
    qx[row_offset + col] = QuantizeScaledToInt8(scaled);
  }
}

template <typename scalar_t, int RowsPerBlock>
__global__ void quantize_activation_int8_rowwise_warp_provided_scale_vec4_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  const int warp_id = threadIdx.x / kQuantWarpSize;
  const int lane = threadIdx.x & (kQuantWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * RowsPerBlock + warp_id;
  if (row >= rows) {
    return;
  }

  float scale = 1.0f;
  if (lane == 0) {
    scale = provided_scale[provided_rows == 1 ? 0 : row];
    if (!(scale > 0.0f)) {
      scale = 1.0f;
    }
    row_scale[row] = scale;
  }
  unsigned int mask = 0xffffffffu;
  scale = __shfl_sync(mask, scale, 0);
  const float inv_scale = 1.0f / scale;

  const int64_t row_offset = row * cols;
  for (int64_t col = static_cast<int64_t>(lane) * 4; col < cols; col += kQuantWarpSize * 4) {
    const auto v0 = static_cast<float>(x[row_offset + col + 0]);
    const auto v1 = static_cast<float>(x[row_offset + col + 1]);
    const auto v2 = static_cast<float>(x[row_offset + col + 2]);
    const auto v3 = static_cast<float>(x[row_offset + col + 3]);
    const uint32_t packed = PackInt8x4(
        QuantizeScaledToInt8(v0 * inv_scale),
        QuantizeScaledToInt8(v1 * inv_scale),
        QuantizeScaledToInt8(v2 * inv_scale),
        QuantizeScaledToInt8(v3 * inv_scale));
    *reinterpret_cast<uint32_t*>(qx + row_offset + col) = packed;
  }
}

template <typename scalar_t, int RowsPerBlock>
__global__ void quantize_activation_int8_rowwise_warp_cached_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  extern __shared__ __align__(16) unsigned char shared_values_raw[];
  auto* shared_values = reinterpret_cast<scalar_t*>(shared_values_raw);
  const int warp_id = threadIdx.x / kQuantWarpSize;
  const int lane = threadIdx.x & (kQuantWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * RowsPerBlock + warp_id;
  if (row >= rows) {
    return;
  }

  float local_max = 0.0f;
  const int64_t row_offset = row * cols;
  scalar_t* row_values = shared_values + static_cast<int64_t>(warp_id) * cols;
  for (int64_t col = lane; col < cols; col += kQuantWarpSize) {
    const scalar_t value_raw = x[row_offset + col];
    row_values[col] = value_raw;
    local_max = fmaxf(local_max, fabsf(static_cast<float>(value_raw)));
  }
  unsigned int mask = 0xffffffffu;
  for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
  }

  float scale = 1.0f;
  if (lane == 0) {
    scale = local_max > 0.0f ? (local_max / 127.0f) : 1.0f;
    row_scale[row] = scale;
  }
  scale = __shfl_sync(mask, scale, 0);
  const float inv_scale = 1.0f / scale;

  for (int64_t col = lane; col < cols; col += kQuantWarpSize) {
    const float scaled = static_cast<float>(row_values[col]) * inv_scale;
    qx[row_offset + col] = QuantizeScaledToInt8(scaled);
  }
}

template <typename scalar_t, int RowsPerBlock>
__global__ void quantize_activation_int8_rowwise_warp_cached_vec4_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  extern __shared__ __align__(16) unsigned char shared_values_raw[];
  auto* shared_values = reinterpret_cast<scalar_t*>(shared_values_raw);
  const int warp_id = threadIdx.x / kQuantWarpSize;
  const int lane = threadIdx.x & (kQuantWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * RowsPerBlock + warp_id;
  if (row >= rows) {
    return;
  }

  float local_max = 0.0f;
  const int64_t row_offset = row * cols;
  scalar_t* row_values = shared_values + static_cast<int64_t>(warp_id) * cols;
  for (int64_t col = static_cast<int64_t>(lane) * 4; col < cols; col += kQuantWarpSize * 4) {
    const scalar_t v0 = x[row_offset + col + 0];
    const scalar_t v1 = x[row_offset + col + 1];
    const scalar_t v2 = x[row_offset + col + 2];
    const scalar_t v3 = x[row_offset + col + 3];
    row_values[col + 0] = v0;
    row_values[col + 1] = v1;
    row_values[col + 2] = v2;
    row_values[col + 3] = v3;
    local_max = fmaxf(local_max, fabsf(static_cast<float>(v0)));
    local_max = fmaxf(local_max, fabsf(static_cast<float>(v1)));
    local_max = fmaxf(local_max, fabsf(static_cast<float>(v2)));
    local_max = fmaxf(local_max, fabsf(static_cast<float>(v3)));
  }
  unsigned int mask = 0xffffffffu;
  for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
  }

  float scale = 1.0f;
  if (lane == 0) {
    scale = local_max > 0.0f ? (local_max / 127.0f) : 1.0f;
    row_scale[row] = scale;
  }
  scale = __shfl_sync(mask, scale, 0);
  const float inv_scale = 1.0f / scale;

  for (int64_t col = static_cast<int64_t>(lane) * 4; col < cols; col += kQuantWarpSize * 4) {
    const uint32_t packed = PackInt8x4(
        QuantizeScaledToInt8(static_cast<float>(row_values[col + 0]) * inv_scale),
        QuantizeScaledToInt8(static_cast<float>(row_values[col + 1]) * inv_scale),
        QuantizeScaledToInt8(static_cast<float>(row_values[col + 2]) * inv_scale),
        QuantizeScaledToInt8(static_cast<float>(row_values[col + 3]) * inv_scale));
    *reinterpret_cast<uint32_t*>(qx + row_offset + col) = packed;
  }
}

template <typename scalar_t>
__global__ void quantize_relu2_activation_int8_rowwise_wide_cached_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  extern __shared__ __align__(16) unsigned char shared_values_raw[];
  auto* shared_values = reinterpret_cast<scalar_t*>(shared_values_raw);
  const int group_id = threadIdx.x / kQuantWideGroupSize;
  const int group_lane = threadIdx.x & (kQuantWideGroupSize - 1);
  const int warp_id = threadIdx.x / kQuantWarpSize;
  const int lane = threadIdx.x & (kQuantWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kQuantWideRowsPerBlock + group_id;
  const bool row_valid = row < rows;

  __shared__ float shared_warp_max[kQuantThreads / kQuantWarpSize];
  __shared__ float shared_scale[kQuantWideRowsPerBlock];

  float local_max = 0.0f;
  const int64_t row_offset = row * cols;
  scalar_t* row_values = shared_values + static_cast<int64_t>(group_id) * cols;
  if (row_valid) {
    for (int64_t col = group_lane; col < cols; col += kQuantWideGroupSize) {
      const scalar_t value_raw = Relu2Cast(x[row_offset + col]);
      row_values[col] = value_raw;
      local_max = fmaxf(local_max, fabsf(static_cast<float>(value_raw)));
    }
  }
  unsigned int mask = 0xffffffffu;
  for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
  }
  if (row_valid && lane == 0) {
    shared_warp_max[warp_id] = local_max;
  }
  __syncthreads();

  if (row_valid && group_lane < kQuantWarpSize) {
    float group_max = group_lane < kQuantWideWarpsPerRow
        ? shared_warp_max[group_id * kQuantWideWarpsPerRow + group_lane]
        : 0.0f;
    for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
      group_max = fmaxf(group_max, __shfl_down_sync(mask, group_max, offset));
    }
    if (group_lane == 0) {
      const float scale = group_max > 0.0f ? (group_max / 127.0f) : 1.0f;
      row_scale[row] = scale;
      shared_scale[group_id] = scale;
    }
  }
  __syncthreads();

  if (row_valid) {
    const float inv_scale = 1.0f / shared_scale[group_id];
    for (int64_t col = group_lane; col < cols; col += kQuantWideGroupSize) {
      const float scaled = static_cast<float>(row_values[col]) * inv_scale;
      qx[row_offset + col] = QuantizeScaledToInt8(scaled);
    }
  }
}

template <typename scalar_t>
__global__ void quantize_activation_int8_rowwise_wide_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  const int group_id = threadIdx.x / kQuantWideGroupSize;
  const int group_lane = threadIdx.x & (kQuantWideGroupSize - 1);
  const int warp_id = threadIdx.x / kQuantWarpSize;
  const int lane = threadIdx.x & (kQuantWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kQuantWideRowsPerBlock + group_id;
  const bool row_valid = row < rows;

  __shared__ float shared_warp_max[kQuantThreads / kQuantWarpSize];
  __shared__ float shared_scale[kQuantWideRowsPerBlock];

  float local_max = 0.0f;
  const int64_t row_offset = row * cols;
  if (row_valid) {
    for (int64_t col = group_lane; col < cols; col += kQuantWideGroupSize) {
      const float value = fabsf(static_cast<float>(x[row_offset + col]));
      local_max = fmaxf(local_max, value);
    }
  }
  unsigned int mask = 0xffffffffu;
  for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
  }
  if (row_valid && lane == 0) {
    shared_warp_max[warp_id] = local_max;
  }
  __syncthreads();

  if (row_valid && group_lane < kQuantWarpSize) {
    float group_max = group_lane < kQuantWideWarpsPerRow
        ? shared_warp_max[group_id * kQuantWideWarpsPerRow + group_lane]
        : 0.0f;
    for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
      group_max = fmaxf(group_max, __shfl_down_sync(mask, group_max, offset));
    }
    if (group_lane == 0) {
      float scale = 1.0f;
      if (provided_scale != nullptr) {
        scale = provided_scale[provided_rows == 1 ? 0 : row];
        if (!(scale > 0.0f)) {
          scale = 1.0f;
        }
      } else {
        scale = group_max > 0.0f ? (group_max / 127.0f) : 1.0f;
      }
      row_scale[row] = scale;
      shared_scale[group_id] = scale;
    }
  }
  __syncthreads();

  if (row_valid) {
    const float inv_scale = 1.0f / shared_scale[group_id];
    for (int64_t col = group_lane; col < cols; col += kQuantWideGroupSize) {
      const float scaled = static_cast<float>(x[row_offset + col]) * inv_scale;
      qx[row_offset + col] = QuantizeScaledToInt8(scaled);
    }
  }
}

template <typename scalar_t>
__global__ void quantize_activation_int8_rowwise_wide_provided_scale_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  const int group_id = threadIdx.x / kQuantWideGroupSize;
  const int group_lane = threadIdx.x & (kQuantWideGroupSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kQuantWideRowsPerBlock + group_id;
  const bool row_valid = row < rows;
  __shared__ float shared_scale[kQuantWideRowsPerBlock];

  if (row_valid && group_lane == 0) {
    float scale = provided_scale[provided_rows == 1 ? 0 : row];
    if (!(scale > 0.0f)) {
      scale = 1.0f;
    }
    row_scale[row] = scale;
    shared_scale[group_id] = scale;
  }
  __syncthreads();

  if (row_valid) {
    const float inv_scale = 1.0f / shared_scale[group_id];
    const int64_t row_offset = row * cols;
    for (int64_t col = group_lane; col < cols; col += kQuantWideGroupSize) {
      const float scaled = static_cast<float>(x[row_offset + col]) * inv_scale;
      qx[row_offset + col] = QuantizeScaledToInt8(scaled);
    }
  }
}

template <typename scalar_t>
__global__ void quantize_activation_int8_rowwise_wide_cached_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols) {
  extern __shared__ __align__(16) unsigned char shared_values_raw[];
  auto* shared_values = reinterpret_cast<scalar_t*>(shared_values_raw);
  const int group_id = threadIdx.x / kQuantWideGroupSize;
  const int group_lane = threadIdx.x & (kQuantWideGroupSize - 1);
  const int warp_id = threadIdx.x / kQuantWarpSize;
  const int lane = threadIdx.x & (kQuantWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kQuantWideRowsPerBlock + group_id;
  const bool row_valid = row < rows;

  __shared__ float shared_warp_max[kQuantThreads / kQuantWarpSize];
  __shared__ float shared_scale[kQuantWideRowsPerBlock];

  float local_max = 0.0f;
  const int64_t row_offset = row * cols;
  scalar_t* row_values = shared_values + static_cast<int64_t>(group_id) * cols;
  if (row_valid) {
    for (int64_t col = group_lane; col < cols; col += kQuantWideGroupSize) {
      const scalar_t value_raw = x[row_offset + col];
      row_values[col] = value_raw;
      local_max = fmaxf(local_max, fabsf(static_cast<float>(value_raw)));
    }
  }
  unsigned int mask = 0xffffffffu;
  for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
  }
  if (row_valid && lane == 0) {
    shared_warp_max[warp_id] = local_max;
  }
  __syncthreads();

  if (row_valid && group_lane < kQuantWarpSize) {
    float group_max = group_lane < kQuantWideWarpsPerRow
        ? shared_warp_max[group_id * kQuantWideWarpsPerRow + group_lane]
        : 0.0f;
    for (int offset = kQuantWarpSize / 2; offset > 0; offset >>= 1) {
      group_max = fmaxf(group_max, __shfl_down_sync(mask, group_max, offset));
    }
    if (group_lane == 0) {
      const float scale = group_max > 0.0f ? (group_max / 127.0f) : 1.0f;
      row_scale[row] = scale;
      shared_scale[group_id] = scale;
    }
  }
  __syncthreads();

  if (row_valid) {
    const float inv_scale = 1.0f / shared_scale[group_id];
    for (int64_t col = group_lane; col < cols; col += kQuantWideGroupSize) {
      const float scaled = static_cast<float>(row_values[col]) * inv_scale;
      qx[row_offset + col] = QuantizeScaledToInt8(scaled);
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> QuantizeActivationInt8RowwiseCuda(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& provided_scale) {
  TORCH_CHECK(x.is_cuda(), "QuantizeActivationInt8RowwiseCuda: x must be a CUDA tensor");
  TORCH_CHECK(IsSupportedInt8FrontendDtype(x.scalar_type()),
              "QuantizeActivationInt8RowwiseCuda: unsupported input dtype");
  TORCH_CHECK(x.dim() >= 2, "QuantizeActivationInt8RowwiseCuda: x must have rank >= 2");

  c10::cuda::CUDAGuard device_guard{x.device()};
  const auto cols = x.size(-1);
  const auto rows = x.numel() / cols;
  auto x_2d = x.reshape({rows, cols}).contiguous();
  auto qx_2d = torch::empty({rows, cols}, x.options().dtype(torch::kInt8));
  auto row_scale = torch::empty({rows}, x.options().dtype(torch::kFloat32));

  c10::optional<torch::Tensor> scale_cast = c10::nullopt;
  if (provided_scale.has_value() && provided_scale.value().defined()) {
    auto scale_value = provided_scale.value().to(x.device(), torch::kFloat32).reshape({-1}).contiguous();
    TORCH_CHECK(scale_value.numel() == 1 || scale_value.numel() == rows,
                "QuantizeActivationInt8RowwiseCuda: provided_scale must have 1 or rows elements");
    scale_cast = scale_value;
  }

  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_2d.scalar_type(),
      "model_stack_cuda_quantize_activation_int8_rowwise",
      [&] {
        if (Int8QuantWideRowEnabled(cols)) {
          const auto blocks = static_cast<unsigned int>((rows + kQuantWideRowsPerBlock - 1) / kQuantWideRowsPerBlock);
          if (scale_cast.has_value()) {
            quantize_activation_int8_rowwise_wide_provided_scale_kernel<scalar_t><<<
                blocks,
                kQuantThreads,
                0,
                stream.stream()>>>(
                x_2d.data_ptr<scalar_t>(),
                scale_cast.value().data_ptr<float>(),
                scale_cast.value().numel(),
                qx_2d.data_ptr<int8_t>(),
                row_scale.data_ptr<float>(),
                rows,
                cols);
          } else if (Int8QuantSharedCacheEnabled(cols, kQuantWideRowsPerBlock, sizeof(scalar_t))) {
            const auto shared_bytes =
                static_cast<size_t>(kQuantWideRowsPerBlock) * static_cast<size_t>(cols) * sizeof(scalar_t);
            quantize_activation_int8_rowwise_wide_cached_kernel<scalar_t><<<
                blocks,
                kQuantThreads,
                shared_bytes,
                stream.stream()>>>(
                x_2d.data_ptr<scalar_t>(),
                qx_2d.data_ptr<int8_t>(),
                row_scale.data_ptr<float>(),
                rows,
                cols);
          } else {
            quantize_activation_int8_rowwise_wide_kernel<scalar_t><<<
                blocks,
                kQuantThreads,
                0,
                stream.stream()>>>(
                x_2d.data_ptr<scalar_t>(),
                scale_cast.has_value() ? scale_cast.value().data_ptr<float>() : nullptr,
                scale_cast.has_value() ? scale_cast.value().numel() : 0,
                qx_2d.data_ptr<int8_t>(),
                row_scale.data_ptr<float>(),
                rows,
                cols);
          }
        } else if (Int8QuantWarpRowEnabled(cols)) {
          const auto launch_warp_rows = [&](auto rows_constant, int threads) {
            constexpr int rows_per_block = decltype(rows_constant)::value;
            const auto blocks = static_cast<unsigned int>((rows + rows_per_block - 1) / rows_per_block);
            if (scale_cast.has_value()) {
              if (Int8QuantVec4Enabled(cols)) {
                quantize_activation_int8_rowwise_warp_provided_scale_vec4_kernel<scalar_t, rows_per_block><<<
                    blocks,
                    threads,
                    0,
                    stream.stream()>>>(
                    x_2d.data_ptr<scalar_t>(),
                    scale_cast.value().data_ptr<float>(),
                    scale_cast.value().numel(),
                    qx_2d.data_ptr<int8_t>(),
                    row_scale.data_ptr<float>(),
                    rows,
                    cols);
              } else {
                quantize_activation_int8_rowwise_warp_provided_scale_kernel<scalar_t, rows_per_block><<<
                    blocks,
                    threads,
                    0,
                    stream.stream()>>>(
                    x_2d.data_ptr<scalar_t>(),
                    scale_cast.value().data_ptr<float>(),
                    scale_cast.value().numel(),
                    qx_2d.data_ptr<int8_t>(),
                    row_scale.data_ptr<float>(),
                    rows,
                    cols);
              }
            } else if (Int8QuantSharedCacheEnabled(cols, rows_per_block, sizeof(scalar_t))) {
              const auto shared_bytes =
                  static_cast<size_t>(rows_per_block) * static_cast<size_t>(cols) * sizeof(scalar_t);
              if (Int8QuantVec4Enabled(cols)) {
                quantize_activation_int8_rowwise_warp_cached_vec4_kernel<scalar_t, rows_per_block><<<
                    blocks,
                    threads,
                    shared_bytes,
                    stream.stream()>>>(
                    x_2d.data_ptr<scalar_t>(),
                    qx_2d.data_ptr<int8_t>(),
                    row_scale.data_ptr<float>(),
                    rows,
                    cols);
              } else {
                quantize_activation_int8_rowwise_warp_cached_kernel<scalar_t, rows_per_block><<<
                    blocks,
                    threads,
                    shared_bytes,
                    stream.stream()>>>(
                    x_2d.data_ptr<scalar_t>(),
                    qx_2d.data_ptr<int8_t>(),
                    row_scale.data_ptr<float>(),
                    rows,
                    cols);
              }
            } else {
              quantize_activation_int8_rowwise_warp_kernel<scalar_t, rows_per_block><<<
                  blocks,
                  threads,
                  0,
                  stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  scale_cast.has_value() ? scale_cast.value().data_ptr<float>() : nullptr,
                  scale_cast.has_value() ? scale_cast.value().numel() : 0,
                  qx_2d.data_ptr<int8_t>(),
                  row_scale.data_ptr<float>(),
                  rows,
                  cols);
            }
          };
          switch (Int8QuantWarpRowsPerBlock()) {
            case kQuantWarp4RowsPerBlock:
              launch_warp_rows(std::integral_constant<int, kQuantWarp4RowsPerBlock>{}, kQuantWarp4Threads);
              break;
            case kQuantWarp16RowsPerBlock:
              launch_warp_rows(std::integral_constant<int, kQuantWarp16RowsPerBlock>{}, kQuantWarp16Threads);
              break;
            default:
              launch_warp_rows(std::integral_constant<int, kQuantWarpRowsPerBlock>{}, kQuantWarpThreads);
              break;
          }
        } else {
          quantize_activation_int8_rowwise_kernel<scalar_t><<<
              static_cast<unsigned int>(rows),
              kQuantThreads,
              0,
              stream.stream()>>>(
              x_2d.data_ptr<scalar_t>(),
              scale_cast.has_value() ? scale_cast.value().data_ptr<float>() : nullptr,
              scale_cast.has_value() ? scale_cast.value().numel() : 0,
              qx_2d.data_ptr<int8_t>(),
              row_scale.data_ptr<float>(),
              rows,
              cols);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  return {qx_2d.view(out_sizes), row_scale};
}

std::tuple<torch::Tensor, torch::Tensor> QuantizeRelu2ActivationInt8RowwiseCuda(
    const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "QuantizeRelu2ActivationInt8RowwiseCuda: x must be a CUDA tensor");
  TORCH_CHECK(IsSupportedInt8FrontendDtype(x.scalar_type()),
              "QuantizeRelu2ActivationInt8RowwiseCuda: unsupported input dtype");
  TORCH_CHECK(x.dim() >= 2, "QuantizeRelu2ActivationInt8RowwiseCuda: x must have rank >= 2");

  c10::cuda::CUDAGuard device_guard{x.device()};
  const auto cols = x.size(-1);
  const auto rows = x.numel() / cols;
  auto x_2d = x.reshape({rows, cols}).contiguous();
  auto qx_2d = torch::empty({rows, cols}, x.options().dtype(torch::kInt8));
  auto row_scale = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_2d.scalar_type(),
      "model_stack_cuda_quantize_relu2_activation_int8_rowwise",
      [&] {
        if (Int8QuantWideRowEnabled(cols) &&
            Int8QuantSharedCacheEnabled(cols, kQuantWideRowsPerBlock, sizeof(scalar_t))) {
          const auto blocks =
              static_cast<unsigned int>((rows + kQuantWideRowsPerBlock - 1) / kQuantWideRowsPerBlock);
          const auto shared_bytes =
              static_cast<size_t>(kQuantWideRowsPerBlock) * static_cast<size_t>(cols) * sizeof(scalar_t);
          quantize_relu2_activation_int8_rowwise_wide_cached_kernel<scalar_t><<<
              blocks,
              kQuantThreads,
              shared_bytes,
              stream.stream()>>>(
              x_2d.data_ptr<scalar_t>(),
              qx_2d.data_ptr<int8_t>(),
              row_scale.data_ptr<float>(),
              rows,
              cols);
        } else if (Int8QuantWarpRowEnabled(cols)) {
          const auto launch_warp_rows = [&](auto rows_constant, int threads) {
            constexpr int rows_per_block = decltype(rows_constant)::value;
            const auto blocks = static_cast<unsigned int>((rows + rows_per_block - 1) / rows_per_block);
            if (!Int8QuantSharedCacheEnabled(cols, rows_per_block, sizeof(scalar_t))) {
              quantize_relu2_activation_int8_rowwise_kernel<scalar_t><<<
                  static_cast<unsigned int>(rows),
                  kQuantThreads,
                  0,
                  stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  qx_2d.data_ptr<int8_t>(),
                  row_scale.data_ptr<float>(),
                  rows,
                  cols);
              return;
            }
            const auto shared_bytes =
                static_cast<size_t>(rows_per_block) * static_cast<size_t>(cols) * sizeof(scalar_t);
            quantize_relu2_activation_int8_rowwise_warp_cached_kernel<scalar_t, rows_per_block><<<
                blocks,
                threads,
                shared_bytes,
                stream.stream()>>>(
                x_2d.data_ptr<scalar_t>(),
                qx_2d.data_ptr<int8_t>(),
                row_scale.data_ptr<float>(),
                rows,
                cols);
          };
          switch (Int8QuantWarpRowsPerBlock()) {
            case kQuantWarp4RowsPerBlock:
              launch_warp_rows(std::integral_constant<int, kQuantWarp4RowsPerBlock>{}, kQuantWarp4Threads);
              break;
            case kQuantWarp16RowsPerBlock:
              launch_warp_rows(std::integral_constant<int, kQuantWarp16RowsPerBlock>{}, kQuantWarp16Threads);
              break;
            default:
              launch_warp_rows(std::integral_constant<int, kQuantWarpRowsPerBlock>{}, kQuantWarpThreads);
              break;
          }
        } else {
          quantize_relu2_activation_int8_rowwise_kernel<scalar_t><<<
              static_cast<unsigned int>(rows),
              kQuantThreads,
              0,
              stream.stream()>>>(
              x_2d.data_ptr<scalar_t>(),
              qx_2d.data_ptr<int8_t>(),
              row_scale.data_ptr<float>(),
              rows,
              cols);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  return {qx_2d.view(out_sizes), row_scale};
}

}  // namespace

bool HasCudaInt8QuantFrontendKernel() {
  return true;
}

std::vector<torch::Tensor> CudaInt8QuantizeActivationForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& provided_scale) {
  auto quantized = QuantizeActivationInt8RowwiseCuda(x, provided_scale);
  return {std::get<0>(quantized), std::get<1>(quantized)};
}

std::vector<torch::Tensor> CudaInt8QuantizeRelu2ActivationForward(
    const torch::Tensor& x) {
  auto quantized = QuantizeRelu2ActivationInt8RowwiseCuda(x);
  return {std::get<0>(quantized), std::get<1>(quantized)};
}

torch::Tensor CudaInt8LinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& provided_scale,
    const c10::optional<torch::ScalarType>& out_dtype) {
  auto quantized = QuantizeActivationInt8RowwiseCuda(x, provided_scale);
  return CudaInt8LinearForward(std::get<0>(quantized), std::get<1>(quantized), qweight, inv_scale, bias, out_dtype);
}

torch::Tensor CudaInt8AttentionFromFloatForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    const c10::optional<torch::ScalarType>& out_dtype,
    const c10::optional<torch::Tensor>& q_provided_scale,
    const c10::optional<torch::Tensor>& k_provided_scale,
    const c10::optional<torch::Tensor>& v_provided_scale) {
  auto qq = QuantizeActivationInt8RowwiseCuda(q, q_provided_scale);
  auto kk = QuantizeActivationInt8RowwiseCuda(k, k_provided_scale);
  auto vv = QuantizeActivationInt8RowwiseCuda(v, v_provided_scale);
  return CudaInt8AttentionForward(
      std::get<0>(qq),
      std::get<1>(qq),
      std::get<0>(kk),
      std::get<1>(kk),
      std::get<0>(vv),
      std::get<1>(vv),
      attn_mask,
      is_causal,
      scale,
      out_dtype);
}
