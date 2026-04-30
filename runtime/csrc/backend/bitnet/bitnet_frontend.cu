#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "bitnet_common.cuh"
#include "bitnet_epilogue.cuh"

namespace t10::bitnet {
namespace {

constexpr int kBitNetTransformThreads = 256;
constexpr int kBitNetRow1Threads = 32;
constexpr int kBitNetRow1ColsPerThread = 8;

struct CachedBitNetRow1QuantScratch {
  int device_index = -1;
  std::uintptr_t stream_key = 0;
  int64_t cols = 0;
  torch::Tensor qx_2d;
  torch::Tensor row_scale;

  bool Matches(int device, std::uintptr_t stream, int64_t width) const {
    return device_index == device && stream_key == stream && cols == width &&
        qx_2d.defined() && row_scale.defined();
  }
};

std::pair<torch::Tensor, torch::Tensor> GetOrCreateBitNetRow1QuantScratch(
    const torch::Tensor& reference,
    int64_t cols,
    cudaStream_t stream) {
  thread_local std::vector<std::unique_ptr<CachedBitNetRow1QuantScratch>> cache;
  const auto device_index = reference.get_device();
  const auto stream_key = reinterpret_cast<std::uintptr_t>(stream);
  for (auto& entry : cache) {
    if (entry != nullptr && entry->Matches(device_index, stream_key, cols)) {
      return {entry->qx_2d, entry->row_scale};
    }
  }

  auto entry = std::make_unique<CachedBitNetRow1QuantScratch>();
  entry->device_index = device_index;
  entry->stream_key = stream_key;
  entry->cols = cols;
  entry->qx_2d = torch::empty({1, cols}, reference.options().dtype(torch::kInt8));
  entry->row_scale = torch::empty({1}, torch::TensorOptions().device(reference.device()).dtype(torch::kFloat32));
  auto scratch = std::make_pair(entry->qx_2d, entry->row_scale);
  cache.push_back(std::move(entry));
  return scratch;
}

enum class GatedActivationKind : int {
  kSiLU = 0,
  kGELU = 1,
  kReLU = 2,
  kLeakyRelu0p5Squared = 3,
  kReLU2 = 4,
};

GatedActivationKind ParseGatedActivation(const std::string& activation) {
  if (activation == "swiglu" || activation == "gated-silu" || activation == "silu" || activation == "swish") {
    return GatedActivationKind::kSiLU;
  }
  if (activation == "geglu" || activation == "gelu") {
    return GatedActivationKind::kGELU;
  }
  if (activation == "reglu" || activation == "relu") {
    return GatedActivationKind::kReLU;
  }
  if (activation == "relu2" || activation == "squared_relu" || activation == "squared-relu") {
    return GatedActivationKind::kReLU2;
  }
  if (activation == "leaky_relu_0p5_squared" || activation == "leaky-relu-0p5-squared" ||
      activation == "leaky_relu2" || activation == "leaky-relu2" ||
      activation == "leaky_relu_0.5_squared" || activation == "leaky-relu-0.5-squared") {
    return GatedActivationKind::kLeakyRelu0p5Squared;
  }
  return GatedActivationKind::kSiLU;
}

__device__ inline float GatedActivationGeluExact(float x) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ inline float ApplyGatedActivationValue(float x, GatedActivationKind activation) {
  switch (activation) {
    case GatedActivationKind::kSiLU:
      return x / (1.0f + expf(-x));
    case GatedActivationKind::kGELU:
      return GatedActivationGeluExact(x);
    case GatedActivationKind::kReLU:
      return x > 0.0f ? x : 0.0f;
    case GatedActivationKind::kReLU2: {
      const float y = x > 0.0f ? x : 0.0f;
      return y * y;
    }
    case GatedActivationKind::kLeakyRelu0p5Squared: {
      const float y = x > 0.0f ? x : (0.5f * x);
      return y * y;
    }
  }
  return x;
}

template <typename scalar_t>
__device__ inline float BitNetGatedInputValueAfterPreScaleDevice(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    int64_t row,
    int64_t col,
    int64_t hidden,
    GatedActivationKind activation) {
  const int64_t base = row * (hidden * 2);
  float value = ApplyGatedActivationValue(static_cast<float>(x[base + col]), activation) *
      static_cast<float>(x[base + hidden + col]);
  if (pre_scale != nullptr) {
    value /= static_cast<float>(pre_scale[col]);
  }
  return value;
}

template <typename scalar_t>
__global__ void bitnet_transform_input_nospin_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols,
    int qmax,
    bool apply_quant) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  if (!apply_quant) {
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
      float value = BitNetInputValueAfterPreScaleDevice(x, pre_scale, row * cols + col, col);
      out[row * cols + col] = CastOutput<scalar_t>(value);
    }
    return;
  }

  __shared__ float shared_max[kBitNetTransformThreads];
  float local_max = 0.0f;
  if (provided_scale == nullptr) {
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
      float value = BitNetInputValueAfterPreScaleDevice(x, pre_scale, row * cols + col, col);
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  if (provided_scale == nullptr) {
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
      }
      __syncthreads();
    }
  }

  float scale = 1.0f;
  if (provided_scale != nullptr) {
    scale = fmaxf(provided_scale[provided_rows == 1 ? 0 : row], 1.0e-8f);
  } else {
    scale = fmaxf(shared_max[0], 1.0e-8f) / static_cast<float>(qmax);
  }

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = BitNetInputValueAfterPreScaleDevice(x, pre_scale, row * cols + col, col);
    const float scaled = value / scale;
    const float rounded = nearbyintf(scaled);
    const float clamped = fminf(static_cast<float>(qmax), fmaxf(-static_cast<float>(qmax), rounded));
    out[row * cols + col] = CastOutput<scalar_t>(clamped * scale);
  }
}

template <typename scalar_t>
__global__ void bitnet_calibrate_input_row_max_nospin_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    float* __restrict__ row_max,
    int64_t rows,
    int64_t cols) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float shared_max[kBitNetTransformThreads];
  float local_max = 0.0f;
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = BitNetInputValueAfterPreScaleDevice(x, pre_scale, row * cols + col, col);
    local_max = fmaxf(local_max, fabsf(value));
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    row_max[row] = shared_max[0];
  }
}

template <typename scalar_t>
__global__ void bitnet_quantize_input_int8_codes_nospin_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols,
    int qmax) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float shared_max[kBitNetTransformThreads];
  float local_max = 0.0f;
  if (provided_scale == nullptr) {
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
      const float value = BitNetInputValueAfterPreScaleDevice(x, pre_scale, row * cols + col, col);
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  if (provided_scale == nullptr) {
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
      }
      __syncthreads();
    }
  }

  float scale = 1.0f;
  if (provided_scale != nullptr) {
    scale = fmaxf(provided_scale[provided_rows == 1 ? 0 : row], 1.0e-8f);
  } else {
    scale = fmaxf(shared_max[0], 1.0e-8f) / static_cast<float>(qmax);
  }

  if (threadIdx.x == 0) {
    row_scale[row] = scale;
  }
  __syncthreads();

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = BitNetInputValueAfterPreScaleDevice(x, pre_scale, row * cols + col, col);
    const float scaled = value / scale;
    const float rounded = nearbyintf(scaled);
    const float clamped = fminf(static_cast<float>(qmax), fmaxf(-static_cast<float>(qmax), rounded));
    qx[row * cols + col] = static_cast<int8_t>(static_cast<int>(clamped));
  }
}

template <typename scalar_t>
__global__ void bitnet_quantize_gated_activation_int8_codes_nospin_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    int8_t* __restrict__ qx,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t hidden,
    int qmax,
    GatedActivationKind activation) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float shared_max[kBitNetTransformThreads];
  float local_max = 0.0f;
  if (provided_scale == nullptr) {
    for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
      const float value =
          BitNetGatedInputValueAfterPreScaleDevice(x, pre_scale, row, col, hidden, activation);
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  if (provided_scale == nullptr) {
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
      }
      __syncthreads();
    }
  }

  float scale = 1.0f;
  if (provided_scale != nullptr) {
    scale = fmaxf(provided_scale[provided_rows == 1 ? 0 : row], 1.0e-8f);
  } else {
    scale = fmaxf(shared_max[0], 1.0e-8f) / static_cast<float>(qmax);
  }

  if (threadIdx.x == 0) {
    row_scale[row] = scale;
  }
  __syncthreads();

  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float value =
        BitNetGatedInputValueAfterPreScaleDevice(x, pre_scale, row, col, hidden, activation);
    const float scaled = value / scale;
    const float rounded = nearbyintf(scaled);
    const float clamped = fminf(static_cast<float>(qmax), fmaxf(-static_cast<float>(qmax), rounded));
    qx[row * hidden + col] = static_cast<int8_t>(static_cast<int>(clamped));
  }
}

template <typename scalar_t, int TileK, int ColsPerThread>
__global__ __launch_bounds__(kBitNetRow1Threads, 8) void bitnet_int8_linear_from_float_row1_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pre_scale,
    const float* __restrict__ provided_scale,
    int64_t provided_rows,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t out_features,
    int64_t in_features,
    int qmax) {
  __shared__ float shared_max[kBitNetRow1Threads];
  __shared__ int8_t x_tile[TileK];

  const int lane = static_cast<int>(threadIdx.x);
  float local_max = 0.0f;
  if (provided_scale == nullptr) {
    for (int64_t k = lane; k < in_features; k += kBitNetRow1Threads) {
      const float value = BitNetInputValueAfterPreScaleDevice(x, pre_scale, k, k);
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  shared_max[lane] = local_max;
  __syncthreads();

  if (provided_scale == nullptr) {
    for (int stride = kBitNetRow1Threads / 2; stride > 0; stride >>= 1) {
      if (lane < stride) {
        shared_max[lane] = fmaxf(shared_max[lane], shared_max[lane + stride]);
      }
      __syncthreads();
    }
  }

  const float scale = provided_scale != nullptr
      ? fmaxf(provided_scale[provided_rows == 1 ? 0 : 0], 1.0e-8f)
      : (fmaxf(shared_max[0], 1.0e-8f) / static_cast<float>(qmax));

  constexpr int TileCols = kBitNetRow1Threads * ColsPerThread;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * TileCols + lane;
  int accum[ColsPerThread];
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    accum[idx] = 0;
  }

  for (int64_t k0 = 0; k0 < in_features; k0 += TileK) {
    for (int tile_k = lane; tile_k < TileK; tile_k += kBitNetRow1Threads) {
      const int64_t global_k = k0 + tile_k;
      int8_t qvalue = 0;
      if (global_k < in_features) {
        const float value = BitNetInputValueAfterPreScaleDevice(x, pre_scale, global_k, global_k);
        qvalue = BitNetQuantizeStaticInputCodeDevice(value, scale, qmax);
      }
      x_tile[tile_k] = qvalue;
    }
    __syncthreads();

    #pragma unroll
    for (int idx = 0; idx < ColsPerThread; ++idx) {
      const int64_t global_col = col0 + static_cast<int64_t>(idx) * kBitNetRow1Threads;
      if (global_col >= out_features) {
        continue;
      }
      const int8_t* w_row = qweight + (global_col * in_features) + k0;
      int acc = accum[idx];
      int kk = 0;
      for (; kk + 3 < TileK && k0 + kk + 3 < in_features; kk += 4) {
        acc = BitNetDotInt8Chunk4Device(&x_tile[kk], &w_row[kk], acc);
      }
      for (; kk < TileK && k0 + kk < in_features; ++kk) {
        acc += static_cast<int>(x_tile[kk]) * static_cast<int>(w_row[kk]);
      }
      accum[idx] = acc;
    }
    __syncthreads();
  }

  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t global_col = col0 + static_cast<int64_t>(idx) * kBitNetRow1Threads;
    if (global_col >= out_features) {
      continue;
    }
    float value = static_cast<float>(accum[idx]) * scale * inv_scale[global_col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[global_col]);
    }
    out[global_col] = CastOutput<scalar_t>(value);
  }
}

}  // namespace

bool HasCudaBitNetInputFrontendKernel() {
  return true;
}

torch::Tensor CudaBitNetCalibrateInputScaleForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    int64_t act_quant_bits) {
  TORCH_CHECK(x.is_cuda(), "CudaBitNetCalibrateInputScaleForward: x must be a CUDA tensor");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetCalibrateInputScaleForward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(x.dim() >= 1, "CudaBitNetCalibrateInputScaleForward: x must have rank >= 1");
  TORCH_CHECK(act_quant_bits >= 2, "CudaBitNetCalibrateInputScaleForward: act_quant_bits must be >= 2");

  c10::cuda::CUDAGuard device_guard{x.device()};
  const auto cols = x.size(-1);
  const auto rows = cols > 0 ? x.numel() / cols : 0;
  auto x_2d = x.reshape({rows, cols}).contiguous();

  c10::optional<torch::Tensor> pre_scale_cast = c10::nullopt;
  if (pre_scale.has_value() && pre_scale.value().defined() && pre_scale.value().numel() > 0) {
    auto scale = pre_scale.value().to(x.device(), x.scalar_type()).reshape({-1}).contiguous();
    TORCH_CHECK(scale.numel() == cols,
                "CudaBitNetCalibrateInputScaleForward: pre_scale must match x last dimension");
    pre_scale_cast = scale;
  }

  if (rows == 0 || cols == 0) {
    return torch::full(
        {1},
        1.0e-8f / static_cast<float>(BitNetQuantMax(act_quant_bits)),
        torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
  }

  auto row_max = torch::empty({rows}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_2d.scalar_type(),
      "model_stack_cuda_bitnet_calibrate_input_scale",
      [&] {
        bitnet_calibrate_input_row_max_nospin_kernel<scalar_t><<<
            static_cast<unsigned int>(rows),
            kBitNetTransformThreads,
            0,
            stream.stream()>>>(
            x_2d.data_ptr<scalar_t>(),
            pre_scale_cast.has_value() ? pre_scale_cast.value().data_ptr<scalar_t>() : nullptr,
            row_max.data_ptr<float>(),
            rows,
            cols);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const float qmax = static_cast<float>(BitNetQuantMax(act_quant_bits));
  return (row_max.clamp_min(1.0e-8f) / qmax).to(torch::kFloat32);
}

torch::Tensor CudaBitNetTransformInputForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale) {
  TORCH_CHECK(x.is_cuda(), "CudaBitNetTransformInputForward: x must be a CUDA tensor");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetTransformInputForward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(x.dim() >= 1, "CudaBitNetTransformInputForward: x must have rank >= 1");

  c10::cuda::CUDAGuard device_guard{x.device()};
  const auto cols = x.size(-1);
  const auto rows = cols > 0 ? x.numel() / cols : 0;
  auto x_2d = x.reshape({rows, cols}).contiguous();
  auto out_2d = torch::empty_like(x_2d);

  c10::optional<torch::Tensor> pre_scale_cast = c10::nullopt;
  if (pre_scale.has_value() && pre_scale.value().defined() && pre_scale.value().numel() > 0) {
    auto scale = pre_scale.value().to(x.device(), x.scalar_type()).reshape({-1}).contiguous();
    TORCH_CHECK(scale.numel() == cols,
                "CudaBitNetTransformInputForward: pre_scale must match x last dimension");
    pre_scale_cast = scale;
  }

  const auto mode_name = act_quant_mode;
  const bool apply_quant = !(mode_name.empty() || mode_name == "none" || mode_name == "off");
  c10::optional<torch::Tensor> provided_scale = c10::nullopt;
  if (apply_quant) {
    TORCH_CHECK(act_quant_bits >= 2, "CudaBitNetTransformInputForward: act_quant_bits must be >= 2");
    if (mode_name == "static_int8") {
      TORCH_CHECK(act_scale.has_value() && act_scale.value().defined(),
                  "CudaBitNetTransformInputForward: static_int8 requires act_scale");
      auto scale = act_scale.value().to(x.device(), torch::kFloat32).reshape({-1}).contiguous();
      TORCH_CHECK(scale.numel() == 1 || scale.numel() == rows,
                  "CudaBitNetTransformInputForward: act_scale must have 1 or rows elements");
      provided_scale = scale;
    } else {
      TORCH_CHECK(mode_name == "dynamic_int8",
                  "CudaBitNetTransformInputForward: unsupported activation quant mode");
      TORCH_CHECK(act_quant_method == "absmax" || act_quant_method == "mse" || act_quant_method.empty(),
                  "CudaBitNetTransformInputForward: unsupported dynamic activation calibration method");
      // Leave provided_scale unset for dynamic quantization. The transform kernel computes
      // one row-local scale per token row, avoiding a separate calibration launch.
    }
  }

  if (rows == 0 || cols == 0) {
    return out_2d.view(x.sizes());
  }

  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const int qmax = apply_quant ? BitNetQuantMax(act_quant_bits) : 1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_2d.scalar_type(),
      "model_stack_cuda_bitnet_transform_input",
      [&] {
        bitnet_transform_input_nospin_kernel<scalar_t><<<
            static_cast<unsigned int>(rows),
            kBitNetTransformThreads,
            0,
            stream.stream()>>>(
            x_2d.data_ptr<scalar_t>(),
            pre_scale_cast.has_value() ? pre_scale_cast.value().data_ptr<scalar_t>() : nullptr,
            provided_scale.has_value() ? provided_scale.value().data_ptr<float>() : nullptr,
            provided_scale.has_value() ? provided_scale.value().numel() : 0,
            out_2d.data_ptr<scalar_t>(),
            rows,
            cols,
            qmax,
            apply_quant);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out_2d.view(x.sizes());
}

std::tuple<torch::Tensor, torch::Tensor> CudaBitNetQuantizeActivationInt8CodesForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale) {
  TORCH_CHECK(x.is_cuda(), "CudaBitNetQuantizeActivationInt8CodesForward: x must be a CUDA tensor");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetQuantizeActivationInt8CodesForward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(x.dim() >= 2, "CudaBitNetQuantizeActivationInt8CodesForward: x must have rank >= 2");
  TORCH_CHECK(
      act_quant_bits >= 2 && act_quant_bits <= 8,
      "CudaBitNetQuantizeActivationInt8CodesForward: act_quant_bits must be in [2, 8]");

  c10::cuda::CUDAGuard device_guard{x.device()};
  const auto cols = x.size(-1);
  const auto rows = cols > 0 ? x.numel() / cols : 0;
  auto x_2d = x.reshape({rows, cols}).contiguous();
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  auto scratch = rows == 1
      ? GetOrCreateBitNetRow1QuantScratch(x, cols, stream.stream())
      : std::make_pair(
            torch::empty({rows, cols}, x.options().dtype(torch::kInt8)),
            torch::empty({rows}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)));
  auto qx_2d = scratch.first;
  auto row_scale = scratch.second;

  c10::optional<torch::Tensor> pre_scale_cast = c10::nullopt;
  if (pre_scale.has_value() && pre_scale.value().defined() && pre_scale.value().numel() > 0) {
    auto scale = pre_scale.value().to(x.device(), x.scalar_type()).reshape({-1}).contiguous();
    TORCH_CHECK(
        scale.numel() == cols,
        "CudaBitNetQuantizeActivationInt8CodesForward: pre_scale must match x last dimension");
    pre_scale_cast = scale;
  }

  const auto mode_name = act_quant_mode;
  TORCH_CHECK(
      mode_name == "dynamic_int8" || mode_name == "static_int8",
      "CudaBitNetQuantizeActivationInt8CodesForward: unsupported activation quant mode");
  c10::optional<torch::Tensor> provided_scale = c10::nullopt;
  if (mode_name == "static_int8") {
    TORCH_CHECK(
        act_scale.has_value() && act_scale.value().defined(),
        "CudaBitNetQuantizeActivationInt8CodesForward: static_int8 requires act_scale");
    auto scale = act_scale.value().to(x.device(), torch::kFloat32).reshape({-1}).contiguous();
    TORCH_CHECK(
        scale.numel() == 1 || scale.numel() == rows,
        "CudaBitNetQuantizeActivationInt8CodesForward: act_scale must have 1 or rows elements");
    provided_scale = scale;
  } else {
    TORCH_CHECK(
        act_quant_method == "absmax" || act_quant_method == "mse" || act_quant_method.empty(),
        "CudaBitNetQuantizeActivationInt8CodesForward: unsupported dynamic activation calibration method");
    // Leave provided_scale unset for dynamic quantization. The quantize kernel computes
    // one row-local scale per token row, avoiding a separate calibration launch.
  }

  if (rows == 0 || cols == 0) {
    std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
    return {qx_2d.view(out_sizes), row_scale};
  }

  const int qmax = BitNetQuantMax(act_quant_bits);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_2d.scalar_type(),
      "model_stack_cuda_bitnet_quantize_activation_int8_codes",
      [&] {
        bitnet_quantize_input_int8_codes_nospin_kernel<scalar_t><<<
            static_cast<unsigned int>(rows),
            kBitNetTransformThreads,
            0,
            stream.stream()>>>(
            x_2d.data_ptr<scalar_t>(),
            pre_scale_cast.has_value() ? pre_scale_cast.value().data_ptr<scalar_t>() : nullptr,
            provided_scale.has_value() ? provided_scale.value().data_ptr<float>() : nullptr,
            provided_scale.has_value() ? provided_scale.value().numel() : 0,
            qx_2d.data_ptr<int8_t>(),
            row_scale.data_ptr<float>(),
            rows,
            cols,
            qmax);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  return {qx_2d.view(out_sizes), row_scale};
}

std::tuple<torch::Tensor, torch::Tensor> CudaBitNetQuantizeGatedActivationInt8CodesForward(
    const torch::Tensor& x,
    const std::string& activation,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale) {
  TORCH_CHECK(x.is_cuda(), "CudaBitNetQuantizeGatedActivationInt8CodesForward: x must be a CUDA tensor");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetQuantizeGatedActivationInt8CodesForward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(x.dim() >= 2, "CudaBitNetQuantizeGatedActivationInt8CodesForward: x must have rank >= 2");
  TORCH_CHECK(
      x.size(-1) % 2 == 0,
      "CudaBitNetQuantizeGatedActivationInt8CodesForward: x last dimension must be even");
  TORCH_CHECK(
      act_quant_bits >= 2 && act_quant_bits <= 8,
      "CudaBitNetQuantizeGatedActivationInt8CodesForward: act_quant_bits must be in [2, 8]");

  c10::cuda::CUDAGuard device_guard{x.device()};
  const auto cols = x.size(-1);
  const auto hidden = cols / 2;
  const auto rows = cols > 0 ? x.numel() / cols : 0;
  auto x_2d = x.reshape({rows, cols}).contiguous();
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  auto scratch = rows == 1
      ? GetOrCreateBitNetRow1QuantScratch(x, hidden, stream.stream())
      : std::make_pair(
            torch::empty({rows, hidden}, x.options().dtype(torch::kInt8)),
            torch::empty({rows}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)));
  auto qx_2d = scratch.first;
  auto row_scale = scratch.second;

  c10::optional<torch::Tensor> pre_scale_cast = c10::nullopt;
  if (pre_scale.has_value() && pre_scale.value().defined() && pre_scale.value().numel() > 0) {
    auto scale = pre_scale.value().to(x.device(), x.scalar_type()).reshape({-1}).contiguous();
    TORCH_CHECK(
        scale.numel() == hidden,
        "CudaBitNetQuantizeGatedActivationInt8CodesForward: pre_scale must match gated hidden dimension");
    pre_scale_cast = scale;
  }

  const auto mode_name = act_quant_mode;
  TORCH_CHECK(
      mode_name == "dynamic_int8" || mode_name == "static_int8",
      "CudaBitNetQuantizeGatedActivationInt8CodesForward: unsupported activation quant mode");
  c10::optional<torch::Tensor> provided_scale = c10::nullopt;
  if (mode_name == "static_int8") {
    TORCH_CHECK(
        act_scale.has_value() && act_scale.value().defined(),
        "CudaBitNetQuantizeGatedActivationInt8CodesForward: static_int8 requires act_scale");
    auto scale = act_scale.value().to(x.device(), torch::kFloat32).reshape({-1}).contiguous();
    TORCH_CHECK(
        scale.numel() == 1 || scale.numel() == rows,
        "CudaBitNetQuantizeGatedActivationInt8CodesForward: act_scale must have 1 or rows elements");
    provided_scale = scale;
  } else {
    TORCH_CHECK(
        act_quant_method == "absmax" || act_quant_method == "mse" || act_quant_method.empty(),
        "CudaBitNetQuantizeGatedActivationInt8CodesForward: unsupported dynamic activation calibration method");
    TORCH_CHECK(
        rows > 0 && rows <= 8,
        "CudaBitNetQuantizeGatedActivationInt8CodesForward: fused dynamic gating path only supports rows in [1, 8]");
  }

  if (rows == 0 || hidden == 0) {
    std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
    out_sizes.back() = hidden;
    return {qx_2d.view(out_sizes), row_scale};
  }

  const int qmax = BitNetQuantMax(act_quant_bits);
  const auto activation_kind = ParseGatedActivation(activation);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_2d.scalar_type(),
      "model_stack_cuda_bitnet_quantize_gated_activation_int8_codes",
      [&] {
        bitnet_quantize_gated_activation_int8_codes_nospin_kernel<scalar_t><<<
            static_cast<unsigned int>(rows),
            kBitNetTransformThreads,
            0,
            stream.stream()>>>(
            x_2d.data_ptr<scalar_t>(),
            pre_scale_cast.has_value() ? pre_scale_cast.value().data_ptr<scalar_t>() : nullptr,
            provided_scale.has_value() ? provided_scale.value().data_ptr<float>() : nullptr,
            provided_scale.has_value() ? provided_scale.value().numel() : 0,
            qx_2d.data_ptr<int8_t>(),
            row_scale.data_ptr<float>(),
            rows,
            hidden,
            qmax,
            activation_kind);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = hidden;
  return {qx_2d.view(out_sizes), row_scale};
}

torch::Tensor CudaBitNetInt8LinearFromFloatRow1Forward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  // This entry is kept as a direct CUDA helper, but the production native decode
  // path does not auto-select it on Ampere. The split BitNet quantize + int8
  // linear route is faster end to end for the current small-GEMM backend.
  TORCH_CHECK(x.is_cuda(), "CudaBitNetInt8LinearFromFloatRow1Forward: x must be a CUDA tensor");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetInt8LinearFromFloatRow1Forward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(x.dim() >= 2, "CudaBitNetInt8LinearFromFloatRow1Forward: x must have rank >= 2");
  TORCH_CHECK(qweight.is_cuda() && inv_scale.is_cuda(),
              "CudaBitNetInt8LinearFromFloatRow1Forward: qweight and inv_scale must be CUDA tensors");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt8,
              "CudaBitNetInt8LinearFromFloatRow1Forward: qweight must use int8 storage");
  TORCH_CHECK(inv_scale.scalar_type() == torch::kFloat32,
              "CudaBitNetInt8LinearFromFloatRow1Forward: inv_scale must use float32 storage");
  TORCH_CHECK(act_quant_bits >= 2 && act_quant_bits <= 8,
              "CudaBitNetInt8LinearFromFloatRow1Forward: act_quant_bits must be in [2, 8]");

  c10::cuda::CUDAGuard device_guard{x.device()};
  const auto cols = x.size(-1);
  const auto rows = cols > 0 ? x.numel() / cols : 0;
  TORCH_CHECK(rows == 1, "CudaBitNetInt8LinearFromFloatRow1Forward: only supports rows == 1");
  auto x_2d = x.reshape({rows, cols}).contiguous();
  auto qweight_contig = qweight.contiguous();
  auto inv_scale_contig = inv_scale.contiguous();

  const auto output_dtype = out_dtype.has_value() ? out_dtype.value() : x.scalar_type();
  TORCH_CHECK(IsSupportedLinearDtype(output_dtype),
              "CudaBitNetInt8LinearFromFloatRow1Forward: unsupported output dtype");
  TORCH_CHECK(
      output_dtype == x.scalar_type(),
      "CudaBitNetInt8LinearFromFloatRow1Forward: output dtype must match x dtype");

  c10::optional<torch::Tensor> pre_scale_cast = c10::nullopt;
  if (pre_scale.has_value() && pre_scale.value().defined() && pre_scale.value().numel() > 0) {
    auto scale = pre_scale.value().to(x.device(), x.scalar_type()).reshape({-1}).contiguous();
    TORCH_CHECK(
        scale.numel() == cols,
        "CudaBitNetInt8LinearFromFloatRow1Forward: pre_scale must match x last dimension");
    pre_scale_cast = scale;
  }

  c10::optional<torch::Tensor> provided_scale = c10::nullopt;
  const auto mode_name = act_quant_mode;
  if (mode_name == "static_int8") {
    TORCH_CHECK(
        act_scale.has_value() && act_scale.value().defined(),
        "CudaBitNetInt8LinearFromFloatRow1Forward: static_int8 requires act_scale");
    auto scale = act_scale.value().to(x.device(), torch::kFloat32).reshape({-1}).contiguous();
    TORCH_CHECK(
        scale.numel() == 1 || scale.numel() == rows,
        "CudaBitNetInt8LinearFromFloatRow1Forward: act_scale must have 1 or rows elements");
    provided_scale = scale;
  } else {
    TORCH_CHECK(
        mode_name == "dynamic_int8",
        "CudaBitNetInt8LinearFromFloatRow1Forward: unsupported activation quant mode");
    TORCH_CHECK(
        act_quant_method.empty() || act_quant_method == "absmax",
        "CudaBitNetInt8LinearFromFloatRow1Forward: dynamic row1 frontend supports absmax only");
  }

  auto out_2d = torch::empty({1, qweight_contig.size(0)}, x.options().dtype(output_dtype));
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    auto bias_value = bias.value().to(x.device(), output_dtype).contiguous();
    TORCH_CHECK(
        bias_value.numel() == qweight_contig.size(0),
        "CudaBitNetInt8LinearFromFloatRow1Forward: bias size mismatch");
    bias_cast = bias_value;
  }

  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const int qmax = BitNetQuantMax(act_quant_bits);
  const dim3 row1_blocks(
      static_cast<unsigned int>(
          (qweight_contig.size(0) + (kBitNetRow1Threads * kBitNetRow1ColsPerThread) - 1) /
          (kBitNetRow1Threads * kBitNetRow1ColsPerThread)));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output_dtype,
      "model_stack_cuda_bitnet_int8_linear_from_float_row1",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias_cast.has_value() && bias_cast.value().defined()) {
          bias_ptr = bias_cast.value().data_ptr<scalar_t>();
        }
        if (cols >= kPrefillSm80TileK && DeviceIsSm80OrLater(x_2d)) {
          bitnet_int8_linear_from_float_row1_kernel<scalar_t, kPrefillSm80TileK, kBitNetRow1ColsPerThread>
              <<<row1_blocks, kBitNetRow1Threads, 0, stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  pre_scale_cast.has_value() ? pre_scale_cast.value().data_ptr<scalar_t>() : nullptr,
                  provided_scale.has_value() ? provided_scale.value().data_ptr<float>() : nullptr,
                  provided_scale.has_value() ? provided_scale.value().numel() : 0,
                  qweight_contig.data_ptr<int8_t>(),
                  inv_scale_contig.data_ptr<float>(),
                  bias_ptr,
                  out_2d.data_ptr<scalar_t>(),
                  qweight_contig.size(0),
                  cols,
                  qmax);
        } else {
          bitnet_int8_linear_from_float_row1_kernel<scalar_t, kPrefillGenericTileK, kBitNetRow1ColsPerThread>
              <<<row1_blocks, kBitNetRow1Threads, 0, stream.stream()>>>(
                  x_2d.data_ptr<scalar_t>(),
                  pre_scale_cast.has_value() ? pre_scale_cast.value().data_ptr<scalar_t>() : nullptr,
                  provided_scale.has_value() ? provided_scale.value().data_ptr<float>() : nullptr,
                  provided_scale.has_value() ? provided_scale.value().numel() : 0,
                  qweight_contig.data_ptr<int8_t>(),
                  inv_scale_contig.data_ptr<float>(),
                  bias_ptr,
                  out_2d.data_ptr<scalar_t>(),
                  qweight_contig.size(0),
                  cols,
                  qmax);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = qweight_contig.size(0);
  return out_2d.view(out_sizes);
}

}  // namespace t10::bitnet
