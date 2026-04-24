#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cuda_runtime.h>
#include <string>

namespace {

constexpr int kThreads = 256;

enum class GatedActivationKind : int {
  kSiLU = 0,
  kGELU = 1,
  kReLU = 2,
  kLeakyRelu0p5Squared = 3,
};

GatedActivationKind ParseActivation(const std::string& activation) {
  if (activation == "swiglu" || activation == "gated-silu" || activation == "silu" || activation == "swish") {
    return GatedActivationKind::kSiLU;
  }
  if (activation == "geglu" || activation == "gelu") {
    return GatedActivationKind::kGELU;
  }
  if (activation == "reglu" || activation == "relu") {
    return GatedActivationKind::kReLU;
  }
  if (activation == "leaky_relu_0p5_squared" || activation == "leaky-relu-0p5-squared" ||
      activation == "leaky_relu_0.5_squared" || activation == "leaky-relu-0.5-squared") {
    return GatedActivationKind::kLeakyRelu0p5Squared;
  }
  return GatedActivationKind::kSiLU;
}

__device__ inline float GeluExact(float x) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ inline float ApplyActivation(float x, GatedActivationKind activation) {
  switch (activation) {
    case GatedActivationKind::kSiLU:
      return x / (1.0f + expf(-x));
    case GatedActivationKind::kGELU:
      return GeluExact(x);
    case GatedActivationKind::kReLU:
      return x > 0.0f ? x : 0.0f;
    case GatedActivationKind::kLeakyRelu0p5Squared: {
      const float y = x > 0.0f ? x : (0.5f * x);
      return y * y;
    }
  }
  return x;
}

template <typename scalar_t>
__global__ void gated_activation_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols,
    GatedActivationKind activation) {
  const int64_t hidden = cols / 2;
  const int64_t elements = rows * hidden;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const int64_t row = idx / hidden;
  const int64_t col = idx % hidden;
  const int64_t base = row * cols;
  const float a = static_cast<float>(x[base + col]);
  const float gate = static_cast<float>(x[base + hidden + col]);
  const scalar_t activated =
      static_cast<scalar_t>(ApplyActivation(a, activation));
  const float value = static_cast<float>(activated) * gate;
  out[idx] = static_cast<scalar_t>(value);
}

}  // namespace

bool HasCudaGatedActivationKernel() {
  return true;
}

torch::Tensor CudaGatedActivationForward(
    const torch::Tensor& x,
    const std::string& activation) {
  TORCH_CHECK(x.is_cuda(), "CudaGatedActivationForward: x must be a CUDA tensor");
  TORCH_CHECK(x.dim() >= 1, "CudaGatedActivationForward: x must have rank >= 1");
  TORCH_CHECK(x.size(-1) % 2 == 0, "CudaGatedActivationForward: hidden width must be even");

  c10::cuda::CUDAGuard device_guard{x.device()};

  auto x_contig = x.contiguous();
  const auto cols = x_contig.size(-1);
  const auto hidden = cols / 2;
  const auto rows = x_contig.numel() / cols;

  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = hidden;
  auto out = torch::empty(out_sizes, x_contig.options());

  const auto activation_kind = ParseActivation(activation);
  const auto total = rows * hidden;
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_gated_activation_forward",
      [&] {
        gated_activation_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            rows,
            cols,
            activation_kind);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
