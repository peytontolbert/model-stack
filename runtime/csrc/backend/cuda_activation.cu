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

enum class ActivationKind : int {
  kSiLU = 0,
  kGELU = 1,
  kReLU = 2,
};

ActivationKind ParseActivation(const std::string& activation) {
  if (activation == "silu" || activation == "swish" || activation == "swiglu" || activation == "gated-silu") {
    return ActivationKind::kSiLU;
  }
  if (activation == "gelu" || activation == "geglu") {
    return ActivationKind::kGELU;
  }
  if (activation == "relu" || activation == "reglu") {
    return ActivationKind::kReLU;
  }
  return ActivationKind::kGELU;
}

__device__ inline float GeluExact(float x) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ inline float ApplyActivationValue(float x, ActivationKind activation) {
  switch (activation) {
    case ActivationKind::kSiLU:
      return x / (1.0f + expf(-x));
    case ActivationKind::kGELU:
      return GeluExact(x);
    case ActivationKind::kReLU:
      return x > 0.0f ? x : 0.0f;
  }
  return x;
}

template <typename scalar_t>
__global__ void activation_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int64_t elements,
    ActivationKind activation) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  const float value = static_cast<float>(x[idx]);
  out[idx] = static_cast<scalar_t>(ApplyActivationValue(value, activation));
}

}  // namespace

bool HasCudaActivationKernel() {
  return true;
}

torch::Tensor CudaActivationForward(
    const torch::Tensor& x,
    const std::string& activation) {
  TORCH_CHECK(x.is_cuda(), "CudaActivationForward: x must be a CUDA tensor");
  TORCH_CHECK(x.dim() >= 1, "CudaActivationForward: x must have rank >= 1");

  c10::cuda::CUDAGuard device_guard{x.device()};

  auto x_contig = x.contiguous();
  auto out = torch::empty_like(x_contig);
  const auto elements = x_contig.numel();
  const auto activation_kind = ParseActivation(activation);
  const dim3 blocks(static_cast<unsigned int>((elements + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_activation_forward",
      [&] {
        activation_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            elements,
            activation_kind);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
