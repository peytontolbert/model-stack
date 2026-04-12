#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void residual_add_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ update,
    scalar_t* __restrict__ out,
    int64_t numel,
    float residual_scale) {
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  const float value =
      static_cast<float>(x[idx]) + (residual_scale * static_cast<float>(update[idx]));
  out[idx] = static_cast<scalar_t>(value);
}

}  // namespace

bool HasCudaResidualAddKernel() {
  return true;
}

torch::Tensor CudaResidualAddForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    double residual_scale) {
  TORCH_CHECK(x.is_cuda() && update.is_cuda(), "CudaResidualAddForward: x and update must be CUDA tensors");
  TORCH_CHECK(x.dim() >= 1 && update.dim() >= 1, "CudaResidualAddForward: x and update must have rank >= 1");
  TORCH_CHECK(x.sizes() == update.sizes(), "CudaResidualAddForward: x and update shape mismatch");
  TORCH_CHECK(x.scalar_type() == update.scalar_type(), "CudaResidualAddForward: x and update dtype mismatch");
  TORCH_CHECK(std::isfinite(residual_scale), "CudaResidualAddForward: residual_scale must be finite");

  c10::cuda::CUDAGuard device_guard{x.device()};

  auto x_contig = x.contiguous();
  auto update_contig = update.contiguous();
  auto out = torch::empty_like(x_contig);
  const auto numel = x_contig.numel();
  const dim3 blocks(static_cast<unsigned int>((numel + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_residual_add_forward",
      [&] {
        residual_add_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_contig.data_ptr<scalar_t>(),
            update_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<float>(residual_scale));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.view(x.sizes());
}
