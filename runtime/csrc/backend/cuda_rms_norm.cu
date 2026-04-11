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
__global__ void rms_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float shared_sum[kThreads];
  float thread_sum = 0.0f;
  const int64_t row_offset = row * cols;

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = static_cast<float>(x[row_offset + col]);
    thread_sum += value * value;
  }
  shared_sum[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf((shared_sum[0] / static_cast<float>(cols)) + eps);
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = static_cast<float>(x[row_offset + col]) * inv_rms;
    if (weight != nullptr) {
      value *= static_cast<float>(weight[col]);
    }
    out[row_offset + col] = static_cast<scalar_t>(value);
  }
}

}  // namespace

bool HasCudaRmsNormKernel() {
  return true;
}

torch::Tensor CudaRmsNormForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "CudaRmsNormForward: x must be a CUDA tensor");
  TORCH_CHECK(x.dim() >= 1, "CudaRmsNormForward: x must have at least one dimension");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "CudaRmsNormForward: eps must be positive and finite");

  const auto cols = x.size(-1);
  TORCH_CHECK(cols > 0, "CudaRmsNormForward: last dimension must be non-empty");

  c10::cuda::CUDAGuard device_guard{x.device()};

  auto x_contig = x.contiguous();
  torch::Tensor weight_contig;
  const torch::Tensor* weight_ptr = nullptr;
  if (weight.has_value() && weight.value().defined()) {
    weight_contig = weight.value().to(x.device(), x.scalar_type()).contiguous();
    TORCH_CHECK(weight_contig.dim() == 1, "CudaRmsNormForward: weight must be rank-1");
    TORCH_CHECK(weight_contig.size(0) == cols, "CudaRmsNormForward: weight size mismatch");
    weight_ptr = &weight_contig;
  }

  auto out = torch::empty_like(x_contig);
  const auto rows = x_contig.numel() / cols;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_rms_norm_forward",
      [&] {
        rms_norm_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_contig.data_ptr<scalar_t>(),
            weight_ptr != nullptr ? weight_ptr->data_ptr<scalar_t>() : nullptr,
            out.data_ptr<scalar_t>(),
            rows,
            cols,
            static_cast<float>(eps));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.view(x.sizes());
}
