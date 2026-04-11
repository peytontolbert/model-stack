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
__global__ void add_rms_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ combined,
    scalar_t* __restrict__ normalized,
    int64_t rows,
    int64_t cols,
    float residual_scale,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float shared_sum[kThreads];
  float thread_sum = 0.0f;
  const int64_t row_offset = row * cols;

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t idx = row_offset + col;
    const float value =
        static_cast<float>(x[idx]) + (residual_scale * static_cast<float>(update[idx]));
    const scalar_t combined_value = static_cast<scalar_t>(value);
    combined[idx] = combined_value;
    const float rounded = static_cast<float>(combined_value);
    thread_sum += rounded * rounded;
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
    const int64_t idx = row_offset + col;
    float value = static_cast<float>(combined[idx]) * inv_rms;
    if (weight != nullptr) {
      value *= static_cast<float>(weight[col]);
    }
    normalized[idx] = static_cast<scalar_t>(value);
  }
}

}  // namespace

bool HasCudaAddRmsNormKernel() {
  return true;
}

std::vector<torch::Tensor> CudaAddRmsNormForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& weight,
    double residual_scale,
    double eps) {
  TORCH_CHECK(x.is_cuda() && update.is_cuda(), "CudaAddRmsNormForward: x and update must be CUDA tensors");
  TORCH_CHECK(x.dim() >= 1 && update.dim() >= 1, "CudaAddRmsNormForward: x and update must have rank >= 1");
  TORCH_CHECK(x.sizes() == update.sizes(), "CudaAddRmsNormForward: x and update shape mismatch");
  TORCH_CHECK(x.scalar_type() == update.scalar_type(), "CudaAddRmsNormForward: x and update dtype mismatch");
  TORCH_CHECK(std::isfinite(residual_scale), "CudaAddRmsNormForward: residual_scale must be finite");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "CudaAddRmsNormForward: eps must be positive and finite");

  const auto cols = x.size(-1);
  TORCH_CHECK(cols > 0, "CudaAddRmsNormForward: last dimension must be non-empty");

  c10::cuda::CUDAGuard device_guard{x.device()};

  auto x_contig = x.contiguous();
  auto update_contig = update.contiguous();
  torch::Tensor weight_contig;
  const torch::Tensor* weight_ptr = nullptr;
  if (weight.has_value() && weight.value().defined()) {
    weight_contig = weight.value().to(x.device(), x.scalar_type()).contiguous();
    TORCH_CHECK(weight_contig.dim() == 1, "CudaAddRmsNormForward: weight must be rank-1");
    TORCH_CHECK(weight_contig.size(0) == cols, "CudaAddRmsNormForward: weight size mismatch");
    weight_ptr = &weight_contig;
  }

  auto combined = torch::empty_like(x_contig);
  auto normalized = torch::empty_like(x_contig);
  const auto rows = x_contig.numel() / cols;
  const dim3 blocks(static_cast<unsigned int>(rows));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_add_rms_norm_forward",
      [&] {
        add_rms_norm_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_contig.data_ptr<scalar_t>(),
            update_contig.data_ptr<scalar_t>(),
            weight_ptr != nullptr ? weight_ptr->data_ptr<scalar_t>() : nullptr,
            combined.data_ptr<scalar_t>(),
            normalized.data_ptr<scalar_t>(),
            rows,
            cols,
            static_cast<float>(residual_scale),
            static_cast<float>(eps));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {combined.view(x.sizes()), normalized.view(x.sizes())};
}
