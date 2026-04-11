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
__global__ void layer_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float shared_sum[kThreads];
  __shared__ float shared_sqsum[kThreads];
  float thread_sum = 0.0f;
  float thread_sqsum = 0.0f;
  const int64_t row_offset = row * cols;

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = static_cast<float>(x[row_offset + col]);
    thread_sum += value;
    thread_sqsum += value * value;
  }
  shared_sum[threadIdx.x] = thread_sum;
  shared_sqsum[threadIdx.x] = thread_sqsum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
      shared_sqsum[threadIdx.x] += shared_sqsum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float mean = shared_sum[0] / static_cast<float>(cols);
  const float var = fmaxf((shared_sqsum[0] / static_cast<float>(cols)) - (mean * mean), 0.0f);
  const float inv_std = rsqrtf(var + eps);
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = (static_cast<float>(x[row_offset + col]) - mean) * inv_std;
    if (weight != nullptr) {
      value *= static_cast<float>(weight[col]);
    }
    if (bias != nullptr) {
      value += static_cast<float>(bias[col]);
    }
    out[row_offset + col] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void add_layer_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
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
  __shared__ float shared_sqsum[kThreads];
  float thread_sum = 0.0f;
  float thread_sqsum = 0.0f;
  const int64_t row_offset = row * cols;

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t idx = row_offset + col;
    const float value =
        static_cast<float>(x[idx]) + (residual_scale * static_cast<float>(update[idx]));
    const scalar_t combined_value = static_cast<scalar_t>(value);
    combined[idx] = combined_value;
    const float rounded = static_cast<float>(combined_value);
    thread_sum += rounded;
    thread_sqsum += rounded * rounded;
  }
  shared_sum[threadIdx.x] = thread_sum;
  shared_sqsum[threadIdx.x] = thread_sqsum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
      shared_sqsum[threadIdx.x] += shared_sqsum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float mean = shared_sum[0] / static_cast<float>(cols);
  const float var = fmaxf((shared_sqsum[0] / static_cast<float>(cols)) - (mean * mean), 0.0f);
  const float inv_std = rsqrtf(var + eps);
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t idx = row_offset + col;
    float value = (static_cast<float>(combined[idx]) - mean) * inv_std;
    if (weight != nullptr) {
      value *= static_cast<float>(weight[col]);
    }
    if (bias != nullptr) {
      value += static_cast<float>(bias[col]);
    }
    normalized[idx] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
void launch_layer_norm_forward(
    const torch::Tensor& x,
    const torch::Tensor* weight,
    const torch::Tensor* bias,
    torch::Tensor* out,
    int64_t rows,
    int64_t cols,
    float eps,
    cudaStream_t stream) {
  const dim3 blocks(static_cast<unsigned int>(rows));
  const dim3 threads(kThreads);
  layer_norm_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      x.data_ptr<scalar_t>(),
      weight != nullptr ? weight->data_ptr<scalar_t>() : nullptr,
      bias != nullptr ? bias->data_ptr<scalar_t>() : nullptr,
      out->data_ptr<scalar_t>(),
      rows,
      cols,
      eps);
}

template <typename scalar_t>
void launch_add_layer_norm_forward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const torch::Tensor* weight,
    const torch::Tensor* bias,
    torch::Tensor* combined,
    torch::Tensor* normalized,
    int64_t rows,
    int64_t cols,
    float residual_scale,
    float eps,
    cudaStream_t stream) {
  const dim3 blocks(static_cast<unsigned int>(rows));
  const dim3 threads(kThreads);
  add_layer_norm_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      x.data_ptr<scalar_t>(),
      update.data_ptr<scalar_t>(),
      weight != nullptr ? weight->data_ptr<scalar_t>() : nullptr,
      bias != nullptr ? bias->data_ptr<scalar_t>() : nullptr,
      combined->data_ptr<scalar_t>(),
      normalized->data_ptr<scalar_t>(),
      rows,
      cols,
      residual_scale,
      eps);
}

}  // namespace

bool HasCudaLayerNormKernel() {
  return true;
}

bool HasCudaAddLayerNormKernel() {
  return true;
}

torch::Tensor CudaLayerNormForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "CudaLayerNormForward: x must be a CUDA tensor");
  TORCH_CHECK(x.dim() >= 1, "CudaLayerNormForward: x must have at least one dimension");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "CudaLayerNormForward: eps must be positive and finite");

  const auto cols = x.size(-1);
  TORCH_CHECK(cols > 0, "CudaLayerNormForward: last dimension must be non-empty");

  c10::cuda::CUDAGuard device_guard{x.device()};

  auto x_contig = x.contiguous();
  torch::Tensor weight_contig;
  torch::Tensor bias_contig;
  const torch::Tensor* weight_ptr = nullptr;
  const torch::Tensor* bias_ptr = nullptr;

  if (weight.has_value() && weight.value().defined()) {
    weight_contig = weight.value().to(x.device(), x.scalar_type()).contiguous();
    TORCH_CHECK(weight_contig.dim() == 1, "CudaLayerNormForward: weight must be rank-1");
    TORCH_CHECK(weight_contig.size(0) == cols, "CudaLayerNormForward: weight size mismatch");
    weight_ptr = &weight_contig;
  }
  if (bias.has_value() && bias.value().defined()) {
    bias_contig = bias.value().to(x.device(), x.scalar_type()).contiguous();
    TORCH_CHECK(bias_contig.dim() == 1, "CudaLayerNormForward: bias must be rank-1");
    TORCH_CHECK(bias_contig.size(0) == cols, "CudaLayerNormForward: bias size mismatch");
    bias_ptr = &bias_contig;
  }

  auto out = torch::empty_like(x_contig);
  const auto rows = x_contig.numel() / cols;
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_layer_norm_forward",
      [&] {
        launch_layer_norm_forward<scalar_t>(
            x_contig,
            weight_ptr,
            bias_ptr,
            &out,
            rows,
            cols,
            static_cast<float>(eps),
            stream.stream());
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.view(x.sizes());
}

std::vector<torch::Tensor> CudaAddLayerNormForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double residual_scale,
    double eps) {
  TORCH_CHECK(x.is_cuda() && update.is_cuda(), "CudaAddLayerNormForward: x and update must be CUDA tensors");
  TORCH_CHECK(x.dim() >= 1 && update.dim() >= 1, "CudaAddLayerNormForward: x and update must have rank >= 1");
  TORCH_CHECK(x.sizes() == update.sizes(), "CudaAddLayerNormForward: x and update shape mismatch");
  TORCH_CHECK(x.scalar_type() == update.scalar_type(), "CudaAddLayerNormForward: x and update dtype mismatch");
  TORCH_CHECK(std::isfinite(residual_scale), "CudaAddLayerNormForward: residual_scale must be finite");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "CudaAddLayerNormForward: eps must be positive and finite");

  const auto cols = x.size(-1);
  TORCH_CHECK(cols > 0, "CudaAddLayerNormForward: last dimension must be non-empty");

  c10::cuda::CUDAGuard device_guard{x.device()};

  auto x_contig = x.contiguous();
  auto update_contig = update.contiguous();
  torch::Tensor weight_contig;
  torch::Tensor bias_contig;
  const torch::Tensor* weight_ptr = nullptr;
  const torch::Tensor* bias_ptr = nullptr;

  if (weight.has_value() && weight.value().defined()) {
    weight_contig = weight.value().to(x.device(), x.scalar_type()).contiguous();
    TORCH_CHECK(weight_contig.dim() == 1, "CudaAddLayerNormForward: weight must be rank-1");
    TORCH_CHECK(weight_contig.size(0) == cols, "CudaAddLayerNormForward: weight size mismatch");
    weight_ptr = &weight_contig;
  }
  if (bias.has_value() && bias.value().defined()) {
    bias_contig = bias.value().to(x.device(), x.scalar_type()).contiguous();
    TORCH_CHECK(bias_contig.dim() == 1, "CudaAddLayerNormForward: bias must be rank-1");
    TORCH_CHECK(bias_contig.size(0) == cols, "CudaAddLayerNormForward: bias size mismatch");
    bias_ptr = &bias_contig;
  }

  auto combined = torch::empty_like(x_contig);
  auto normalized = torch::empty_like(x_contig);
  const auto rows = x_contig.numel() / cols;
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_add_layer_norm_forward",
      [&] {
        launch_add_layer_norm_forward<scalar_t>(
            x_contig,
            update_contig,
            weight_ptr,
            bias_ptr,
            &combined,
            &normalized,
            rows,
            cols,
            static_cast<float>(residual_scale),
            static_cast<float>(eps),
            stream.stream());
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {combined.view(x.sizes()), normalized.view(x.sizes())};
}
