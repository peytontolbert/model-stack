#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreads = 256;

bool IsSupportedFp8LinearDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

template <typename scalar_t>
__global__ void fp8_linear_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight_fp8,
    float weight_scale,
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
  const scalar_t* x_row = x + (row * in_features);
  const scalar_t* w_row = weight_fp8 + (col * in_features);

  float acc = 0.0f;
  for (int64_t k = 0; k < in_features; ++k) {
    acc += static_cast<float>(x_row[k]) * static_cast<float>(w_row[k]);
  }
  float value = acc * weight_scale;
  if (bias != nullptr) {
    value += static_cast<float>(bias[col]);
  }
  out[row * out_features + col] = static_cast<scalar_t>(value);
}

}  // namespace

bool HasCudaFp8LinearKernel() {
  return true;
}

torch::Tensor CudaFp8LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    double weight_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.is_cuda(), "CudaFp8LinearForward: x must be a CUDA tensor");
  TORCH_CHECK(weight_fp8.is_cuda(), "CudaFp8LinearForward: weight_fp8 must be a CUDA tensor");
  TORCH_CHECK(IsSupportedFp8LinearDtype(x.scalar_type()), "CudaFp8LinearForward: unsupported x dtype");
  TORCH_CHECK(IsSupportedFp8LinearDtype(weight_fp8.scalar_type()), "CudaFp8LinearForward: unsupported weight dtype");
  TORCH_CHECK(x.dim() >= 2, "CudaFp8LinearForward: x must have rank >= 2");
  TORCH_CHECK(weight_fp8.dim() == 2, "CudaFp8LinearForward: weight_fp8 must be rank-2");
  TORCH_CHECK(x.size(-1) == weight_fp8.size(1), "CudaFp8LinearForward: input feature size mismatch");

  const auto output_dtype = out_dtype.value_or(x.scalar_type());
  TORCH_CHECK(IsSupportedFp8LinearDtype(output_dtype), "CudaFp8LinearForward: unsupported output dtype");

  c10::cuda::CUDAGuard device_guard{x.device()};
  auto x_cast = x.to(x.device(), output_dtype).contiguous();
  auto weight_cast = weight_fp8.to(x.device(), output_dtype).contiguous();
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    bias_cast = bias.value().to(x.device(), output_dtype).contiguous();
  }

  const auto in_features = x_cast.size(-1);
  const auto rows = x_cast.numel() / in_features;
  const auto out_features = weight_cast.size(0);
  auto out_2d = torch::empty({rows, out_features}, x_cast.options());
  if (rows == 0 || out_features == 0) {
    std::vector<int64_t> out_sizes(x_cast.sizes().begin(), x_cast.sizes().end());
    out_sizes.back() = out_features;
    return out_2d.view(out_sizes);
  }

  const dim3 threads(kThreads);
  const dim3 blocks(static_cast<unsigned int>((rows * out_features + kThreads - 1) / kThreads));
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_cast.scalar_type(),
      "model_stack_cuda_fp8_linear_forward",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias_cast.has_value() && bias_cast.value().defined()) {
          bias_ptr = bias_cast.value().data_ptr<scalar_t>();
        }
        fp8_linear_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_cast.data_ptr<scalar_t>(),
            weight_cast.data_ptr<scalar_t>(),
            static_cast<float>(weight_scale),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            rows,
            out_features,
            in_features);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(x_cast.sizes().begin(), x_cast.sizes().end());
  out_sizes.back() = out_features;
  return out_2d.view(out_sizes);
}
