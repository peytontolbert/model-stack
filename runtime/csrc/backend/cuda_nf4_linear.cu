#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kThreads = 256;

bool IsSupportedNf4LinearDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

__device__ inline float DecodeNf4Nibble(uint8_t nibble) {
  switch (nibble & 0x0F) {
    case 0: return -1.0f;
    case 1: return -0.6961928f;
    case 2: return -0.52507305f;
    case 3: return -0.3949175f;
    case 4: return -0.28444138f;
    case 5: return -0.18477343f;
    case 6: return -0.09105004f;
    case 7: return 0.0f;
    case 8: return 0.0795803f;
    case 9: return 0.1609302f;
    case 10: return 0.2461123f;
    case 11: return 0.33791524f;
    case 12: return 0.44070983f;
    case 13: return 0.5626170f;
    case 14: return 0.72295684f;
    default: return 1.0f;
  }
}

template <typename scalar_t>
__global__ void nf4_linear_forward_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const float* __restrict__ weight_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t packed_cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * out_features;
  if (idx >= total) {
    return;
  }
  const int64_t row = idx / out_features;
  const int64_t col = idx % out_features;
  const scalar_t* x_row = x + (row * in_features);
  const uint8_t* w_row = packed_weight + (col * packed_cols);

  float acc = 0.0f;
  for (int64_t k = 0; k < in_features; ++k) {
    const uint8_t packed = w_row[k >> 1];
    const uint8_t nibble = (k & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    acc += static_cast<float>(x_row[k]) * DecodeNf4Nibble(nibble);
  }
  float value = acc * weight_scale[col];
  if (bias != nullptr) {
    value += static_cast<float>(bias[col]);
  }
  out[row * out_features + col] = static_cast<scalar_t>(value);
}

}  // namespace

bool HasCudaNf4LinearKernel() {
  return true;
}

torch::Tensor CudaNf4LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda(), "CudaNf4LinearForward: x must be a CUDA tensor");
  TORCH_CHECK(packed_weight.is_cuda(), "CudaNf4LinearForward: packed_weight must be a CUDA tensor");
  TORCH_CHECK(weight_scale.is_cuda(), "CudaNf4LinearForward: weight_scale must be a CUDA tensor");
  TORCH_CHECK(IsSupportedNf4LinearDtype(x.scalar_type()), "CudaNf4LinearForward: unsupported x dtype");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8, "CudaNf4LinearForward: packed_weight must be uint8");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32, "CudaNf4LinearForward: weight_scale must be float32");
  TORCH_CHECK(x.dim() >= 2, "CudaNf4LinearForward: x must have rank >= 2");
  TORCH_CHECK(packed_weight.dim() == 2, "CudaNf4LinearForward: packed_weight must be rank-2");
  TORCH_CHECK(weight_scale.dim() == 1, "CudaNf4LinearForward: weight_scale must be rank-1");
  TORCH_CHECK(weight_scale.size(0) == packed_weight.size(0), "CudaNf4LinearForward: weight_scale size mismatch");

  c10::cuda::CUDAGuard device_guard{x.device()};
  auto x_contig = x.contiguous();
  auto packed_contig = packed_weight.contiguous();
  auto scale_contig = weight_scale.contiguous();
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    bias_cast = bias.value().to(x_contig.device(), x_contig.scalar_type()).contiguous();
  }

  const auto in_features = x_contig.size(-1);
  const auto rows = x_contig.numel() / in_features;
  const auto out_features = packed_contig.size(0);
  const auto packed_cols = packed_contig.size(1);
  TORCH_CHECK(packed_cols == (in_features + 1) / 2, "CudaNf4LinearForward: packed weight column count mismatch");

  auto out_2d = torch::empty({rows, out_features}, x_contig.options());
  if (rows == 0 || out_features == 0) {
    std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
    out_sizes.back() = out_features;
    return out_2d.view(out_sizes);
  }

  const dim3 threads(kThreads);
  const dim3 blocks(static_cast<unsigned int>((rows * out_features + kThreads - 1) / kThreads));
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_nf4_linear_forward",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias_cast.has_value() && bias_cast.value().defined()) {
          bias_ptr = bias_cast.value().data_ptr<scalar_t>();
        }
        nf4_linear_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_contig.data_ptr<scalar_t>(),
            packed_contig.data_ptr<uint8_t>(),
            scale_contig.data_ptr<float>(),
            bias_ptr,
            out_2d.data_ptr<scalar_t>(),
            rows,
            out_features,
            in_features,
            packed_cols);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = out_features;
  return out_2d.view(out_sizes);
}
