#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <cmath>
#include <tuple>
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

bool IsSupportedInt8FrontendDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
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
    const float rounded = nearbyintf(scaled);
    const float clamped = fminf(127.0f, fmaxf(-127.0f, rounded));
    qx[row * cols + col] = static_cast<int8_t>(static_cast<int>(clamped));
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
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  return {qx_2d.view(out_sizes), row_scale};
}

}  // namespace

bool HasCudaInt8QuantFrontendKernel() {
  return true;
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
