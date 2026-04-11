#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <vector>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void apply_rotary_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    int64_t bh,
    int64_t t,
    int64_t dh) {
  const int64_t half = dh / 2;
  const int64_t elements = bh * t * half;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }

  const int64_t col = idx % half;
  const int64_t time_idx = (idx / half) % t;
  const int64_t outer = idx / (half * t);
  const int64_t base = (outer * t + time_idx) * dh;

  const float cos_lo = static_cast<float>(cos[time_idx * dh + col]);
  const float cos_hi = static_cast<float>(cos[time_idx * dh + half + col]);
  const float sin_lo = static_cast<float>(sin[time_idx * dh + col]);
  const float sin_hi = static_cast<float>(sin[time_idx * dh + half + col]);

  const float q_lo = static_cast<float>(q[base + col]);
  const float q_hi = static_cast<float>(q[base + half + col]);
  const float k_lo = static_cast<float>(k[base + col]);
  const float k_hi = static_cast<float>(k[base + half + col]);

  q_out[base + col] = static_cast<scalar_t>((q_lo * cos_lo) - (q_hi * sin_lo));
  q_out[base + half + col] = static_cast<scalar_t>((q_hi * cos_hi) + (q_lo * sin_hi));
  k_out[base + col] = static_cast<scalar_t>((k_lo * cos_lo) - (k_hi * sin_lo));
  k_out[base + half + col] = static_cast<scalar_t>((k_hi * cos_hi) + (k_lo * sin_hi));
}

}  // namespace

bool HasCudaRopeKernel() {
  return true;
}

std::vector<torch::Tensor> CudaApplyRotaryForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda(), "CudaApplyRotaryForward: q and k must be CUDA tensors");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "CudaApplyRotaryForward: cos and sin must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "CudaApplyRotaryForward: q and k dtype mismatch");
  TORCH_CHECK(cos.scalar_type() == q.scalar_type(), "CudaApplyRotaryForward: cos dtype mismatch");
  TORCH_CHECK(sin.scalar_type() == q.scalar_type(), "CudaApplyRotaryForward: sin dtype mismatch");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "CudaApplyRotaryForward: q and k must be rank-4");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "CudaApplyRotaryForward: cos and sin must be rank-2");
  TORCH_CHECK(q.size(3) % 2 == 0, "CudaApplyRotaryForward: head_dim must be even");

  c10::cuda::CUDAGuard device_guard{q.device()};

  auto q_contig = q.contiguous();
  auto k_contig = k.contiguous();
  auto cos_contig = cos.contiguous();
  auto sin_contig = sin.contiguous();

  auto q_out = torch::empty_like(q_contig);
  auto k_out = torch::empty_like(k_contig);

  const auto bh = q_contig.size(0) * q_contig.size(1);
  const auto t = q_contig.size(2);
  const auto dh = q_contig.size(3);
  const auto elements = bh * t * (dh / 2);
  const dim3 blocks(static_cast<unsigned int>((elements + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q_contig.scalar_type(),
      "model_stack_cuda_apply_rotary_forward",
      [&] {
        apply_rotary_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            q_contig.data_ptr<scalar_t>(),
            k_contig.data_ptr<scalar_t>(),
            cos_contig.data_ptr<scalar_t>(),
            sin_contig.data_ptr<scalar_t>(),
            q_out.data_ptr<scalar_t>(),
            k_out.data_ptr<scalar_t>(),
            bh,
            t,
            dh);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_out};
}
