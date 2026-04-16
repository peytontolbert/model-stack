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
__global__ void append_columns_kernel(
    const scalar_t* __restrict__ base,
    const scalar_t* __restrict__ suffix,
    scalar_t* __restrict__ out,
    int64_t batch_size,
    int64_t base_cols,
    int64_t suffix_cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch_size * (base_cols + suffix_cols);
  if (idx >= total) {
    return;
  }

  const int64_t col = idx % (base_cols + suffix_cols);
  const int64_t row = idx / (base_cols + suffix_cols);
  if (col < base_cols) {
    out[idx] = base[(row * base_cols) + col];
    return;
  }
  out[idx] = suffix[(row * suffix_cols) + (col - base_cols)];
}

template <typename scalar_t>
torch::Tensor AppendColumnsTyped(
    const torch::Tensor& base,
    const torch::Tensor& suffix) {
  auto base_contig = base.contiguous();
  auto suffix_contig = suffix.contiguous();
  auto out = torch::empty(
      {base_contig.size(0), base_contig.size(1) + suffix_contig.size(1)},
      base_contig.options());
  const auto total = out.numel();
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(base_contig.get_device());
  append_columns_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
      base_contig.data_ptr<scalar_t>(),
      suffix_contig.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      base_contig.size(0),
      base_contig.size(1),
      suffix_contig.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor AppendColumnsDispatch(
    const torch::Tensor& base,
    const torch::Tensor& suffix) {
  TORCH_CHECK(base.scalar_type() == suffix.scalar_type(),
              "CudaAppendTokensForward: dtype mismatch");
  TORCH_CHECK(base.is_cuda() && suffix.is_cuda(),
              "CudaAppendTokensForward: tensors must be CUDA");
  TORCH_CHECK(base.dim() == 2 && suffix.dim() == 2,
              "CudaAppendTokensForward: tensors must be rank-2");
  TORCH_CHECK(base.size(0) == suffix.size(0),
              "CudaAppendTokensForward: batch mismatch");
  TORCH_CHECK(
      base.scalar_type() == torch::kBool ||
      base.scalar_type() == torch::kInt ||
      base.scalar_type() == torch::kLong,
      "CudaAppendTokensForward: only bool/int32/int64 tensors are supported");

  if (base.scalar_type() == torch::kBool) {
    return AppendColumnsTyped<bool>(base, suffix);
  }

  torch::Tensor out;
  AT_DISPATCH_INTEGRAL_TYPES(
      base.scalar_type(),
      "model_stack_cuda_append_tokens_forward",
      [&] {
        out = AppendColumnsTyped<scalar_t>(base, suffix);
      });
  return out;
}

}  // namespace

bool HasCudaAppendTokensKernel() {
  return true;
}

std::vector<torch::Tensor> CudaAppendTokensForward(
    const torch::Tensor& seq,
    const torch::Tensor& next_id,
    const c10::optional<torch::Tensor>& attention_mask) {
  TORCH_CHECK(seq.is_cuda() && next_id.is_cuda(),
              "CudaAppendTokensForward: seq and next_id must be CUDA tensors");
  c10::cuda::CUDAGuard device_guard{seq.device()};

  auto next_seq = AppendColumnsDispatch(seq, next_id);
  if (!attention_mask.has_value() || !attention_mask.value().defined()) {
    return {next_seq, torch::Tensor()};
  }

  const auto& mask = attention_mask.value();
  TORCH_CHECK(mask.is_cuda(), "CudaAppendTokensForward: attention_mask must be a CUDA tensor");
  TORCH_CHECK(mask.size(0) == next_id.size(0),
              "CudaAppendTokensForward: attention_mask batch mismatch");
  auto suffix = torch::ones(
      {next_id.size(0), next_id.size(1)},
      torch::TensorOptions().dtype(mask.scalar_type()).device(mask.device()));
  auto next_mask = AppendColumnsDispatch(mask, suffix);
  return {next_seq, next_mask};
}
