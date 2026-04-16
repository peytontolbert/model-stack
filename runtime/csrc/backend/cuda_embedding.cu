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
__global__ void embedding_forward_kernel(
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ out,
    int64_t num_indices,
    int64_t embedding_dim) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_indices * embedding_dim;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % embedding_dim;
  const int64_t row = idx / embedding_dim;
  const int64_t token = indices[row];
  out[idx] = weight[token * embedding_dim + d];
}

}  // namespace

bool HasCudaEmbeddingKernel() {
  return true;
}

torch::Tensor CudaEmbeddingForward(
    const torch::Tensor& weight,
    const torch::Tensor& indices,
    int64_t padding_idx) {
  TORCH_CHECK(weight.is_cuda() && indices.is_cuda(), "CudaEmbeddingForward: weight and indices must be CUDA tensors");
  TORCH_CHECK(weight.dim() == 2, "CudaEmbeddingForward: weight must be rank-2");
  TORCH_CHECK(indices.scalar_type() == torch::kLong || indices.scalar_type() == torch::kInt,
              "CudaEmbeddingForward: indices must be int32 or int64");
  TORCH_CHECK(weight.device() == indices.device(), "CudaEmbeddingForward: weight and indices device mismatch");
  TORCH_CHECK(padding_idx >= -1 && padding_idx < weight.size(0),
              "CudaEmbeddingForward: padding_idx out of range");

  c10::cuda::CUDAGuard device_guard{weight.device()};

  auto weight_contig = weight.contiguous();
  auto indices_contig = indices.to(torch::kLong).contiguous();
  if (indices_contig.numel() == 0) {
    std::vector<int64_t> out_sizes(indices_contig.sizes().begin(), indices_contig.sizes().end());
    out_sizes.push_back(weight_contig.size(1));
    return torch::empty(out_sizes, weight_contig.options());
  }

  const auto min_index = indices_contig.min().item<int64_t>();
  const auto max_index = indices_contig.max().item<int64_t>();
  TORCH_CHECK(min_index >= 0, "CudaEmbeddingForward: indices must be non-negative");
  TORCH_CHECK(max_index < weight_contig.size(0), "CudaEmbeddingForward: index out of range");

  std::vector<int64_t> out_sizes(indices_contig.sizes().begin(), indices_contig.sizes().end());
  out_sizes.push_back(weight_contig.size(1));
  auto out = torch::empty(out_sizes, weight_contig.options());

  const auto num_indices = indices_contig.numel();
  const auto embedding_dim = weight_contig.size(1);
  const auto total = num_indices * embedding_dim;
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight_contig.scalar_type(),
      "model_stack_cuda_embedding_forward",
      [&] {
        embedding_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            weight_contig.data_ptr<scalar_t>(),
            indices_contig.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            num_indices,
            embedding_dim);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
