#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>

#include <cstdint>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kMaxThreads = 256;
constexpr int kVecBytes = 16;

template <typename scalar_t>
struct alignas(kVecBytes) Vec128 {
  scalar_t elems[kVecBytes / sizeof(scalar_t)];
};

inline int ResolveEmbeddingThreads(
    int64_t embedding_dim,
    size_t scalar_bytes,
    bool use_vec16) {
  const int64_t work_items = use_vec16
      ? std::max<int64_t>(1, (embedding_dim * static_cast<int64_t>(scalar_bytes)) / kVecBytes)
      : std::max<int64_t>(1, embedding_dim);
  if (work_items <= 32) {
    return 32;
  }
  if (work_items <= 64) {
    return 64;
  }
  if (work_items <= 128) {
    return 128;
  }
  return kMaxThreads;
}

inline bool CanUseEmbeddingVec16(
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t embedding_dim) {
  const auto row_bytes = embedding_dim * static_cast<int64_t>(weight.element_size());
  if (row_bytes <= 0 || (row_bytes % kVecBytes) != 0) {
    return false;
  }
  const auto weight_ptr = reinterpret_cast<uintptr_t>(weight.data_ptr());
  const auto out_ptr = reinterpret_cast<uintptr_t>(out.data_ptr());
  return (weight_ptr % kVecBytes) == 0 && (out_ptr % kVecBytes) == 0;
}

template <typename index_t, bool Checked>
__device__ inline bool load_embedding_token(
    index_t raw_index,
    int64_t* token_out,
    int64_t num_embeddings) {
  const int64_t token = static_cast<int64_t>(raw_index);
  if constexpr (Checked) {
    if (C10_UNLIKELY(token < 0 || token >= num_embeddings)) {
      CUDA_KERNEL_ASSERT(token >= 0 && token < num_embeddings);
      return false;
    }
  }
  *token_out = token;
  return true;
}

template <typename scalar_t, typename index_t, bool Checked>
__global__ void embedding_forward_rowwise_scalar_kernel(
    const scalar_t* __restrict__ weight,
    const index_t* __restrict__ indices,
    scalar_t* __restrict__ out,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_indices) {
    return;
  }

  int64_t token = 0;
  if (!load_embedding_token<index_t, Checked>(indices[row], &token, num_embeddings)) {
    return;
  }

  const scalar_t* src = weight + (token * embedding_dim);
  scalar_t* dst = out + (row * embedding_dim);
  for (int64_t col = threadIdx.x; col < embedding_dim; col += blockDim.x) {
    dst[col] = src[col];
  }
}

template <typename scalar_t, typename index_t, bool Checked>
__global__ void embedding_forward_rowwise_vec16_kernel(
    const scalar_t* __restrict__ weight,
    const index_t* __restrict__ indices,
    scalar_t* __restrict__ out,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_indices) {
    return;
  }

  int64_t token = 0;
  if (!load_embedding_token<index_t, Checked>(indices[row], &token, num_embeddings)) {
    return;
  }

  constexpr int kElemsPerVec = kVecBytes / sizeof(scalar_t);
  using Vec = Vec128<scalar_t>;
  const auto* src = reinterpret_cast<const Vec*>(weight + (token * embedding_dim));
  auto* dst = reinterpret_cast<Vec*>(out + (row * embedding_dim));
  const int64_t vec_count = embedding_dim / kElemsPerVec;
  for (int64_t vec = threadIdx.x; vec < vec_count; vec += blockDim.x) {
    dst[vec] = src[vec];
  }
}

}  // namespace

bool HasCudaEmbeddingKernel() {
  return true;
}

torch::Tensor CudaEmbeddingForwardUnchecked(
    const torch::Tensor& weight,
    const torch::Tensor& indices) {
  TORCH_CHECK(weight.is_cuda() && indices.is_cuda(), "CudaEmbeddingForwardUnchecked: weight and indices must be CUDA tensors");
  TORCH_CHECK(weight.dim() == 2, "CudaEmbeddingForwardUnchecked: weight must be rank-2");
  TORCH_CHECK(indices.scalar_type() == torch::kLong || indices.scalar_type() == torch::kInt,
              "CudaEmbeddingForwardUnchecked: indices must be int32 or int64");
  TORCH_CHECK(weight.device() == indices.device(), "CudaEmbeddingForwardUnchecked: weight and indices device mismatch");

  c10::cuda::CUDAGuard device_guard{weight.device()};

  auto weight_contig = weight.contiguous();
  auto indices_contig = indices.to(torch::kLong).contiguous();
  if (indices_contig.numel() == 0) {
    std::vector<int64_t> out_sizes(indices_contig.sizes().begin(), indices_contig.sizes().end());
    out_sizes.push_back(weight_contig.size(1));
    return torch::empty(out_sizes, weight_contig.options());
  }

  std::vector<int64_t> out_sizes(indices_contig.sizes().begin(), indices_contig.sizes().end());
  out_sizes.push_back(weight_contig.size(1));
  auto out = torch::empty(out_sizes, weight_contig.options());

  const auto num_indices = indices_contig.numel();
  const auto embedding_dim = weight_contig.size(1);
  const bool use_vec16 = CanUseEmbeddingVec16(weight_contig, out, embedding_dim);
  const dim3 blocks(static_cast<unsigned int>(num_indices));
  const dim3 threads(ResolveEmbeddingThreads(embedding_dim, weight_contig.element_size(), use_vec16));
  auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight_contig.scalar_type(),
      "model_stack_cuda_embedding_forward_unchecked",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices_contig.scalar_type(),
            "model_stack_cuda_embedding_forward_unchecked_index",
            [&] {
              if (use_vec16) {
                embedding_forward_rowwise_vec16_kernel<scalar_t, index_t, false><<<blocks, threads, 0, stream.stream()>>>(
                    weight_contig.data_ptr<scalar_t>(),
                    indices_contig.data_ptr<index_t>(),
                    out.data_ptr<scalar_t>(),
                    num_indices,
                    embedding_dim,
                    weight_contig.size(0));
              } else {
                embedding_forward_rowwise_scalar_kernel<scalar_t, index_t, false><<<blocks, threads, 0, stream.stream()>>>(
                    weight_contig.data_ptr<scalar_t>(),
                    indices_contig.data_ptr<index_t>(),
                    out.data_ptr<scalar_t>(),
                    num_indices,
                    embedding_dim,
                    weight_contig.size(0));
              }
            });
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
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
  auto indices_contig = indices.contiguous();
  if (indices_contig.numel() == 0) {
    std::vector<int64_t> out_sizes(indices_contig.sizes().begin(), indices_contig.sizes().end());
    out_sizes.push_back(weight_contig.size(1));
    return torch::empty(out_sizes, weight_contig.options());
  }

  std::vector<int64_t> out_sizes(indices_contig.sizes().begin(), indices_contig.sizes().end());
  out_sizes.push_back(weight_contig.size(1));
  auto out = torch::empty(out_sizes, weight_contig.options());

  const auto num_indices = indices_contig.numel();
  const auto embedding_dim = weight_contig.size(1);
  const bool use_vec16 = CanUseEmbeddingVec16(weight_contig, out, embedding_dim);
  const dim3 blocks(static_cast<unsigned int>(num_indices));
  const dim3 threads(ResolveEmbeddingThreads(embedding_dim, weight_contig.element_size(), use_vec16));
  auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight_contig.scalar_type(),
      "model_stack_cuda_embedding_forward",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices_contig.scalar_type(),
            "model_stack_cuda_embedding_forward_index",
            [&] {
              if (use_vec16) {
                embedding_forward_rowwise_vec16_kernel<scalar_t, index_t, true><<<blocks, threads, 0, stream.stream()>>>(
                    weight_contig.data_ptr<scalar_t>(),
                    indices_contig.data_ptr<index_t>(),
                    out.data_ptr<scalar_t>(),
                    num_indices,
                    embedding_dim,
                    weight_contig.size(0));
              } else {
                embedding_forward_rowwise_scalar_kernel<scalar_t, index_t, true><<<blocks, threads, 0, stream.stream()>>>(
                    weight_contig.data_ptr<scalar_t>(),
                    indices_contig.data_ptr<index_t>(),
                    out.data_ptr<scalar_t>(),
                    num_indices,
                    embedding_dim,
                    weight_contig.size(0));
              }
            });
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
