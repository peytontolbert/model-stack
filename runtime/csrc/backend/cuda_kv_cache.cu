#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void kv_cache_write_forward_kernel(
    scalar_t* __restrict__ cache,
    const scalar_t* __restrict__ chunk,
    int64_t heads,
    int64_t cache_seq,
    int64_t chunk_seq,
    int64_t head_dim,
    int64_t start) {
  const int64_t total = heads * chunk_seq * head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % head_dim;
  const int64_t t = (idx / head_dim) % chunk_seq;
  const int64_t h = idx / (head_dim * chunk_seq);

  const int64_t chunk_offset = ((h * chunk_seq) + t) * head_dim + d;
  const int64_t cache_offset = ((h * cache_seq) + (start + t)) * head_dim + d;
  cache[cache_offset] = chunk[chunk_offset];
}

}  // namespace

bool HasCudaKvCacheKernel() {
  return true;
}

torch::Tensor CudaKvCacheWriteForward(
    const torch::Tensor& cache,
    const torch::Tensor& chunk,
    int64_t start) {
  TORCH_CHECK(cache.is_cuda() && chunk.is_cuda(), "CudaKvCacheWriteForward: tensors must be CUDA");
  TORCH_CHECK(cache.dim() == 3 && chunk.dim() == 3, "CudaKvCacheWriteForward: tensors must be rank-3");
  TORCH_CHECK(cache.scalar_type() == chunk.scalar_type(), "CudaKvCacheWriteForward: dtype mismatch");

  c10::cuda::CUDAGuard device_guard{cache.device()};

  auto cache_contig = cache.contiguous();
  auto chunk_contig = chunk.contiguous();
  const auto total = chunk_contig.numel();
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(cache.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      cache_contig.scalar_type(),
      "model_stack_cuda_kv_cache_write_forward",
      [&] {
        kv_cache_write_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            cache_contig.data_ptr<scalar_t>(),
            chunk_contig.data_ptr<scalar_t>(),
            cache_contig.size(0),
            cache_contig.size(1),
            chunk_contig.size(1),
            cache_contig.size(2),
            start);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return cache_contig;
}
