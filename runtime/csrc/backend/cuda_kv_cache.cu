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

template <typename scalar_t>
__global__ void kv_cache_gather_forward_kernel(
    const scalar_t* __restrict__ cache,
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ out,
    int64_t heads,
    int64_t cache_seq,
    int64_t gather_seq,
    int64_t head_dim) {
  const int64_t total = heads * gather_seq * head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % head_dim;
  const int64_t t = (idx / head_dim) % gather_seq;
  const int64_t h = idx / (head_dim * gather_seq);
  const int64_t pos = positions[t];

  const int64_t cache_offset = ((h * cache_seq) + pos) * head_dim + d;
  const int64_t out_offset = ((h * gather_seq) + t) * head_dim + d;
  out[out_offset] = cache[cache_offset];
}

template <typename scalar_t, bool BatchedPositions>
__global__ void kv_cache_gather_bhsd_forward_kernel(
    const scalar_t* __restrict__ cache,
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t heads,
    int64_t cache_seq,
    int64_t gather_seq,
    int64_t head_dim) {
  const int64_t total = batch * heads * gather_seq * head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % head_dim;
  const int64_t t = (idx / head_dim) % gather_seq;
  const int64_t h = (idx / (head_dim * gather_seq)) % heads;
  const int64_t b = idx / (head_dim * gather_seq * heads);
  const int64_t pos = BatchedPositions ? positions[b * gather_seq + t] : positions[t];

  const int64_t cache_offset = ((((b * heads) + h) * cache_seq) + pos) * head_dim + d;
  const int64_t out_offset = ((((b * heads) + h) * gather_seq) + t) * head_dim + d;
  out[out_offset] = cache[cache_offset];
}

template <typename scalar_t, bool BatchedPositions>
__global__ void paged_kv_gather_forward_kernel(
    const scalar_t* __restrict__ pages,
    const int64_t* __restrict__ block_table,
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t max_blocks,
    int64_t heads,
    int64_t page_size,
    int64_t gather_seq,
    int64_t head_dim) {
  const int64_t total = batch * heads * gather_seq * head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % head_dim;
  const int64_t t = (idx / head_dim) % gather_seq;
  const int64_t h = (idx / (head_dim * gather_seq)) % heads;
  const int64_t b = idx / (head_dim * gather_seq * heads);
  const int64_t pos = BatchedPositions ? positions[b * gather_seq + t] : positions[t];
  const int64_t block_idx = pos / page_size;
  const int64_t page_offset = pos % page_size;
  const int64_t page_id = block_table[b * max_blocks + block_idx];

  const int64_t page_offset_linear = (((page_id * heads) + h) * page_size + page_offset) * head_dim + d;
  const int64_t out_offset = ((((b * heads) + h) * gather_seq) + t) * head_dim + d;
  out[out_offset] = pages[page_offset_linear];
}

template <typename scalar_t, bool BatchedPositions>
__global__ void paged_kv_write_forward_kernel(
    scalar_t* __restrict__ pages,
    const int64_t* __restrict__ block_table,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ values,
    int64_t batch,
    int64_t max_blocks,
    int64_t heads,
    int64_t page_size,
    int64_t write_seq,
    int64_t head_dim) {
  const int64_t total = batch * heads * write_seq * head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % head_dim;
  const int64_t t = (idx / head_dim) % write_seq;
  const int64_t h = (idx / (head_dim * write_seq)) % heads;
  const int64_t b = idx / (head_dim * write_seq * heads);
  const int64_t pos = BatchedPositions ? positions[b * write_seq + t] : positions[t];
  const int64_t block_idx = pos / page_size;
  const int64_t page_offset = pos % page_size;
  const int64_t page_id = block_table[b * max_blocks + block_idx];

  const int64_t page_offset_linear = (((page_id * heads) + h) * page_size + page_offset) * head_dim + d;
  const int64_t value_offset = ((((b * heads) + h) * write_seq) + t) * head_dim + d;
  pages[page_offset_linear] = values[value_offset];
}

template <typename scalar_t>
__global__ void paged_kv_read_last_forward_kernel(
    const scalar_t* __restrict__ pages,
    const int64_t* __restrict__ block_table,
    const int64_t* __restrict__ lengths,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t max_blocks,
    int64_t heads,
    int64_t page_size,
    int64_t keep,
    int64_t max_keep,
    int64_t head_dim) {
  const int64_t total = batch * heads * max_keep * head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % head_dim;
  const int64_t t = (idx / head_dim) % max_keep;
  const int64_t h = (idx / (head_dim * max_keep)) % heads;
  const int64_t b = idx / (head_dim * max_keep * heads);
  const int64_t live_len = lengths[b];
  const int64_t row_keep = live_len < keep ? live_len : keep;
  const int64_t out_offset = ((((b * heads) + h) * max_keep) + t) * head_dim + d;
  if (t >= row_keep) {
    out[out_offset] = static_cast<scalar_t>(0);
    return;
  }

  const int64_t start = live_len - row_keep;
  const int64_t pos = start + t;
  const int64_t block_idx = pos / page_size;
  const int64_t page_offset = pos % page_size;
  const int64_t page_id = block_table[b * max_blocks + block_idx];
  const int64_t page_offset_linear = (((page_id * heads) + h) * page_size + page_offset) * head_dim + d;
  out[out_offset] = pages[page_offset_linear];
}

template <typename scalar_t>
__global__ void paged_kv_read_range_forward_kernel(
    const scalar_t* __restrict__ pages,
    const int64_t* __restrict__ block_table,
    const int64_t* __restrict__ lengths,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t max_blocks,
    int64_t heads,
    int64_t page_size,
    int64_t start,
    int64_t gather_seq,
    int64_t head_dim) {
  const int64_t total = batch * heads * gather_seq * head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % head_dim;
  const int64_t t = (idx / head_dim) % gather_seq;
  const int64_t h = (idx / (head_dim * gather_seq)) % heads;
  const int64_t b = idx / (head_dim * gather_seq * heads);
  const int64_t pos = start + t;
  const int64_t out_offset = ((((b * heads) + h) * gather_seq) + t) * head_dim + d;
  if (pos >= lengths[b]) {
    out[out_offset] = static_cast<scalar_t>(0);
    return;
  }
  const int64_t block_idx = pos / page_size;
  const int64_t page_offset = pos % page_size;
  const int64_t page_id = block_table[b * max_blocks + block_idx];
  const int64_t page_offset_linear = (((page_id * heads) + h) * page_size + page_offset) * head_dim + d;
  out[out_offset] = pages[page_offset_linear];
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

torch::Tensor CudaKvCacheGatherForward(
    const torch::Tensor& cache,
    const torch::Tensor& positions) {
  TORCH_CHECK(cache.is_cuda() && positions.is_cuda(), "CudaKvCacheGatherForward: tensors must be CUDA");
  TORCH_CHECK(cache.dim() == 3 || cache.dim() == 4, "CudaKvCacheGatherForward: cache must be rank-3 or rank-4");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2,
              "CudaKvCacheGatherForward: positions must be rank-1 or rank-2");

  c10::cuda::CUDAGuard device_guard{cache.device()};

  auto cache_contig = cache.contiguous();
  auto positions_contig = positions.to(torch::kLong).contiguous();
  auto stream = c10::cuda::getCurrentCUDAStream(cache.get_device());

  if (cache_contig.dim() == 3) {
    TORCH_CHECK(positions_contig.dim() == 1, "CudaKvCacheGatherForward: rank-3 cache requires rank-1 positions");
    auto out = torch::empty(
        {cache_contig.size(0), positions_contig.size(0), cache_contig.size(2)},
        cache_contig.options());
    const auto total = out.numel();
    const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
    const dim3 threads(kThreads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        cache_contig.scalar_type(),
        "model_stack_cuda_kv_cache_gather_forward_hsd",
        [&] {
          kv_cache_gather_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
              cache_contig.data_ptr<scalar_t>(),
              positions_contig.data_ptr<int64_t>(),
              out.data_ptr<scalar_t>(),
              cache_contig.size(0),
              cache_contig.size(1),
              positions_contig.size(0),
              cache_contig.size(2));
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  const auto gather_seq = positions_contig.dim() == 1 ? positions_contig.size(0) : positions_contig.size(1);
  auto out = torch::empty(
      {cache_contig.size(0), cache_contig.size(1), gather_seq, cache_contig.size(3)},
      cache_contig.options());
  const auto total = out.numel();
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);

  if (positions_contig.dim() == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        cache_contig.scalar_type(),
        "model_stack_cuda_kv_cache_gather_forward_bhsd_shared",
        [&] {
          kv_cache_gather_bhsd_forward_kernel<scalar_t, false><<<blocks, threads, 0, stream.stream()>>>(
              cache_contig.data_ptr<scalar_t>(),
              positions_contig.data_ptr<int64_t>(),
              out.data_ptr<scalar_t>(),
              cache_contig.size(0),
              cache_contig.size(1),
              cache_contig.size(2),
              gather_seq,
              cache_contig.size(3));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        cache_contig.scalar_type(),
        "model_stack_cuda_kv_cache_gather_forward_bhsd_batched",
        [&] {
          kv_cache_gather_bhsd_forward_kernel<scalar_t, true><<<blocks, threads, 0, stream.stream()>>>(
              cache_contig.data_ptr<scalar_t>(),
              positions_contig.data_ptr<int64_t>(),
              out.data_ptr<scalar_t>(),
              cache_contig.size(0),
              cache_contig.size(1),
              cache_contig.size(2),
              gather_seq,
              cache_contig.size(3));
        });
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor CudaPagedKvGatherForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions) {
  TORCH_CHECK(pages.is_cuda() && block_table.is_cuda() && positions.is_cuda(),
              "CudaPagedKvGatherForward: tensors must be CUDA");
  TORCH_CHECK(pages.dim() == 4, "CudaPagedKvGatherForward: pages must be rank-4");
  TORCH_CHECK(block_table.dim() == 2, "CudaPagedKvGatherForward: block_table must be rank-2");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2,
              "CudaPagedKvGatherForward: positions must be rank-1 or rank-2");

  c10::cuda::CUDAGuard device_guard{pages.device()};

  auto pages_contig = pages.contiguous();
  auto block_table_contig = block_table.to(torch::kLong).contiguous();
  auto positions_contig = positions.to(torch::kLong).contiguous();
  const auto gather_seq = positions_contig.dim() == 1 ? positions_contig.size(0) : positions_contig.size(1);
  auto out = torch::empty(
      {block_table_contig.size(0), pages_contig.size(1), gather_seq, pages_contig.size(3)},
      pages_contig.options());
  const auto total = out.numel();
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(pages.get_device());

  if (positions_contig.dim() == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        pages_contig.scalar_type(),
        "model_stack_cuda_paged_kv_gather_forward_shared",
        [&] {
          paged_kv_gather_forward_kernel<scalar_t, false><<<blocks, threads, 0, stream.stream()>>>(
              pages_contig.data_ptr<scalar_t>(),
              block_table_contig.data_ptr<int64_t>(),
              positions_contig.data_ptr<int64_t>(),
              out.data_ptr<scalar_t>(),
              block_table_contig.size(0),
              block_table_contig.size(1),
              pages_contig.size(1),
              pages_contig.size(2),
              gather_seq,
              pages_contig.size(3));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        pages_contig.scalar_type(),
        "model_stack_cuda_paged_kv_gather_forward_batched",
        [&] {
          paged_kv_gather_forward_kernel<scalar_t, true><<<blocks, threads, 0, stream.stream()>>>(
              pages_contig.data_ptr<scalar_t>(),
              block_table_contig.data_ptr<int64_t>(),
              positions_contig.data_ptr<int64_t>(),
              out.data_ptr<scalar_t>(),
              block_table_contig.size(0),
              block_table_contig.size(1),
              pages_contig.size(1),
              pages_contig.size(2),
              gather_seq,
              pages_contig.size(3));
        });
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor CudaPagedKvWriteForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions,
    const torch::Tensor& values) {
  TORCH_CHECK(pages.is_cuda() && block_table.is_cuda() && positions.is_cuda() && values.is_cuda(),
              "CudaPagedKvWriteForward: tensors must be CUDA");
  TORCH_CHECK(pages.dim() == 4, "CudaPagedKvWriteForward: pages must be rank-4");
  TORCH_CHECK(block_table.dim() == 2, "CudaPagedKvWriteForward: block_table must be rank-2");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2,
              "CudaPagedKvWriteForward: positions must be rank-1 or rank-2");
  TORCH_CHECK(values.dim() == 4, "CudaPagedKvWriteForward: values must be rank-4");

  c10::cuda::CUDAGuard device_guard{pages.device()};

  auto pages_contig = pages.contiguous();
  auto block_table_contig = block_table.to(torch::kLong).contiguous();
  auto positions_contig = positions.to(torch::kLong).contiguous();
  auto values_contig = values.contiguous();
  const auto write_seq = positions_contig.dim() == 1 ? positions_contig.size(0) : positions_contig.size(1);
  const auto total = values_contig.numel();
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(pages.get_device());

  if (positions_contig.dim() == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        pages_contig.scalar_type(),
        "model_stack_cuda_paged_kv_write_forward_shared",
        [&] {
          paged_kv_write_forward_kernel<scalar_t, false><<<blocks, threads, 0, stream.stream()>>>(
              pages_contig.data_ptr<scalar_t>(),
              block_table_contig.data_ptr<int64_t>(),
              positions_contig.data_ptr<int64_t>(),
              values_contig.data_ptr<scalar_t>(),
              block_table_contig.size(0),
              block_table_contig.size(1),
              pages_contig.size(1),
              pages_contig.size(2),
              write_seq,
              pages_contig.size(3));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        pages_contig.scalar_type(),
        "model_stack_cuda_paged_kv_write_forward_batched",
        [&] {
          paged_kv_write_forward_kernel<scalar_t, true><<<blocks, threads, 0, stream.stream()>>>(
              pages_contig.data_ptr<scalar_t>(),
              block_table_contig.data_ptr<int64_t>(),
              positions_contig.data_ptr<int64_t>(),
              values_contig.data_ptr<scalar_t>(),
              block_table_contig.size(0),
              block_table_contig.size(1),
              pages_contig.size(1),
              pages_contig.size(2),
              write_seq,
              pages_contig.size(3));
        });
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return pages_contig;
}

std::vector<torch::Tensor> CudaPagedKvReadLastForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t keep) {
  TORCH_CHECK(pages.is_cuda() && block_table.is_cuda() && lengths.is_cuda(),
              "CudaPagedKvReadLastForward: tensors must be CUDA");
  TORCH_CHECK(pages.dim() == 4, "CudaPagedKvReadLastForward: pages must be rank-4");
  TORCH_CHECK(block_table.dim() == 2, "CudaPagedKvReadLastForward: block_table must be rank-2");
  TORCH_CHECK(lengths.dim() == 1, "CudaPagedKvReadLastForward: lengths must be rank-1");

  c10::cuda::CUDAGuard device_guard{pages.device()};

  auto pages_contig = pages.contiguous();
  auto block_table_contig = block_table.to(torch::kLong).contiguous();
  auto lengths_contig = lengths.to(torch::kLong).contiguous();
  auto kept_lengths = torch::clamp(lengths_contig, 0, keep);
  const auto max_keep = kept_lengths.numel() > 0 ? kept_lengths.max().item<int64_t>() : 0;
  auto out = torch::zeros(
      {block_table_contig.size(0), pages_contig.size(1), max_keep, pages_contig.size(3)},
      pages_contig.options());
  if (block_table_contig.size(0) == 0 || max_keep == 0) {
    return {out, kept_lengths};
  }

  const auto total = out.numel();
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(pages.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      pages_contig.scalar_type(),
      "model_stack_cuda_paged_kv_read_last_forward",
      [&] {
        paged_kv_read_last_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            pages_contig.data_ptr<scalar_t>(),
            block_table_contig.data_ptr<int64_t>(),
            lengths_contig.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            block_table_contig.size(0),
            block_table_contig.size(1),
            pages_contig.size(1),
            pages_contig.size(2),
            keep,
            max_keep,
            pages_contig.size(3));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, kept_lengths};
}

torch::Tensor CudaPagedKvReadRangeForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t start,
    int64_t end) {
  TORCH_CHECK(pages.is_cuda() && block_table.is_cuda() && lengths.is_cuda(),
              "CudaPagedKvReadRangeForward: tensors must be CUDA");
  TORCH_CHECK(pages.dim() == 4, "CudaPagedKvReadRangeForward: pages must be rank-4");
  TORCH_CHECK(block_table.dim() == 2, "CudaPagedKvReadRangeForward: block_table must be rank-2");
  TORCH_CHECK(lengths.dim() == 1, "CudaPagedKvReadRangeForward: lengths must be rank-1");

  c10::cuda::CUDAGuard device_guard{pages.device()};

  auto pages_contig = pages.contiguous();
  auto block_table_contig = block_table.to(torch::kLong).contiguous();
  auto lengths_contig = lengths.to(torch::kLong).contiguous();
  const auto gather_seq = end - start;
  auto out = torch::zeros(
      {block_table_contig.size(0), pages_contig.size(1), gather_seq, pages_contig.size(3)},
      pages_contig.options());
  if (block_table_contig.size(0) == 0 || gather_seq == 0) {
    return out;
  }

  const auto total = out.numel();
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(pages.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      pages_contig.scalar_type(),
      "model_stack_cuda_paged_kv_read_range_forward",
      [&] {
        paged_kv_read_range_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            pages_contig.data_ptr<scalar_t>(),
            block_table_contig.data_ptr<int64_t>(),
            lengths_contig.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            block_table_contig.size(0),
            block_table_contig.size(1),
            pages_contig.size(1),
            pages_contig.size(2),
            start,
            gather_seq,
            pages_contig.size(3));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
