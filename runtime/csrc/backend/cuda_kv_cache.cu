#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t PackInt3Code(uint32_t word, int idx, int code) {
  return word | ((static_cast<uint32_t>(code) & 0x7u) << (idx * 3));
}

template <typename scalar_t>
__global__ void int3_pack_lastdim_kernel(
    const scalar_t* __restrict__ x,
    uint8_t* __restrict__ packed,
    float* __restrict__ scale,
    int64_t vectors,
    int64_t dim,
    int64_t packed_dim,
    int64_t groups) {
  const int64_t vector = static_cast<int64_t>(blockIdx.x);
  if (vector >= vectors) {
    return;
  }
  __shared__ float shared_max[kThreads];
  float local_max = 0.0f;
  const int64_t base = vector * dim;
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(static_cast<float>(x[base + d])));
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float s = shared_max[0] > 0.0f ? (shared_max[0] / 3.0f) : 1.0f;
  if (threadIdx.x == 0) {
    scale[vector] = s;
  }
  __syncthreads();

  for (int64_t group = threadIdx.x; group < groups; group += blockDim.x) {
    uint32_t word = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int64_t d = group * 8 + i;
      int code = 3;
      if (d < dim) {
        int q = __float2int_rn(static_cast<float>(x[base + d]) / s);
        q = q < -3 ? -3 : q;
        q = q > 3 ? 3 : q;
        code = q + 3;
      }
      word = PackInt3Code(word, i, code);
    }
    const int64_t out = vector * packed_dim + group * 3;
    packed[out + 0] = static_cast<uint8_t>(word & 0xFFu);
    packed[out + 1] = static_cast<uint8_t>((word >> 8) & 0xFFu);
    packed[out + 2] = static_cast<uint8_t>((word >> 16) & 0xFFu);
  }
}

template <typename scalar_t>
__global__ void int3_dequantize_lastdim_kernel(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scale,
    scalar_t* __restrict__ out,
    int64_t vectors,
    int64_t dim,
    int64_t packed_dim,
    int64_t groups) {
  const int64_t total = vectors * dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int64_t vector = idx / dim;
  const int64_t d = idx % dim;
  const int64_t group = d / 8;
  const int in_group = static_cast<int>(d % 8);
  const int64_t p = vector * packed_dim + group * 3;
  const uint32_t word = static_cast<uint32_t>(packed[p + 0]) |
      (static_cast<uint32_t>(packed[p + 1]) << 8) |
      (static_cast<uint32_t>(packed[p + 2]) << 16);
  const int code = static_cast<int>((word >> (in_group * 3)) & 0x7u);
  const int q = code - 3;
  out[idx] = static_cast<scalar_t>(static_cast<float>(q) * scale[vector]);
}

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

template <typename scalar_t, bool BatchedPositions>
__global__ void paged_kv_write_forward_strided_values_kernel(
    scalar_t* __restrict__ pages,
    const int64_t* __restrict__ block_table,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ values,
    int64_t batch,
    int64_t max_blocks,
    int64_t heads,
    int64_t page_size,
    int64_t write_seq,
    int64_t head_dim,
    int64_t value_s0,
    int64_t value_s1,
    int64_t value_s2,
    int64_t value_s3) {
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
  const int64_t value_offset = b * value_s0 + h * value_s1 + t * value_s2 + d * value_s3;
  pages[page_offset_linear] = values[value_offset];
}

template <typename scalar_t>
__global__ void projected_qkv_rotary_paged_write_kernel(
    const scalar_t* __restrict__ projected,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ k_pages,
    scalar_t* __restrict__ v_pages,
    const int64_t* __restrict__ block_table,
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ q_out,
    int64_t batch,
    int64_t max_blocks,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t page_size,
    int64_t head_dim,
    int64_t q_size,
    int64_t k_size,
    int64_t total_size) {
  const int64_t half = head_dim / 2;
  const int64_t q_elements = batch * q_heads * half;
  const int64_t k_elements = batch * kv_heads * half;
  const int64_t v_elements = batch * kv_heads * head_dim;
  const int64_t total = q_elements + k_elements + v_elements;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  if (idx < q_elements) {
    const int64_t col = idx % half;
    const int64_t h = (idx / half) % q_heads;
    const int64_t b = idx / (half * q_heads);
    const int64_t projected_base = b * total_size + h * head_dim;
    const int64_t out_base = ((b * q_heads + h) * head_dim);
    const float cos_lo = static_cast<float>(cos[col]);
    const float cos_hi = static_cast<float>(cos[half + col]);
    const float sin_lo = static_cast<float>(sin[col]);
    const float sin_hi = static_cast<float>(sin[half + col]);
    const float q_lo = static_cast<float>(projected[projected_base + col]);
    const float q_hi = static_cast<float>(projected[projected_base + half + col]);
    q_out[out_base + col] = static_cast<scalar_t>((q_lo * cos_lo) - (q_hi * sin_lo));
    q_out[out_base + half + col] = static_cast<scalar_t>((q_hi * cos_hi) + (q_lo * sin_hi));
    return;
  }

  const int64_t cache_idx = idx - q_elements;
  if (cache_idx < k_elements) {
    const int64_t col = cache_idx % half;
    const int64_t h = (cache_idx / half) % kv_heads;
    const int64_t b = cache_idx / (half * kv_heads);
    const int64_t pos = positions[b];
    const int64_t block_idx = pos / page_size;
    const int64_t page_offset = pos % page_size;
    const int64_t page_id = block_table[b * max_blocks + block_idx];
    const int64_t projected_base = b * total_size + q_size + h * head_dim;
    const int64_t page_base = (((page_id * kv_heads) + h) * page_size + page_offset) * head_dim;
    const float cos_lo = static_cast<float>(cos[col]);
    const float cos_hi = static_cast<float>(cos[half + col]);
    const float sin_lo = static_cast<float>(sin[col]);
    const float sin_hi = static_cast<float>(sin[half + col]);
    const float k_lo = static_cast<float>(projected[projected_base + col]);
    const float k_hi = static_cast<float>(projected[projected_base + half + col]);
    k_pages[page_base + col] = static_cast<scalar_t>((k_lo * cos_lo) - (k_hi * sin_lo));
    k_pages[page_base + half + col] = static_cast<scalar_t>((k_hi * cos_hi) + (k_lo * sin_hi));
    return;
  }

  const int64_t v_idx = cache_idx - k_elements;
  const int64_t d = v_idx % head_dim;
  const int64_t h = (v_idx / head_dim) % kv_heads;
  const int64_t b = v_idx / (head_dim * kv_heads);
  const int64_t pos = positions[b];
  const int64_t block_idx = pos / page_size;
  const int64_t page_offset = pos % page_size;
  const int64_t page_id = block_table[b * max_blocks + block_idx];
  const int64_t projected_offset = b * total_size + q_size + k_size + h * head_dim + d;
  const int64_t page_offset_linear = (((page_id * kv_heads) + h) * page_size + page_offset) * head_dim + d;
  v_pages[page_offset_linear] = projected[projected_offset];
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
  const bool values_fast_contig = values.is_contiguous();
  auto values_contig = values_fast_contig ? values : torch::Tensor();
  const auto& values_ref = values_fast_contig ? values_contig : values;
  const auto write_seq = positions_contig.dim() == 1 ? positions_contig.size(0) : positions_contig.size(1);
  const auto total = values_ref.numel();
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
          if (values_fast_contig) {
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
          } else {
            paged_kv_write_forward_strided_values_kernel<scalar_t, false><<<blocks, threads, 0, stream.stream()>>>(
                pages_contig.data_ptr<scalar_t>(),
                block_table_contig.data_ptr<int64_t>(),
                positions_contig.data_ptr<int64_t>(),
                values.data_ptr<scalar_t>(),
                block_table_contig.size(0),
                block_table_contig.size(1),
                pages_contig.size(1),
                pages_contig.size(2),
                write_seq,
                pages_contig.size(3),
                values.stride(0),
                values.stride(1),
                values.stride(2),
                values.stride(3));
          }
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        pages_contig.scalar_type(),
        "model_stack_cuda_paged_kv_write_forward_batched",
        [&] {
          if (values_fast_contig) {
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
          } else {
            paged_kv_write_forward_strided_values_kernel<scalar_t, true><<<blocks, threads, 0, stream.stream()>>>(
                pages_contig.data_ptr<scalar_t>(),
                block_table_contig.data_ptr<int64_t>(),
                positions_contig.data_ptr<int64_t>(),
                values.data_ptr<scalar_t>(),
                block_table_contig.size(0),
                block_table_contig.size(1),
                pages_contig.size(1),
                pages_contig.size(2),
                write_seq,
                pages_contig.size(3),
                values.stride(0),
                values.stride(1),
                values.stride(2),
                values.stride(3));
          }
        });
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return pages_contig;
}

std::vector<torch::Tensor> CudaProjectedQkvRotaryPagedWriteForward(
    const torch::Tensor& projected,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads) {
  TORCH_CHECK(projected.is_cuda() && cos.is_cuda() && sin.is_cuda() && k_pages.is_cuda() &&
                  v_pages.is_cuda() && block_table.is_cuda() && positions.is_cuda(),
              "CudaProjectedQkvRotaryPagedWriteForward: tensors must be CUDA");
  TORCH_CHECK(projected.dim() == 3 && projected.size(1) == 1,
              "CudaProjectedQkvRotaryPagedWriteForward: projected must have shape (B,1,QKV)");
  TORCH_CHECK(k_pages.dim() == 4 && v_pages.dim() == 4,
              "CudaProjectedQkvRotaryPagedWriteForward: pages must be rank-4");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(),
              "CudaProjectedQkvRotaryPagedWriteForward: K/V pages shape mismatch");
  TORCH_CHECK(block_table.dim() == 2, "CudaProjectedQkvRotaryPagedWriteForward: block_table must be rank-2");
  TORCH_CHECK(positions.dim() == 1 || (positions.dim() == 2 && positions.size(1) == 1),
              "CudaProjectedQkvRotaryPagedWriteForward: positions must be rank-1 or (B,1)");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2 && cos.sizes() == sin.sizes(),
              "CudaProjectedQkvRotaryPagedWriteForward: cos/sin must be matching rank-2 tensors");
  TORCH_CHECK(projected.scalar_type() == k_pages.scalar_type() &&
                  projected.scalar_type() == v_pages.scalar_type() &&
                  projected.scalar_type() == cos.scalar_type() &&
                  projected.scalar_type() == sin.scalar_type(),
              "CudaProjectedQkvRotaryPagedWriteForward: dtype mismatch");
  TORCH_CHECK(q_heads > 0 && kv_heads > 0, "CudaProjectedQkvRotaryPagedWriteForward: head counts must be positive");
  TORCH_CHECK(q_size + k_size + v_size == projected.size(2),
              "CudaProjectedQkvRotaryPagedWriteForward: projected QKV size mismatch");
  TORCH_CHECK(q_size % q_heads == 0 && k_size % kv_heads == 0 && v_size % kv_heads == 0,
              "CudaProjectedQkvRotaryPagedWriteForward: sizes must divide by head counts");
  const auto head_dim = q_size / q_heads;
  TORCH_CHECK(head_dim > 0 && (head_dim % 2) == 0,
              "CudaProjectedQkvRotaryPagedWriteForward: head_dim must be positive and even");
  TORCH_CHECK(k_size / kv_heads == head_dim && v_size / kv_heads == head_dim,
              "CudaProjectedQkvRotaryPagedWriteForward: Q/K/V head_dim mismatch");
  TORCH_CHECK(k_pages.size(1) == kv_heads && k_pages.size(3) == head_dim,
              "CudaProjectedQkvRotaryPagedWriteForward: page head shape mismatch");
  TORCH_CHECK(block_table.size(0) == projected.size(0),
              "CudaProjectedQkvRotaryPagedWriteForward: block_table batch mismatch");
  TORCH_CHECK(positions.numel() == projected.size(0),
              "CudaProjectedQkvRotaryPagedWriteForward: positions batch mismatch");
  TORCH_CHECK(cos.size(0) == 1 && cos.size(1) == head_dim,
              "CudaProjectedQkvRotaryPagedWriteForward: single-token cos/sin row required");

  c10::cuda::CUDAGuard device_guard{projected.device()};

  auto projected_contig = projected.contiguous();
  auto cos_contig = cos.contiguous();
  auto sin_contig = sin.contiguous();
  auto k_pages_contig = k_pages.contiguous();
  auto v_pages_contig = v_pages.contiguous();
  auto block_table_contig = block_table.to(torch::kLong).contiguous();
  auto positions_contig = positions.to(torch::kLong).contiguous().view({projected_contig.size(0)});
  auto q_out = torch::empty({projected_contig.size(0), q_heads, 1, head_dim}, projected_contig.options());

  const int64_t total = projected_contig.size(0) * q_heads * (head_dim / 2) +
      projected_contig.size(0) * kv_heads * (head_dim / 2) +
      projected_contig.size(0) * kv_heads * head_dim;
  if (total == 0) {
    return {q_out, k_pages_contig, v_pages_contig};
  }
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(projected.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      projected_contig.scalar_type(),
      "model_stack_cuda_projected_qkv_rotary_paged_write_forward",
      [&] {
        projected_qkv_rotary_paged_write_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            projected_contig.data_ptr<scalar_t>(),
            cos_contig.data_ptr<scalar_t>(),
            sin_contig.data_ptr<scalar_t>(),
            k_pages_contig.data_ptr<scalar_t>(),
            v_pages_contig.data_ptr<scalar_t>(),
            block_table_contig.data_ptr<int64_t>(),
            positions_contig.data_ptr<int64_t>(),
            q_out.data_ptr<scalar_t>(),
            projected_contig.size(0),
            block_table_contig.size(1),
            q_heads,
            kv_heads,
            k_pages_contig.size(2),
            head_dim,
            q_size,
            k_size,
            projected_contig.size(2));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_pages_contig, v_pages_contig};
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

std::vector<torch::Tensor> CudaInt3PackLastDimForward(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "CudaInt3PackLastDimForward: x must be CUDA");
  TORCH_CHECK(x.dim() >= 1, "CudaInt3PackLastDimForward: x must have rank >= 1");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "CudaInt3PackLastDimForward: x must be float32, float16, or bfloat16");
  c10::cuda::CUDAGuard device_guard{x.device()};
  auto x_contig = x.contiguous();
  const auto dim = x_contig.size(-1);
  TORCH_CHECK(dim > 0, "CudaInt3PackLastDimForward: last dim must be non-empty");
  const auto vectors = x_contig.numel() / dim;
  const auto groups = (dim + 7) / 8;
  const auto packed_dim = groups * 3;
  std::vector<int64_t> packed_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  packed_sizes.back() = packed_dim;
  std::vector<int64_t> scale_sizes(x_contig.sizes().begin(), x_contig.sizes().end() - 1);
  auto packed = torch::empty(packed_sizes, x_contig.options().dtype(torch::kUInt8));
  auto scale = torch::empty(scale_sizes, x_contig.options().dtype(torch::kFloat32));
  if (vectors == 0) {
    return {packed, scale};
  }
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const dim3 blocks(static_cast<unsigned int>(vectors));
  const dim3 threads(kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "model_stack_cuda_int3_pack_lastdim",
      [&] {
        int3_pack_lastdim_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_contig.data_ptr<scalar_t>(),
            packed.data_ptr<uint8_t>(),
            scale.data_ptr<float>(),
            vectors,
            dim,
            packed_dim,
            groups);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scale};
}

torch::Tensor CudaInt3DequantizeLastDimForward(
    const torch::Tensor& packed,
    const torch::Tensor& scale,
    int64_t original_last_dim,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(packed.is_cuda() && scale.is_cuda(), "CudaInt3DequantizeLastDimForward: tensors must be CUDA");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "CudaInt3DequantizeLastDimForward: packed must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "CudaInt3DequantizeLastDimForward: scale must be float32");
  TORCH_CHECK(packed.dim() >= 1, "CudaInt3DequantizeLastDimForward: packed must have rank >= 1");
  TORCH_CHECK(original_last_dim > 0, "CudaInt3DequantizeLastDimForward: original_last_dim must be positive");
  TORCH_CHECK(packed.size(-1) % 3 == 0, "CudaInt3DequantizeLastDimForward: packed last dim must be divisible by 3");
  c10::cuda::CUDAGuard device_guard{packed.device()};
  auto packed_contig = packed.contiguous();
  auto scale_contig = scale.contiguous();
  const auto packed_dim = packed_contig.size(-1);
  const auto groups = packed_dim / 3;
  const auto vectors = packed_contig.numel() / packed_dim;
  TORCH_CHECK(scale_contig.numel() == vectors, "CudaInt3DequantizeLastDimForward: scale shape mismatch");
  std::vector<int64_t> out_sizes(packed_contig.sizes().begin(), packed_contig.sizes().end());
  out_sizes.back() = original_last_dim;
  const auto dtype = out_dtype.has_value() ? out_dtype.value() : torch::kFloat32;
  auto out = torch::empty(out_sizes, packed_contig.options().dtype(dtype));
  if (vectors == 0) {
    return out;
  }
  auto stream = c10::cuda::getCurrentCUDAStream(packed.get_device());
  const auto total = vectors * original_last_dim;
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "model_stack_cuda_int3_dequantize_lastdim",
      [&] {
        int3_dequantize_lastdim_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            packed_contig.data_ptr<uint8_t>(),
            scale_contig.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            vectors,
            original_last_dim,
            packed_dim,
            groups);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
