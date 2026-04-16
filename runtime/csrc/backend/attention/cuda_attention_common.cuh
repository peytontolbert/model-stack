#pragma once

#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace t10::cuda::attention {

constexpr int kSmallRowThreads = 64;
constexpr int kMediumRowThreads = 128;
constexpr int kLargeRowThreads = 256;

template <typename scalar_t, int Threads, bool HasMask>
__global__ void generic_attention_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const void* __restrict__ mask,
    int mask_kind,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t tq,
    int64_t sk,
    int64_t dh,
    float scale,
    bool is_causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch * q_heads * tq;
  if (row >= total_rows) {
    return;
  }

  __shared__ float shared[Threads];

  const int64_t query_idx = row % tq;
  const int64_t tmp = row / tq;
  const int64_t head_idx = tmp % q_heads;
  const int64_t batch_idx = tmp / q_heads;
  const int64_t head_group = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_group;

  const int64_t q_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * dh;
  const int64_t kv_base = ((batch_idx * kv_heads) + kv_head_idx) * sk * dh;
  const int64_t mask_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * sk;

  float local_max = -INFINITY;
  for (int64_t s = threadIdx.x; s < sk; s += blockDim.x) {
    float score = 0.0f;
    const int64_t k_base = kv_base + s * dh;
    for (int64_t d = 0; d < dh; ++d) {
      score += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
    }
    score *= scale;
    if constexpr (HasMask) {
      if (mask_kind == 1) {
        if (static_cast<const bool*>(mask)[mask_base + s]) {
          score = -INFINITY;
        }
      } else if (mask_kind == 2) {
        score += static_cast<float>(static_cast<const scalar_t*>(mask)[mask_base + s]);
      } else if (mask_kind == 3) {
        score += static_cast<const float*>(mask)[mask_base + s];
      }
    }
    if (is_causal && s > query_idx) {
      score = -INFINITY;
    }
    local_max = fmaxf(local_max, score);
  }

  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = shared[0];

  float local_sum = 0.0f;
  for (int64_t s = threadIdx.x; s < sk; s += blockDim.x) {
    float score = 0.0f;
    const int64_t k_base = kv_base + s * dh;
    for (int64_t d = 0; d < dh; ++d) {
      score += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
    }
    score *= scale;
    if constexpr (HasMask) {
      if (mask_kind == 1) {
        if (static_cast<const bool*>(mask)[mask_base + s]) {
          score = -INFINITY;
        }
      } else if (mask_kind == 2) {
        score += static_cast<float>(static_cast<const scalar_t*>(mask)[mask_base + s]);
      } else if (mask_kind == 3) {
        score += static_cast<const float*>(mask)[mask_base + s];
      }
    }
    if (is_causal && s > query_idx) {
      score = -INFINITY;
    }
    local_sum += expf(score - row_max);
  }

  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float denom = fmaxf(shared[0], 1e-20f);

  for (int64_t d = threadIdx.x; d < dh; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t s = 0; s < sk; ++s) {
      float score = 0.0f;
      const int64_t k_base = kv_base + s * dh;
      for (int64_t kk = 0; kk < dh; ++kk) {
        score += static_cast<float>(q[q_base + kk]) * static_cast<float>(k[k_base + kk]);
      }
      score *= scale;
      if constexpr (HasMask) {
        if (mask_kind == 1) {
          if (static_cast<const bool*>(mask)[mask_base + s]) {
            score = -INFINITY;
          }
        } else if (mask_kind == 2) {
          score += static_cast<float>(static_cast<const scalar_t*>(mask)[mask_base + s]);
        } else if (mask_kind == 3) {
          score += static_cast<const float*>(mask)[mask_base + s];
        }
      }
      if (is_causal && s > query_idx) {
        score = -INFINITY;
      }
      const float weight = expf(score - row_max) / denom;
      acc += weight * static_cast<float>(v[kv_base + s * dh + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
inline void LaunchGenericAttentionKernel(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const void* mask_ptr,
    int mask_kind,
    const torch::Tensor& out,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t q_len,
    int64_t kv_len,
    int64_t head_dim,
    float scale_value,
    bool is_causal,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(batch * q_heads * q_len));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      generic_attention_forward_kernel<scalar_t, kSmallRowThreads, true>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              batch,
              q_heads,
              kv_heads,
              q_len,
              kv_len,
              head_dim,
              scale_value,
              is_causal);
      return;
    case kMediumRowThreads:
      generic_attention_forward_kernel<scalar_t, kMediumRowThreads, true>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              batch,
              q_heads,
              kv_heads,
              q_len,
              kv_len,
              head_dim,
              scale_value,
              is_causal);
      return;
    default:
      generic_attention_forward_kernel<scalar_t, kLargeRowThreads, true>
          <<<blocks, kLargeRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              batch,
              q_heads,
              kv_heads,
              q_len,
              kv_len,
              head_dim,
              scale_value,
              is_causal);
      return;
  }
}

template <typename scalar_t>
inline void LaunchGenericAttentionKernelNoMask(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t q_len,
    int64_t kv_len,
    int64_t head_dim,
    float scale_value,
    bool is_causal,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(batch * q_heads * q_len));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      generic_attention_forward_kernel<scalar_t, kSmallRowThreads, false>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              batch,
              q_heads,
              kv_heads,
              q_len,
              kv_len,
              head_dim,
              scale_value,
              is_causal);
      return;
    case kMediumRowThreads:
      generic_attention_forward_kernel<scalar_t, kMediumRowThreads, false>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              batch,
              q_heads,
              kv_heads,
              q_len,
              kv_len,
              head_dim,
              scale_value,
              is_causal);
      return;
    default:
      generic_attention_forward_kernel<scalar_t, kLargeRowThreads, false>
          <<<blocks, kLargeRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              batch,
              q_heads,
              kv_heads,
              q_len,
              kv_len,
              head_dim,
              scale_value,
              is_causal);
      return;
  }
}

}  // namespace t10::cuda::attention
