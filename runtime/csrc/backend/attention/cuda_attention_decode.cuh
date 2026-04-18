#pragma once

#include "cuda_attention_common.cuh"
#include "../cuda_device_arch.cuh"

#include "../../descriptors/attention_desc.h"
#include "../../policy/attention_policy.h"

namespace t10::cuda::attention {

template <typename scalar_t, int Threads, bool HasMask>
__global__ void decode_attention_q1_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const void* __restrict__ mask,
    int mask_kind,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t sk,
    int64_t dh,
    float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch * q_heads;
  if (row >= total_rows) {
    return;
  }

  __shared__ float shared[Threads];

  const int64_t head_idx = row % q_heads;
  const int64_t batch_idx = row / q_heads;
  const int64_t head_group = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_group;

  const int64_t q_base = ((batch_idx * q_heads) + head_idx) * dh;
  const int64_t kv_base = ((batch_idx * kv_heads) + kv_head_idx) * sk * dh;
  const int64_t mask_base = ((batch_idx * q_heads) + head_idx) * sk;

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
    local_max = fmaxf(local_max, score);
  }

  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = Threads / 2; stride > 0; stride /= 2) {
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
    local_sum += expf(score - row_max);
  }

  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = Threads / 2; stride > 0; stride /= 2) {
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
      const float weight = expf(score - row_max) / denom;
      acc += weight * static_cast<float>(v[kv_base + s * dh + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, int Threads, bool HasMask>
__global__ void decode_attention_q1_mha_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const void* __restrict__ mask,
    int mask_kind,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t q_heads,
    int64_t sk,
    int64_t dh,
    float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch * q_heads;
  if (row >= total_rows) {
    return;
  }

  __shared__ float shared[Threads];

  const int64_t head_idx = row % q_heads;
  const int64_t batch_idx = row / q_heads;

  const int64_t q_base = ((batch_idx * q_heads) + head_idx) * dh;
  const int64_t kv_base = ((batch_idx * q_heads) + head_idx) * sk * dh;
  const int64_t mask_base = ((batch_idx * q_heads) + head_idx) * sk;

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
    local_max = fmaxf(local_max, score);
  }

  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = Threads / 2; stride > 0; stride /= 2) {
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
    local_sum += expf(score - row_max);
  }

  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = Threads / 2; stride > 0; stride /= 2) {
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
      const float weight = expf(score - row_max) / denom;
      acc += weight * static_cast<float>(v[kv_base + s * dh + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, int Threads, int HeadDim>
__global__ void decode_attention_q1_hdim_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const void* __restrict__ mask,
    int mask_kind,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t sk,
    float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch * q_heads;
  if (row >= total_rows) {
    return;
  }

  __shared__ float shared[Threads];

  const int64_t head_idx = row % q_heads;
  const int64_t batch_idx = row / q_heads;
  const int64_t head_group = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_group;

  const int64_t q_base = ((batch_idx * q_heads) + head_idx) * HeadDim;
  const int64_t kv_base = ((batch_idx * kv_heads) + kv_head_idx) * sk * HeadDim;
  const int64_t mask_base = ((batch_idx * q_heads) + head_idx) * sk;

  float local_max = -INFINITY;
  for (int64_t s = threadIdx.x; s < sk; s += blockDim.x) {
    float score = 0.0f;
    const int64_t k_base = kv_base + s * HeadDim;
    #pragma unroll
    for (int d = 0; d < HeadDim; ++d) {
      score += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
    }
    score *= scale;
    if (mask_kind == 1) {
      if (static_cast<const bool*>(mask)[mask_base + s]) {
        score = -INFINITY;
      }
    } else if (mask_kind == 2) {
      score += static_cast<float>(static_cast<const scalar_t*>(mask)[mask_base + s]);
    } else if (mask_kind == 3) {
      score += static_cast<const float*>(mask)[mask_base + s];
    }
    local_max = fmaxf(local_max, score);
  }

  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = shared[0];

  float local_sum = 0.0f;
  for (int64_t s = threadIdx.x; s < sk; s += blockDim.x) {
    float score = 0.0f;
    const int64_t k_base = kv_base + s * HeadDim;
    #pragma unroll
    for (int d = 0; d < HeadDim; ++d) {
      score += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
    }
    score *= scale;
    if (mask_kind == 1) {
      if (static_cast<const bool*>(mask)[mask_base + s]) {
        score = -INFINITY;
      }
    } else if (mask_kind == 2) {
      score += static_cast<float>(static_cast<const scalar_t*>(mask)[mask_base + s]);
    } else if (mask_kind == 3) {
      score += static_cast<const float*>(mask)[mask_base + s];
    }
    local_sum += expf(score - row_max);
  }

  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float denom = fmaxf(shared[0], 1e-20f);

  for (int64_t d = threadIdx.x; d < HeadDim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t s = 0; s < sk; ++s) {
      float score = 0.0f;
      const int64_t k_base = kv_base + s * HeadDim;
      #pragma unroll
      for (int kk = 0; kk < HeadDim; ++kk) {
        score += static_cast<float>(q[q_base + kk]) * static_cast<float>(k[k_base + kk]);
      }
      score *= scale;
      if (mask_kind == 1) {
        if (static_cast<const bool*>(mask)[mask_base + s]) {
          score = -INFINITY;
        }
      } else if (mask_kind == 2) {
        score += static_cast<float>(static_cast<const scalar_t*>(mask)[mask_base + s]);
      } else if (mask_kind == 3) {
        score += static_cast<const float*>(mask)[mask_base + s];
      }
      const float weight = expf(score - row_max) / denom;
      acc += weight * static_cast<float>(v[kv_base + s * HeadDim + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, int Threads, int HeadDim>
__global__ __launch_bounds__(Threads, 2) void decode_attention_q1_hdim_sm90_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const void* __restrict__ mask,
    int mask_kind,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t sk,
    float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch * q_heads;
  if (row >= total_rows) {
    return;
  }

  __shared__ float reduce_shared[Threads];
  __shared__ float q_shared[HeadDim];
  __shared__ float broadcast_shared;

  constexpr int kOutputsPerThread = (HeadDim + Threads - 1) / Threads;
  float accum[kOutputsPerThread];
  #pragma unroll
  for (int idx = 0; idx < kOutputsPerThread; ++idx) {
    accum[idx] = 0.0f;
  }

  const int64_t head_idx = row % q_heads;
  const int64_t batch_idx = row / q_heads;
  const int64_t head_group = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_group;

  const int64_t q_base = ((batch_idx * q_heads) + head_idx) * HeadDim;
  const int64_t kv_base = ((batch_idx * kv_heads) + kv_head_idx) * sk * HeadDim;
  const int64_t mask_base = ((batch_idx * q_heads) + head_idx) * sk;

  for (int d = threadIdx.x; d < HeadDim; d += Threads) {
    q_shared[d] = static_cast<float>(q[q_base + d]);
  }
  __syncthreads();

  float row_max = -INFINITY;
  for (int64_t s = 0; s < sk; ++s) {
    const int64_t k_base = kv_base + s * HeadDim;
    float partial = 0.0f;
    for (int d = threadIdx.x; d < HeadDim; d += Threads) {
      partial += q_shared[d] * static_cast<float>(k[k_base + d]);
    }
    const float dot = BlockReduceSum<Threads>(partial, reduce_shared);
    if (threadIdx.x == 0) {
      float score = dot * scale;
      if (mask_kind == 1) {
        if (static_cast<const bool*>(mask)[mask_base + s]) {
          score = -INFINITY;
        }
      } else if (mask_kind == 2) {
        score += static_cast<float>(static_cast<const scalar_t*>(mask)[mask_base + s]);
      } else if (mask_kind == 3) {
        score += static_cast<const float*>(mask)[mask_base + s];
      }
      row_max = fmaxf(row_max, score);
      broadcast_shared = row_max;
    }
    __syncthreads();
    row_max = broadcast_shared;
  }

  float denom = 0.0f;
  for (int64_t s = 0; s < sk; ++s) {
    const int64_t k_base = kv_base + s * HeadDim;
    float partial = 0.0f;
    for (int d = threadIdx.x; d < HeadDim; d += Threads) {
      partial += q_shared[d] * static_cast<float>(k[k_base + d]);
    }
    const float dot = BlockReduceSum<Threads>(partial, reduce_shared);
    if (threadIdx.x == 0) {
      float score = dot * scale;
      if (mask_kind == 1) {
        if (static_cast<const bool*>(mask)[mask_base + s]) {
          score = -INFINITY;
        }
      } else if (mask_kind == 2) {
        score += static_cast<float>(static_cast<const scalar_t*>(mask)[mask_base + s]);
      } else if (mask_kind == 3) {
        score += static_cast<const float*>(mask)[mask_base + s];
      }
      denom += expf(score - row_max);
      broadcast_shared = denom;
    }
    __syncthreads();
    denom = broadcast_shared;
  }
  denom = fmaxf(denom, 1e-20f);

  for (int64_t s = 0; s < sk; ++s) {
    const int64_t k_base = kv_base + s * HeadDim;
    float partial = 0.0f;
    for (int d = threadIdx.x; d < HeadDim; d += Threads) {
      partial += q_shared[d] * static_cast<float>(k[k_base + d]);
    }
    const float dot = BlockReduceSum<Threads>(partial, reduce_shared);
    if (threadIdx.x == 0) {
      float score = dot * scale;
      if (mask_kind == 1) {
        if (static_cast<const bool*>(mask)[mask_base + s]) {
          score = -INFINITY;
        }
      } else if (mask_kind == 2) {
        score += static_cast<float>(static_cast<const scalar_t*>(mask)[mask_base + s]);
      } else if (mask_kind == 3) {
        score += static_cast<const float*>(mask)[mask_base + s];
      }
      broadcast_shared = expf(score - row_max) / denom;
    }
    __syncthreads();
    const float weight = broadcast_shared;
    const int64_t v_base = kv_base + s * HeadDim;
    #pragma unroll
    for (int idx = 0; idx < kOutputsPerThread; ++idx) {
      const int d = threadIdx.x + idx * Threads;
      if (d < HeadDim) {
        accum[idx] += weight * static_cast<float>(v[v_base + d]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int idx = 0; idx < kOutputsPerThread; ++idx) {
    const int d = threadIdx.x + idx * Threads;
    if (d < HeadDim) {
      out[q_base + d] = static_cast<scalar_t>(accum[idx]);
    }
  }
}

template <typename scalar_t>
inline void LaunchGenericAttentionDecode(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const void* mask_ptr,
    int mask_kind,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream,
    int row_reduce_threads) {
  LaunchGenericAttentionKernel<scalar_t>(
      q_contig,
      k_contig,
      v_contig,
      mask_ptr,
      mask_kind,
      out,
      desc.batch,
      desc.q_heads,
      desc.kv_heads,
      desc.q_len,
      desc.kv_len,
      desc.head_dim,
      scale_value,
      false,
      stream,
      row_reduce_threads);
}

template <typename scalar_t>
inline void LaunchDecodeAttentionQ1(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const void* mask_ptr,
    int mask_kind,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(desc.batch * desc.q_heads));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      decode_attention_q1_forward_kernel<scalar_t, kSmallRowThreads, true>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
    case kMediumRowThreads:
      decode_attention_q1_forward_kernel<scalar_t, kMediumRowThreads, true>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
    default:
      decode_attention_q1_forward_kernel<scalar_t, kLargeRowThreads, true>
          <<<blocks, kLargeRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
  }
}

template <typename scalar_t>
inline void LaunchDecodeAttentionQ1NoMask(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(desc.batch * desc.q_heads));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      decode_attention_q1_forward_kernel<scalar_t, kSmallRowThreads, false>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
    case kMediumRowThreads:
      decode_attention_q1_forward_kernel<scalar_t, kMediumRowThreads, false>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
    default:
      decode_attention_q1_forward_kernel<scalar_t, kLargeRowThreads, false>
          <<<blocks, kLargeRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
  }
}

template <typename scalar_t>
inline void LaunchDecodeAttentionQ1MHA(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const void* mask_ptr,
    int mask_kind,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(desc.batch * desc.q_heads));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      decode_attention_q1_mha_forward_kernel<scalar_t, kSmallRowThreads, true>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
    case kMediumRowThreads:
      decode_attention_q1_mha_forward_kernel<scalar_t, kMediumRowThreads, true>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
    default:
      decode_attention_q1_mha_forward_kernel<scalar_t, kLargeRowThreads, true>
          <<<blocks, kLargeRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
  }
}

template <typename scalar_t>
inline void LaunchDecodeAttentionQ1MHANoMask(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(desc.batch * desc.q_heads));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      decode_attention_q1_mha_forward_kernel<scalar_t, kSmallRowThreads, false>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
    case kMediumRowThreads:
      decode_attention_q1_mha_forward_kernel<scalar_t, kMediumRowThreads, false>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
    default:
      decode_attention_q1_mha_forward_kernel<scalar_t, kLargeRowThreads, false>
          <<<blocks, kLargeRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.kv_len,
              desc.head_dim,
              scale_value);
      return;
  }
}

template <typename scalar_t, int HeadDim>
inline void LaunchDecodeAttentionQ1Specialized(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const void* mask_ptr,
    int mask_kind,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(desc.batch * desc.q_heads));
  const bool use_sm90 = t10::cuda::DeviceIsSm90OrLater(q_contig);
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      if (use_sm90) {
        decode_attention_q1_hdim_sm90_forward_kernel<scalar_t, kSmallRowThreads, HeadDim>
            <<<blocks, kSmallRowThreads, 0, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_contig.data_ptr<scalar_t>(),
                v_contig.data_ptr<scalar_t>(),
                mask_ptr,
                mask_kind,
                out.data_ptr<scalar_t>(),
                desc.batch,
                desc.q_heads,
                desc.kv_heads,
                desc.kv_len,
                scale_value);
      } else {
        decode_attention_q1_hdim_forward_kernel<scalar_t, kSmallRowThreads, HeadDim>
            <<<blocks, kSmallRowThreads, 0, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_contig.data_ptr<scalar_t>(),
                v_contig.data_ptr<scalar_t>(),
                mask_ptr,
                mask_kind,
                out.data_ptr<scalar_t>(),
                desc.batch,
                desc.q_heads,
                desc.kv_heads,
                desc.kv_len,
                scale_value);
      }
      return;
    case kMediumRowThreads:
      if (use_sm90) {
        decode_attention_q1_hdim_sm90_forward_kernel<scalar_t, kMediumRowThreads, HeadDim>
            <<<blocks, kMediumRowThreads, 0, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_contig.data_ptr<scalar_t>(),
                v_contig.data_ptr<scalar_t>(),
                mask_ptr,
                mask_kind,
                out.data_ptr<scalar_t>(),
                desc.batch,
                desc.q_heads,
                desc.kv_heads,
                desc.kv_len,
                scale_value);
      } else {
        decode_attention_q1_hdim_forward_kernel<scalar_t, kMediumRowThreads, HeadDim>
            <<<blocks, kMediumRowThreads, 0, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_contig.data_ptr<scalar_t>(),
                v_contig.data_ptr<scalar_t>(),
                mask_ptr,
                mask_kind,
                out.data_ptr<scalar_t>(),
                desc.batch,
                desc.q_heads,
                desc.kv_heads,
                desc.kv_len,
                scale_value);
      }
      return;
    default:
      if (use_sm90) {
        decode_attention_q1_hdim_sm90_forward_kernel<scalar_t, kLargeRowThreads, HeadDim>
            <<<blocks, kLargeRowThreads, 0, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_contig.data_ptr<scalar_t>(),
                v_contig.data_ptr<scalar_t>(),
                mask_ptr,
                mask_kind,
                out.data_ptr<scalar_t>(),
                desc.batch,
                desc.q_heads,
                desc.kv_heads,
                desc.kv_len,
                scale_value);
      } else {
        decode_attention_q1_hdim_forward_kernel<scalar_t, kLargeRowThreads, HeadDim>
            <<<blocks, kLargeRowThreads, 0, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_contig.data_ptr<scalar_t>(),
                v_contig.data_ptr<scalar_t>(),
                mask_ptr,
                mask_kind,
                out.data_ptr<scalar_t>(),
                desc.batch,
                desc.q_heads,
                desc.kv_heads,
                desc.kv_len,
                scale_value);
      }
      return;
  }
}

template <typename scalar_t>
inline void LaunchPlannedAttentionDecode(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const void* mask_ptr,
    int mask_kind,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    const t10::policy::AttentionPlan& plan,
    float scale_value,
    cudaStream_t stream) {
  switch (plan.kernel) {
    case t10::policy::AttentionKernelKind::kDecodeQ1MHANoMask:
      LaunchDecodeAttentionQ1MHANoMask<scalar_t>(
          q_contig,
          k_contig,
          v_contig,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    case t10::policy::AttentionKernelKind::kDecodeQ1NoMask:
      LaunchDecodeAttentionQ1NoMask<scalar_t>(
          q_contig,
          k_contig,
          v_contig,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    case t10::policy::AttentionKernelKind::kDecodeQ1MHA:
      LaunchDecodeAttentionQ1MHA<scalar_t>(
          q_contig,
          k_contig,
          v_contig,
          mask_ptr,
          mask_kind,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    case t10::policy::AttentionKernelKind::kDecodeQ1Hdim32:
      LaunchDecodeAttentionQ1Specialized<scalar_t, 32>(
          q_contig,
          k_contig,
          v_contig,
          mask_ptr,
          mask_kind,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    case t10::policy::AttentionKernelKind::kDecodeQ1Hdim64:
      LaunchDecodeAttentionQ1Specialized<scalar_t, 64>(
          q_contig,
          k_contig,
          v_contig,
          mask_ptr,
          mask_kind,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    case t10::policy::AttentionKernelKind::kDecodeQ1Hdim96:
      LaunchDecodeAttentionQ1Specialized<scalar_t, 96>(
          q_contig,
          k_contig,
          v_contig,
          mask_ptr,
          mask_kind,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    case t10::policy::AttentionKernelKind::kDecodeQ1Hdim128:
      LaunchDecodeAttentionQ1Specialized<scalar_t, 128>(
          q_contig,
          k_contig,
          v_contig,
          mask_ptr,
          mask_kind,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    case t10::policy::AttentionKernelKind::kDecodeQ1:
      LaunchDecodeAttentionQ1<scalar_t>(
          q_contig,
          k_contig,
          v_contig,
          mask_ptr,
          mask_kind,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    default:
      LaunchGenericAttentionDecode<scalar_t>(
          q_contig,
          k_contig,
          v_contig,
          mask_ptr,
          mask_kind,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
  }
}

}  // namespace t10::cuda::attention
