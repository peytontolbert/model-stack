#pragma once

#include "cuda_attention_common.cuh"
#include "../cuda_device_arch.cuh"

#include "../../descriptors/attention_desc.h"
#include "../../policy/attention_policy.h"
#include "cuda_attention_cutlass_prefill.cuh"
#include "cuda_attention_pytorch_memeff_prefill.cuh"
#include "cuda_attention_sm80_inference_prefill.cuh"

#include <cstdlib>
#include <mma.h>
#include <type_traits>

namespace t10::cuda::attention {

constexpr int kPrefillTensorCoreTileM = 16;
constexpr int kPrefillTensorCoreTileN = 16;
constexpr int kPrefillTensorCoreTileK = 16;
constexpr int kPrefillTensorCoreWarps = 4;
constexpr int kPrefillTensorCoreThreads = kPrefillTensorCoreWarps * 32;
constexpr int kPrefillTensorCoreRowsPerBlock = kPrefillTensorCoreWarps * kPrefillTensorCoreTileM;

__device__ inline float WarpReduceSum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ inline float WarpReduceMax(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float FastExp(float value) {
  return __expf(value);
}

inline bool AttentionPrefillTensorCoreDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_TENSORCORE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

template <typename scalar_t>
struct PrefillWmmaInputType;

template <>
struct PrefillWmmaInputType<at::Half> {
  using type = __half;
};

template <>
struct PrefillWmmaInputType<at::BFloat16> {
  using type = __nv_bfloat16;
};

template <typename scalar_t>
using PrefillWmmaInputTypeT = typename PrefillWmmaInputType<scalar_t>::type;

template <typename scalar_t, int HeadDim>
__device__ inline void compute_prefill_score_tile_tensorcore(
    const scalar_t* q_tile,
    const scalar_t* k_tile_col_major,
    float* score_tile) {
  using wmma_scalar_t = PrefillWmmaInputTypeT<scalar_t>;
  static_assert((HeadDim % kPrefillTensorCoreTileK) == 0, "Tensor Core prefill head_dim must be a multiple of 16");

  nvcuda::wmma::fragment<
      nvcuda::wmma::accumulator,
      kPrefillTensorCoreTileM,
      kPrefillTensorCoreTileN,
      kPrefillTensorCoreTileK,
      float>
      acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

  #pragma unroll
  for (int chunk = 0; chunk < HeadDim; chunk += kPrefillTensorCoreTileK) {
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        kPrefillTensorCoreTileM,
        kPrefillTensorCoreTileN,
        kPrefillTensorCoreTileK,
        wmma_scalar_t,
        nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        kPrefillTensorCoreTileM,
        kPrefillTensorCoreTileN,
        kPrefillTensorCoreTileK,
        wmma_scalar_t,
        nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::load_matrix_sync(
        a_frag,
        reinterpret_cast<const wmma_scalar_t*>(q_tile + chunk),
        HeadDim);
    nvcuda::wmma::load_matrix_sync(
        b_frag,
        reinterpret_cast<const wmma_scalar_t*>(k_tile_col_major + chunk),
        HeadDim);
    nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  nvcuda::wmma::store_matrix_sync(
      score_tile,
      acc_frag,
      kPrefillTensorCoreTileN,
      nvcuda::wmma::mem_row_major);
}

template <typename scalar_t, int Threads, bool HasMask>
__global__ void prefill_attention_mha_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const void* __restrict__ mask,
    int mask_kind,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t q_heads,
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

  const int64_t q_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * dh;
  const int64_t kv_base = ((batch_idx * q_heads) + head_idx) * sk * dh;
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
    if (is_causal && s > query_idx) {
      score = -INFINITY;
    }
    local_sum += FastExp(score - row_max);
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
      if (is_causal && s > query_idx) {
        score = -INFINITY;
      }
      const float weight = FastExp(score - row_max) / denom;
      acc += weight * static_cast<float>(v[kv_base + s * dh + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, int Threads, int HeadDim, int TileN>
__global__ void prefill_attention_hdim_tiled_forward_kernel(
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
    float scale,
    bool is_causal) {
  constexpr int kWarpSizeLocal = 32;
  constexpr int kRowsPerBlock = Threads / kWarpSizeLocal;
  constexpr int kDimsPerLane = (HeadDim + kWarpSizeLocal - 1) / kWarpSizeLocal;

  const int64_t q_tiles = (tq + kRowsPerBlock - 1) / kRowsPerBlock;
  const int64_t tile_idx = static_cast<int64_t>(blockIdx.x) % q_tiles;
  const int64_t tmp = static_cast<int64_t>(blockIdx.x) / q_tiles;
  const int64_t head_idx = tmp % q_heads;
  const int64_t batch_idx = tmp / q_heads;

  const int warp_id = threadIdx.x / kWarpSizeLocal;
  const int lane = threadIdx.x % kWarpSizeLocal;

  const int64_t query_idx = tile_idx * kRowsPerBlock + warp_id;
  const bool row_active = query_idx < tq;

  const int64_t head_group = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_group;
  const int64_t row_q_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * HeadDim;
  const int64_t kv_base = ((batch_idx * kv_heads) + kv_head_idx) * sk * HeadDim;
  const int64_t row_mask_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * sk;

  __shared__ float q_tile[kRowsPerBlock * HeadDim];
  __shared__ scalar_t k_tile[TileN * HeadDim];
  __shared__ scalar_t v_tile[TileN * HeadDim];
  __shared__ float score_tile[kRowsPerBlock * TileN];
  __shared__ float weight_tile[kRowsPerBlock * TileN];
  __shared__ float row_max_shared[kRowsPerBlock];
  __shared__ float row_sum_shared[kRowsPerBlock];
  __shared__ float row_rescale_shared[kRowsPerBlock];

  for (int d = lane; d < HeadDim; d += kWarpSizeLocal) {
    q_tile[warp_id * HeadDim + d] = row_active ? static_cast<float>(q[row_q_base + d]) : 0.0f;
  }
  if (lane == 0) {
    row_max_shared[warp_id] = -INFINITY;
    row_sum_shared[warp_id] = 0.0f;
    row_rescale_shared[warp_id] = 0.0f;
  }

  float accum[kDimsPerLane];
  #pragma unroll
  for (int idx = 0; idx < kDimsPerLane; ++idx) {
    accum[idx] = 0.0f;
  }

  __syncthreads();

  for (int64_t s_tile_start = 0; s_tile_start < sk; s_tile_start += TileN) {
    const int valid_k = static_cast<int>((sk - s_tile_start) < TileN ? (sk - s_tile_start) : TileN);
    for (int idx = threadIdx.x; idx < valid_k * HeadDim; idx += Threads) {
      const int col = idx / HeadDim;
      const int d = idx % HeadDim;
      const int64_t kv_idx = s_tile_start + col;
      const int64_t base = kv_base + kv_idx * HeadDim + d;
      k_tile[idx] = k[base];
      v_tile[idx] = v[base];
    }
    __syncthreads();

    if (row_active) {
      for (int col = 0; col < valid_k; ++col) {
        float partial = 0.0f;
        #pragma unroll
        for (int idx = 0; idx < kDimsPerLane; ++idx) {
          const int d = lane + idx * kWarpSizeLocal;
          if (d < HeadDim) {
            partial += q_tile[warp_id * HeadDim + d] * static_cast<float>(k_tile[col * HeadDim + d]);
          }
        }
        const float dot = WarpReduceSum(partial);
        if (lane == 0) {
          float score = dot * scale;
          if (mask_kind == 1) {
            if (static_cast<const bool*>(mask)[row_mask_base + s_tile_start + col]) {
              score = -INFINITY;
            }
          } else if (mask_kind == 2) {
            score += static_cast<float>(static_cast<const scalar_t*>(mask)[row_mask_base + s_tile_start + col]);
          } else if (mask_kind == 3) {
            score += static_cast<const float*>(mask)[row_mask_base + s_tile_start + col];
          }
          if (is_causal && (s_tile_start + col) > query_idx) {
            score = -INFINITY;
          }
          score_tile[warp_id * TileN + col] = score;
        }
      }
    }

    __syncwarp();

    if (row_active) {
      const float prev_row_max = row_max_shared[warp_id];
      const float prev_row_sum = row_sum_shared[warp_id];

      float local_tile_max = -INFINITY;
      for (int col = lane; col < valid_k; col += kWarpSizeLocal) {
        local_tile_max = fmaxf(local_tile_max, score_tile[warp_id * TileN + col]);
      }
      const float tile_max = WarpReduceMax(local_tile_max);

      float new_row_max = -INFINITY;
      float prev_scale = 0.0f;
      if (lane == 0) {
        new_row_max = fmaxf(prev_row_max, tile_max);
        prev_scale =
            (isfinite(prev_row_max) && isfinite(new_row_max)) ? FastExp(prev_row_max - new_row_max) : 0.0f;
        row_max_shared[warp_id] = new_row_max;
        row_rescale_shared[warp_id] = prev_scale;
      }
      new_row_max = __shfl_sync(0xffffffffu, new_row_max, 0);
      prev_scale = __shfl_sync(0xffffffffu, prev_scale, 0);

      float local_tile_sum = 0.0f;
      if (isfinite(new_row_max)) {
        for (int col = lane; col < valid_k; col += kWarpSizeLocal) {
          const float score = score_tile[warp_id * TileN + col];
          const float weight = isfinite(score) ? FastExp(score - new_row_max) : 0.0f;
          weight_tile[warp_id * TileN + col] = weight;
          local_tile_sum += weight;
        }
      } else {
        for (int col = lane; col < valid_k; col += kWarpSizeLocal) {
          weight_tile[warp_id * TileN + col] = 0.0f;
        }
      }

      const float tile_sum = WarpReduceSum(local_tile_sum);
      if (lane == 0) {
        row_sum_shared[warp_id] = prev_row_sum * prev_scale + tile_sum;
      }
    }

    __syncwarp();

    if (row_active) {
      const float prev_scale = row_rescale_shared[warp_id];
      #pragma unroll
      for (int idx = 0; idx < kDimsPerLane; ++idx) {
        accum[idx] *= prev_scale;
      }
      for (int col = 0; col < valid_k; ++col) {
        const float weight = weight_tile[warp_id * TileN + col];
        #pragma unroll
        for (int idx = 0; idx < kDimsPerLane; ++idx) {
          const int d = lane + idx * kWarpSizeLocal;
          if (d < HeadDim) {
            accum[idx] += weight * static_cast<float>(v_tile[col * HeadDim + d]);
          }
        }
      }
    }

    __syncthreads();
  }

  if (row_active) {
    const float inv_sum = 1.0f / fmaxf(row_sum_shared[warp_id], 1e-20f);
    #pragma unroll
    for (int idx = 0; idx < kDimsPerLane; ++idx) {
      const int d = lane + idx * kWarpSizeLocal;
      if (d < HeadDim) {
        out[row_q_base + d] = static_cast<scalar_t>(accum[idx] * inv_sum);
      }
    }
  }
}

template <typename scalar_t, int HeadDim, bool IsCausal>
__global__ __launch_bounds__(kPrefillTensorCoreThreads, 2) void prefill_attention_hdim_tensorcore_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t tq,
    int64_t sk,
    float scale) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int lane_row = lane & 15;
  const int64_t tiles_per_head = (tq + kPrefillTensorCoreRowsPerBlock - 1) / kPrefillTensorCoreRowsPerBlock;
  const int64_t total_blocks = batch * q_heads * tiles_per_head;

  __shared__ scalar_t q_tile[kPrefillTensorCoreRowsPerBlock * HeadDim];
  __shared__ scalar_t k_tile_col_major[kPrefillTensorCoreTileN * HeadDim];
  __shared__ scalar_t v_tile[kPrefillTensorCoreTileN * HeadDim];
  __shared__ float row_max[kPrefillTensorCoreRowsPerBlock];
  __shared__ float row_sum[kPrefillTensorCoreRowsPerBlock];
  __shared__ float row_rescale[kPrefillTensorCoreRowsPerBlock];
  __shared__ float score_tile[kPrefillTensorCoreRowsPerBlock * kPrefillTensorCoreTileN];
  __shared__ float weight_tile[kPrefillTensorCoreRowsPerBlock * kPrefillTensorCoreTileN];
  __shared__ float out_accum[kPrefillTensorCoreRowsPerBlock * HeadDim];

  for (int64_t logical_block = static_cast<int64_t>(blockIdx.x);
       logical_block < total_blocks;
       logical_block += static_cast<int64_t>(gridDim.x)) {
    const int64_t tile_idx = logical_block % tiles_per_head;
    const int64_t tmp = logical_block / tiles_per_head;
    const int64_t head_idx = tmp % q_heads;
    const int64_t batch_idx = tmp / q_heads;
    const int64_t q_tile_start = tile_idx * kPrefillTensorCoreRowsPerBlock;
    const int valid_q = static_cast<int>((tq - q_tile_start) < kPrefillTensorCoreRowsPerBlock
                                             ? (tq - q_tile_start)
                                             : kPrefillTensorCoreRowsPerBlock);
    if (valid_q <= 0) {
      continue;
    }

    const int64_t head_group = q_heads / kv_heads;
    const int64_t kv_head_idx = head_idx / head_group;

    for (int idx = tid; idx < kPrefillTensorCoreRowsPerBlock * HeadDim; idx += kPrefillTensorCoreThreads) {
      const int row = idx / HeadDim;
      const int d = idx % HeadDim;
      if (row < valid_q) {
        const int64_t q_idx = q_tile_start + row;
        const int64_t q_base = (((batch_idx * q_heads) + head_idx) * tq + q_idx) * HeadDim;
        q_tile[idx] = q[q_base + d];
        out_accum[idx] = 0.0f;
      } else {
        q_tile[idx] = static_cast<scalar_t>(0);
        out_accum[idx] = 0.0f;
      }
    }
    if (tid < kPrefillTensorCoreRowsPerBlock) {
      row_max[tid] = -INFINITY;
      row_sum[tid] = 0.0f;
      row_rescale[tid] = 0.0f;
    }
    __syncthreads();

    for (int64_t s_tile_start = 0; s_tile_start < sk; s_tile_start += kPrefillTensorCoreTileN) {
      const int valid_k = static_cast<int>(
          (sk - s_tile_start) < kPrefillTensorCoreTileN ? (sk - s_tile_start) : kPrefillTensorCoreTileN);
      for (int idx = tid; idx < kPrefillTensorCoreTileN * HeadDim; idx += kPrefillTensorCoreThreads) {
        const int col = idx / HeadDim;
        const int d = idx % HeadDim;
        if (col < valid_k) {
          const int64_t kv_idx = s_tile_start + col;
          const int64_t kv_base = (((batch_idx * kv_heads) + kv_head_idx) * sk + kv_idx) * HeadDim;
          k_tile_col_major[col * HeadDim + d] = k[kv_base + d];
          v_tile[col * HeadDim + d] = v[kv_base + d];
        } else {
          k_tile_col_major[col * HeadDim + d] = static_cast<scalar_t>(0);
          v_tile[col * HeadDim + d] = static_cast<scalar_t>(0);
        }
      }
      __syncthreads();

      if (warp < ((valid_q + kPrefillTensorCoreTileM - 1) / kPrefillTensorCoreTileM)) {
        compute_prefill_score_tile_tensorcore<scalar_t, HeadDim>(
            q_tile + warp * kPrefillTensorCoreTileM * HeadDim,
            k_tile_col_major,
            score_tile + warp * kPrefillTensorCoreTileM * kPrefillTensorCoreTileN);
      }
      __syncthreads();

      if (lane < kPrefillTensorCoreTileM) {
        const int row = warp * kPrefillTensorCoreTileM + lane_row;
        if (row < valid_q) {
          const float prev_row_max = row_max[row];
          const float prev_row_sum = row_sum[row];
          const int64_t q_idx = q_tile_start + row;

          float tile_max = -INFINITY;
          #pragma unroll
          for (int col = 0; col < kPrefillTensorCoreTileN; ++col) {
            if (col >= valid_k) {
              break;
            }
            const int64_t kv_idx = s_tile_start + col;
            if constexpr (IsCausal) {
              if (kv_idx > q_idx) {
                continue;
              }
            }
            const float score = score_tile[row * kPrefillTensorCoreTileN + col] * scale;
            if (!::isfinite(score)) {
              continue;
            }
            tile_max = fmaxf(tile_max, score);
          }

          const float new_row_max = fmaxf(prev_row_max, tile_max);
          float prev_scale = 0.0f;
          if (::isfinite(prev_row_max) && ::isfinite(new_row_max)) {
            prev_scale = FastExp(prev_row_max - new_row_max);
          }

          float tile_sum = 0.0f;
          #pragma unroll
          for (int col = 0; col < kPrefillTensorCoreTileN; ++col) {
            float weight = 0.0f;
            if (col < valid_k) {
              const int64_t kv_idx = s_tile_start + col;
              bool allowed = true;
              if constexpr (IsCausal) {
                if (kv_idx > q_idx) {
                  allowed = false;
                }
              }
              if (allowed && ::isfinite(new_row_max)) {
                const float score = score_tile[row * kPrefillTensorCoreTileN + col] * scale;
                if (::isfinite(score)) {
                  weight = FastExp(score - new_row_max);
                }
              }
            }
            weight_tile[row * kPrefillTensorCoreTileN + col] = weight;
            tile_sum += weight;
          }

          row_max[row] = new_row_max;
          row_sum[row] = prev_row_sum * prev_scale + tile_sum;
          row_rescale[row] = prev_scale;
        }
      }
      __syncthreads();

      for (int idx = tid; idx < valid_q * HeadDim; idx += kPrefillTensorCoreThreads) {
        const int row = idx / HeadDim;
        const int d = idx % HeadDim;
        float accum = out_accum[idx] * row_rescale[row];
        #pragma unroll
        for (int col = 0; col < kPrefillTensorCoreTileN; ++col) {
          if (col >= valid_k) {
            break;
          }
          accum += weight_tile[row * kPrefillTensorCoreTileN + col] *
                   static_cast<float>(v_tile[col * HeadDim + d]);
        }
        out_accum[idx] = accum;
      }
      __syncthreads();
    }

    for (int idx = tid; idx < valid_q * HeadDim; idx += kPrefillTensorCoreThreads) {
      const int row = idx / HeadDim;
      const int d = idx % HeadDim;
      const float denom = fmaxf(row_sum[row], 1e-20f);
      const float value = out_accum[idx] / denom;
      const int64_t q_idx = q_tile_start + row;
      const int64_t out_base = (((batch_idx * q_heads) + head_idx) * tq + q_idx) * HeadDim;
      out[out_base + d] = static_cast<scalar_t>(value);
    }
    __syncthreads();
  }
}

template <typename scalar_t, int Threads, int HeadDim>
__global__ void prefill_attention_hdim_forward_kernel(
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
    float scale,
    bool is_causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch * q_heads * tq;
  if (row >= total_rows) {
    return;
  }

  extern __shared__ float shared_mem[];
  float* scores_shared = shared_mem;
  float* q_shared = scores_shared + sk;
  float* reduce_shared = q_shared + HeadDim;

  const int64_t query_idx = row % tq;
  const int64_t tmp = row / tq;
  const int64_t head_idx = tmp % q_heads;
  const int64_t batch_idx = tmp / q_heads;
  const int64_t head_group = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_group;

  const int64_t q_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * HeadDim;
  const int64_t kv_base = ((batch_idx * kv_heads) + kv_head_idx) * sk * HeadDim;
  const int64_t mask_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * sk;

  for (int d = threadIdx.x; d < HeadDim; d += Threads) {
    q_shared[d] = static_cast<float>(q[q_base + d]);
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int64_t s = threadIdx.x; s < sk; s += Threads) {
    const int64_t k_base = kv_base + s * HeadDim;
    float score = 0.0f;
    #pragma unroll
    for (int d = 0; d < HeadDim; ++d) {
      score += q_shared[d] * static_cast<float>(k[k_base + d]);
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
    if (is_causal && s > query_idx) {
      score = -INFINITY;
    }
    scores_shared[s] = score;
    local_max = fmaxf(local_max, score);
  }
  const float row_max = BlockReduceMax<Threads>(local_max, reduce_shared);

  float local_sum = 0.0f;
  if (isfinite(row_max)) {
    for (int64_t s = threadIdx.x; s < sk; s += Threads) {
      const float weight = FastExp(scores_shared[s] - row_max);
      scores_shared[s] = weight;
      local_sum += weight;
    }
  } else {
    for (int64_t s = threadIdx.x; s < sk; s += Threads) {
      scores_shared[s] = 0.0f;
    }
  }
  float denom = BlockReduceSum<Threads>(local_sum, reduce_shared);
  denom = fmaxf(denom, 1e-20f);
  const float inv_denom = 1.0f / denom;
  for (int64_t s = threadIdx.x; s < sk; s += Threads) {
    scores_shared[s] *= inv_denom;
  }
  __syncthreads();

  for (int d = threadIdx.x; d < HeadDim; d += Threads) {
    float acc = 0.0f;
    for (int64_t s = 0; s < sk; ++s) {
      acc += scores_shared[s] * static_cast<float>(v[kv_base + s * HeadDim + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, int Threads, int HeadDim>
__global__ void prefill_attention_hdim_legacy_forward_kernel(
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

  const int64_t q_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * HeadDim;
  const int64_t kv_base = ((batch_idx * kv_heads) + kv_head_idx) * sk * HeadDim;
  const int64_t mask_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * sk;

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
    if (is_causal && s > query_idx) {
      score = -INFINITY;
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
    if (is_causal && s > query_idx) {
      score = -INFINITY;
    }
    local_sum += FastExp(score - row_max);
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
      if (is_causal && s > query_idx) {
        score = -INFINITY;
      }
      const float weight = FastExp(score - row_max) / denom;
      acc += weight * static_cast<float>(v[kv_base + s * HeadDim + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, int Threads, int HeadDim>
__global__ __launch_bounds__(Threads, 2) void prefill_attention_hdim_sm90_forward_kernel(
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
    float scale,
    bool is_causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch * q_heads * tq;
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

  const int64_t query_idx = row % tq;
  const int64_t tmp = row / tq;
  const int64_t head_idx = tmp % q_heads;
  const int64_t batch_idx = tmp / q_heads;
  const int64_t head_group = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_group;

  const int64_t q_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * HeadDim;
  const int64_t kv_base = ((batch_idx * kv_heads) + kv_head_idx) * sk * HeadDim;
  const int64_t mask_base = (((batch_idx * q_heads) + head_idx) * tq + query_idx) * sk;

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
      if (is_causal && s > query_idx) {
        score = -INFINITY;
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
      if (is_causal && s > query_idx) {
        score = -INFINITY;
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
      if (is_causal && s > query_idx) {
        score = -INFINITY;
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
inline void LaunchGenericAttentionPrefill(
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
      desc.causal,
      stream,
      row_reduce_threads);
}

template <typename scalar_t>
inline void LaunchGenericAttentionPrefillNoMask(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream,
    int row_reduce_threads) {
  LaunchGenericAttentionKernelNoMask<scalar_t>(
      q_contig,
      k_contig,
      v_contig,
      out,
      desc.batch,
      desc.q_heads,
      desc.kv_heads,
      desc.q_len,
      desc.kv_len,
      desc.head_dim,
      scale_value,
      desc.causal,
      stream,
      row_reduce_threads);
}

template <typename scalar_t, int HeadDim>
inline void LaunchPrefillAttentionSpecialized(
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
  if (TryLaunchModelStackSm80InferenceAttentionPrefill<scalar_t, HeadDim>(
          q_contig,
          k_contig,
          v_contig,
          out,
          desc,
          scale_value,
          stream)) {
    return;
  }
  if (TryLaunchPyTorchMemEffAttentionPrefill<scalar_t, HeadDim>(
          q_contig,
          k_contig,
          v_contig,
          out,
          desc,
          scale_value,
          stream)) {
    return;
  }
  if (TryLaunchCutlassAttentionPrefill<scalar_t, HeadDim>(
          q_contig,
          k_contig,
          v_contig,
          out,
          desc,
          scale_value,
          stream)) {
    return;
  }
  const dim3 blocks(static_cast<unsigned int>(desc.batch * desc.q_heads * desc.q_len));
  const bool use_sm90 = t10::cuda::DeviceIsSm90OrLater(q_contig);
  constexpr size_t kMaxSharedScoreBytes = 48 * 1024;
  constexpr int kTiledPrefillTileN = 64;
  constexpr bool kTiledDtype =
      std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>;
  constexpr bool kTensorCoreDtype =
      std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>;
  const bool prefer_tiled_prefill =
      kTiledDtype &&
      !use_sm90 &&
      (q_contig.scalar_type() == torch::kFloat16 || q_contig.scalar_type() == torch::kBFloat16);
  const bool prefer_tensorcore_prefill =
      !AttentionPrefillTensorCoreDisabled() &&
      kTensorCoreDtype &&
      HeadDim == 64 &&
      !use_sm90 &&
      DeviceIsSm80OrLater(q_contig) &&
      desc.mask_kind == t10::desc::AttentionMaskKind::kNone &&
      desc.kv_len >= 128 &&
      desc.q_len > 1;
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      if (use_sm90) {
        prefill_attention_hdim_sm90_forward_kernel<scalar_t, kSmallRowThreads, HeadDim>
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
                desc.q_len,
                desc.kv_len,
                scale_value,
                desc.causal);
      } else {
        constexpr int kRowsPerBlock = kSmallRowThreads / 32;
        if constexpr (kTensorCoreDtype && HeadDim == 64) {
          if (prefer_tensorcore_prefill) {
            const int64_t q_tiles = (desc.q_len + kPrefillTensorCoreRowsPerBlock - 1) / kPrefillTensorCoreRowsPerBlock;
            const dim3 tensorcore_blocks(static_cast<unsigned int>(desc.batch * desc.q_heads * q_tiles));
            if (desc.causal) {
              prefill_attention_hdim_tensorcore_forward_kernel<scalar_t, HeadDim, true>
                  <<<tensorcore_blocks, kPrefillTensorCoreThreads, 0, stream>>>(
                      q_contig.data_ptr<scalar_t>(),
                      k_contig.data_ptr<scalar_t>(),
                      v_contig.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(),
                      desc.batch,
                      desc.q_heads,
                      desc.kv_heads,
                      desc.q_len,
                      desc.kv_len,
                      scale_value);
            } else {
              prefill_attention_hdim_tensorcore_forward_kernel<scalar_t, HeadDim, false>
                  <<<tensorcore_blocks, kPrefillTensorCoreThreads, 0, stream>>>(
                      q_contig.data_ptr<scalar_t>(),
                      k_contig.data_ptr<scalar_t>(),
                      v_contig.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(),
                      desc.batch,
                      desc.q_heads,
                      desc.kv_heads,
                      desc.q_len,
                      desc.kv_len,
                      scale_value);
            }
            return;
          }
        }
        if constexpr (kTiledDtype) {
          if (prefer_tiled_prefill && desc.kv_len >= kTiledPrefillTileN && desc.q_len >= kRowsPerBlock) {
            const int64_t q_tiles = (desc.q_len + kRowsPerBlock - 1) / kRowsPerBlock;
            const dim3 tiled_blocks(static_cast<unsigned int>(desc.batch * desc.q_heads * q_tiles));
            prefill_attention_hdim_tiled_forward_kernel<scalar_t, kSmallRowThreads, HeadDim, kTiledPrefillTileN>
                <<<tiled_blocks, kSmallRowThreads, 0, stream>>>(
                    q_contig.data_ptr<scalar_t>(),
                    k_contig.data_ptr<scalar_t>(),
                    v_contig.data_ptr<scalar_t>(),
                    mask_ptr,
                    mask_kind,
                    out.data_ptr<scalar_t>(),
                    desc.batch,
                    desc.q_heads,
                    desc.kv_heads,
                    desc.q_len,
                    desc.kv_len,
                    scale_value,
                    desc.causal);
            return;
          }
        }
        const size_t shared_score_bytes =
            (static_cast<size_t>(desc.kv_len) + HeadDim + kSmallRowThreads) * sizeof(float);
        if (shared_score_bytes <= kMaxSharedScoreBytes) {
          prefill_attention_hdim_forward_kernel<scalar_t, kSmallRowThreads, HeadDim>
              <<<blocks, kSmallRowThreads, shared_score_bytes, stream>>>(
                  q_contig.data_ptr<scalar_t>(),
                  k_contig.data_ptr<scalar_t>(),
                  v_contig.data_ptr<scalar_t>(),
                  mask_ptr,
                  mask_kind,
                  out.data_ptr<scalar_t>(),
                  desc.batch,
                  desc.q_heads,
                  desc.kv_heads,
                  desc.q_len,
                  desc.kv_len,
                  scale_value,
                  desc.causal);
        } else {
          prefill_attention_hdim_legacy_forward_kernel<scalar_t, kSmallRowThreads, HeadDim>
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
                  desc.q_len,
                  desc.kv_len,
                  scale_value,
                  desc.causal);
          }
      }
      return;
    case kMediumRowThreads:
      if (use_sm90) {
        prefill_attention_hdim_sm90_forward_kernel<scalar_t, kMediumRowThreads, HeadDim>
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
                desc.q_len,
                desc.kv_len,
                scale_value,
                desc.causal);
      } else {
        constexpr int kRowsPerBlock = kMediumRowThreads / 32;
        if constexpr (kTensorCoreDtype && HeadDim == 64) {
          if (prefer_tensorcore_prefill) {
            const int64_t q_tiles = (desc.q_len + kPrefillTensorCoreRowsPerBlock - 1) / kPrefillTensorCoreRowsPerBlock;
            const dim3 tensorcore_blocks(static_cast<unsigned int>(desc.batch * desc.q_heads * q_tiles));
            if (desc.causal) {
              prefill_attention_hdim_tensorcore_forward_kernel<scalar_t, HeadDim, true>
                  <<<tensorcore_blocks, kPrefillTensorCoreThreads, 0, stream>>>(
                      q_contig.data_ptr<scalar_t>(),
                      k_contig.data_ptr<scalar_t>(),
                      v_contig.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(),
                      desc.batch,
                      desc.q_heads,
                      desc.kv_heads,
                      desc.q_len,
                      desc.kv_len,
                      scale_value);
            } else {
              prefill_attention_hdim_tensorcore_forward_kernel<scalar_t, HeadDim, false>
                  <<<tensorcore_blocks, kPrefillTensorCoreThreads, 0, stream>>>(
                      q_contig.data_ptr<scalar_t>(),
                      k_contig.data_ptr<scalar_t>(),
                      v_contig.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(),
                      desc.batch,
                      desc.q_heads,
                      desc.kv_heads,
                      desc.q_len,
                      desc.kv_len,
                      scale_value);
            }
            return;
          }
        }
        if constexpr (kTiledDtype) {
          if (prefer_tiled_prefill && desc.kv_len >= kTiledPrefillTileN && desc.q_len >= kRowsPerBlock) {
            const int64_t q_tiles = (desc.q_len + kRowsPerBlock - 1) / kRowsPerBlock;
            const dim3 tiled_blocks(static_cast<unsigned int>(desc.batch * desc.q_heads * q_tiles));
            prefill_attention_hdim_tiled_forward_kernel<scalar_t, kMediumRowThreads, HeadDim, kTiledPrefillTileN>
                <<<tiled_blocks, kMediumRowThreads, 0, stream>>>(
                    q_contig.data_ptr<scalar_t>(),
                    k_contig.data_ptr<scalar_t>(),
                    v_contig.data_ptr<scalar_t>(),
                    mask_ptr,
                    mask_kind,
                    out.data_ptr<scalar_t>(),
                    desc.batch,
                    desc.q_heads,
                    desc.kv_heads,
                    desc.q_len,
                    desc.kv_len,
                    scale_value,
                    desc.causal);
            return;
          }
        }
        const size_t shared_score_bytes =
            (static_cast<size_t>(desc.kv_len) + HeadDim + kMediumRowThreads) * sizeof(float);
        if (shared_score_bytes <= kMaxSharedScoreBytes) {
          prefill_attention_hdim_forward_kernel<scalar_t, kMediumRowThreads, HeadDim>
              <<<blocks, kMediumRowThreads, shared_score_bytes, stream>>>(
                  q_contig.data_ptr<scalar_t>(),
                  k_contig.data_ptr<scalar_t>(),
                  v_contig.data_ptr<scalar_t>(),
                  mask_ptr,
                  mask_kind,
                  out.data_ptr<scalar_t>(),
                  desc.batch,
                  desc.q_heads,
                  desc.kv_heads,
                  desc.q_len,
                  desc.kv_len,
                  scale_value,
                  desc.causal);
        } else {
          prefill_attention_hdim_legacy_forward_kernel<scalar_t, kMediumRowThreads, HeadDim>
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
                  desc.q_len,
                  desc.kv_len,
                  scale_value,
                  desc.causal);
          }
      }
      return;
    default:
      if (use_sm90) {
        prefill_attention_hdim_sm90_forward_kernel<scalar_t, kLargeRowThreads, HeadDim>
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
                desc.q_len,
                desc.kv_len,
                scale_value,
                desc.causal);
      } else {
        constexpr int kRowsPerBlock = kLargeRowThreads / 32;
        if constexpr (kTensorCoreDtype && HeadDim == 64) {
          if (prefer_tensorcore_prefill) {
            const int64_t q_tiles = (desc.q_len + kPrefillTensorCoreRowsPerBlock - 1) / kPrefillTensorCoreRowsPerBlock;
            const dim3 tensorcore_blocks(static_cast<unsigned int>(desc.batch * desc.q_heads * q_tiles));
            if (desc.causal) {
              prefill_attention_hdim_tensorcore_forward_kernel<scalar_t, HeadDim, true>
                  <<<tensorcore_blocks, kPrefillTensorCoreThreads, 0, stream>>>(
                      q_contig.data_ptr<scalar_t>(),
                      k_contig.data_ptr<scalar_t>(),
                      v_contig.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(),
                      desc.batch,
                      desc.q_heads,
                      desc.kv_heads,
                      desc.q_len,
                      desc.kv_len,
                      scale_value);
            } else {
              prefill_attention_hdim_tensorcore_forward_kernel<scalar_t, HeadDim, false>
                  <<<tensorcore_blocks, kPrefillTensorCoreThreads, 0, stream>>>(
                      q_contig.data_ptr<scalar_t>(),
                      k_contig.data_ptr<scalar_t>(),
                      v_contig.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(),
                      desc.batch,
                      desc.q_heads,
                      desc.kv_heads,
                      desc.q_len,
                      desc.kv_len,
                      scale_value);
            }
            return;
          }
        }
        if constexpr (kTiledDtype) {
          if (prefer_tiled_prefill && desc.kv_len >= kTiledPrefillTileN && desc.q_len >= kRowsPerBlock) {
            const int64_t q_tiles = (desc.q_len + kRowsPerBlock - 1) / kRowsPerBlock;
            const dim3 tiled_blocks(static_cast<unsigned int>(desc.batch * desc.q_heads * q_tiles));
            prefill_attention_hdim_tiled_forward_kernel<scalar_t, kLargeRowThreads, HeadDim, kTiledPrefillTileN>
                <<<tiled_blocks, kLargeRowThreads, 0, stream>>>(
                    q_contig.data_ptr<scalar_t>(),
                    k_contig.data_ptr<scalar_t>(),
                    v_contig.data_ptr<scalar_t>(),
                    mask_ptr,
                    mask_kind,
                    out.data_ptr<scalar_t>(),
                    desc.batch,
                    desc.q_heads,
                    desc.kv_heads,
                    desc.q_len,
                    desc.kv_len,
                    scale_value,
                    desc.causal);
            return;
          }
        }
        const size_t shared_score_bytes =
            (static_cast<size_t>(desc.kv_len) + HeadDim + kLargeRowThreads) * sizeof(float);
        if (shared_score_bytes <= kMaxSharedScoreBytes) {
          prefill_attention_hdim_forward_kernel<scalar_t, kLargeRowThreads, HeadDim>
              <<<blocks, kLargeRowThreads, shared_score_bytes, stream>>>(
                  q_contig.data_ptr<scalar_t>(),
                  k_contig.data_ptr<scalar_t>(),
                  v_contig.data_ptr<scalar_t>(),
                  mask_ptr,
                  mask_kind,
                  out.data_ptr<scalar_t>(),
                  desc.batch,
                  desc.q_heads,
                  desc.kv_heads,
                  desc.q_len,
                  desc.kv_len,
                  scale_value,
                  desc.causal);
        } else {
          prefill_attention_hdim_legacy_forward_kernel<scalar_t, kLargeRowThreads, HeadDim>
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
                  desc.q_len,
                  desc.kv_len,
                  scale_value,
                  desc.causal);
          }
      }
      return;
  }
}

template <typename scalar_t>
inline void LaunchPrefillAttentionMHA(
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
  const dim3 blocks(static_cast<unsigned int>(desc.batch * desc.q_heads * desc.q_len));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      prefill_attention_mha_forward_kernel<scalar_t, kSmallRowThreads, true>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.q_len,
              desc.kv_len,
              desc.head_dim,
              scale_value,
              desc.causal);
      return;
    case kMediumRowThreads:
      prefill_attention_mha_forward_kernel<scalar_t, kMediumRowThreads, true>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.q_len,
              desc.kv_len,
              desc.head_dim,
              scale_value,
              desc.causal);
      return;
    default:
      prefill_attention_mha_forward_kernel<scalar_t, kLargeRowThreads, true>
          <<<blocks, kLargeRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.q_len,
              desc.kv_len,
              desc.head_dim,
              scale_value,
              desc.causal);
      return;
  }
}

template <typename scalar_t>
inline void LaunchPrefillAttentionMHANoMask(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream,
    int row_reduce_threads) {
  const dim3 blocks(static_cast<unsigned int>(desc.batch * desc.q_heads * desc.q_len));
  switch (row_reduce_threads) {
    case kSmallRowThreads:
      prefill_attention_mha_forward_kernel<scalar_t, kSmallRowThreads, false>
          <<<blocks, kSmallRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.q_len,
              desc.kv_len,
              desc.head_dim,
              scale_value,
              desc.causal);
      return;
    case kMediumRowThreads:
      prefill_attention_mha_forward_kernel<scalar_t, kMediumRowThreads, false>
          <<<blocks, kMediumRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.q_len,
              desc.kv_len,
              desc.head_dim,
              scale_value,
              desc.causal);
      return;
    default:
      prefill_attention_mha_forward_kernel<scalar_t, kLargeRowThreads, false>
          <<<blocks, kLargeRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_contig.data_ptr<scalar_t>(),
              v_contig.data_ptr<scalar_t>(),
              nullptr,
              0,
              out.data_ptr<scalar_t>(),
              desc.batch,
              desc.q_heads,
              desc.q_len,
              desc.kv_len,
              desc.head_dim,
              scale_value,
              desc.causal);
      return;
  }
}

template <typename scalar_t>
inline void LaunchPlannedAttentionPrefill(
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
    case t10::policy::AttentionKernelKind::kGenericPrefillNoMask:
      LaunchGenericAttentionPrefillNoMask<scalar_t>(
          q_contig,
          k_contig,
          v_contig,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    case t10::policy::AttentionKernelKind::kPrefillMHANoMask:
      LaunchPrefillAttentionMHANoMask<scalar_t>(
          q_contig,
          k_contig,
          v_contig,
          out,
          desc,
          scale_value,
          stream,
          plan.row_reduce_threads);
      return;
    case t10::policy::AttentionKernelKind::kPrefillMHA:
      LaunchPrefillAttentionMHA<scalar_t>(
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
    case t10::policy::AttentionKernelKind::kPrefillHdim32:
      LaunchPrefillAttentionSpecialized<scalar_t, 32>(
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
    case t10::policy::AttentionKernelKind::kPrefillHdim64:
      LaunchPrefillAttentionSpecialized<scalar_t, 64>(
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
    case t10::policy::AttentionKernelKind::kPrefillHdim96:
      LaunchPrefillAttentionSpecialized<scalar_t, 96>(
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
    case t10::policy::AttentionKernelKind::kPrefillHdim128:
      LaunchPrefillAttentionSpecialized<scalar_t, 128>(
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
      LaunchGenericAttentionPrefill<scalar_t>(
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
