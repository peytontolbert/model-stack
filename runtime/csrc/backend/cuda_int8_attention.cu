#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "cuda_device_arch.cuh"
#include "cuda_hopper_advanced.cuh"

#include <cuda/barrier>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <sm_61_intrinsics.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>

namespace {

using namespace nvcuda;
using block_barrier_t = cuda::barrier<cuda::thread_scope_block>;

constexpr int kGenericThreads = 128;
constexpr int kTensorCoreThreads = 32;
constexpr int kTensorCoreTileM = 16;
constexpr int kTensorCoreTileN = 16;
constexpr int kTensorCoreTileK = 16;
constexpr int kWgmmaAttentionThreads = 128;
constexpr int kWgmmaAttentionScoreTileM = 64;
constexpr int kWgmmaAttentionActiveTileM = 32;
constexpr int kWgmmaAttentionTileN = 8;
constexpr int kWgmmaAttentionTileK = 32;
constexpr int kWgmmaAttentionLayoutNoSwizzle = 0;
constexpr int kSm90PipelineStages = 2;
constexpr int kDecodeMaxThreads = 256;
constexpr int kDecodeTileN = 128;
constexpr int kWarpSize = 32;

__host__ __device__ inline int64_t DivUpInt64(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

bool IsSupportedInt8AttentionOutDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

bool Int8AttentionOptimizedDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_ATTENTION_OPTIMIZED");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8AttentionTensorCoreDisabled() {
  if (Int8AttentionOptimizedDisabled()) {
    return true;
  }
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_ATTENTION_WMMA");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8AttentionSm90PipelineDisabled() {
  if (Int8AttentionOptimizedDisabled()) {
    return true;
  }
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_ATTENTION_SM90_PIPELINE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8AttentionSm90BulkAsyncDisabled() {
  if (Int8AttentionOptimizedDisabled()) {
    return true;
  }
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_ATTENTION_SM90_BULK_ASYNC");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8AttentionPersistentDisabled() {
  if (Int8AttentionOptimizedDisabled()) {
    return true;
  }
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_ATTENTION_PERSISTENT");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8AttentionPersistentEnabled() {
  const char* env = std::getenv("MODEL_STACK_ENABLE_INT8_ATTENTION_PERSISTENT");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8AttentionWgmmaDisabled() {
  if (Int8AttentionOptimizedDisabled()) {
    return true;
  }
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_ATTENTION_WGMMA");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8AttentionWgmmaEnabled() {
  const char* env = std::getenv("MODEL_STACK_ENABLE_INT8_ATTENTION_WGMMA");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

int Int8AttentionPersistentWaves() {
  const char* env = std::getenv("MODEL_STACK_INT8_ATTENTION_PERSISTENT_WAVES");
  if (env == nullptr || env[0] == '\0') {
    return 2;
  }
  return std::max(1, static_cast<int>(std::strtol(env, nullptr, 10)));
}

int64_t Int8AttentionWgmmaMinWork() {
  const char* env = std::getenv("MODEL_STACK_INT8_ATTENTION_WGMMA_MIN_WORK");
  if (env == nullptr || env[0] == '\0') {
    return 262144;
  }
  return std::max<int64_t>(1, std::strtoll(env, nullptr, 10));
}

int Int8AttentionSm90PipelineMinSeq() {
  const char* env = std::getenv("MODEL_STACK_SM90_ATTN_PIPELINE_MIN_SEQ");
  if (env == nullptr || env[0] == '\0') {
    return 64;
  }
  return std::max(1, static_cast<int>(std::strtol(env, nullptr, 10)));
}

int64_t Int8AttentionOptimizedMinWork() {
  const char* env = std::getenv("MODEL_STACK_INT8_ATTENTION_OPTIMIZED_MIN_WORK");
  if (env == nullptr || env[0] == '\0') {
    return 32768;
  }
  return std::max<int64_t>(1, std::strtoll(env, nullptr, 10));
}

int64_t Int8AttentionOptimizedSmallSeqMinHeadDim() {
  const char* env = std::getenv("MODEL_STACK_INT8_ATTENTION_OPTIMIZED_SMALL_SEQ_MIN_HEAD_DIM");
  if (env == nullptr || env[0] == '\0') {
    return 256;
  }
  return std::max<int64_t>(32, std::strtoll(env, nullptr, 10));
}

bool ShouldPreferInt8AttentionOptimizedPath(const torch::Tensor& q, const torch::Tensor& k) {
  if (Int8AttentionOptimizedDisabled()) {
    return false;
  }
  if (q.size(2) <= 1) {
    return false;
  }
  const int64_t work = q.size(2) * k.size(2);
  if (work >= Int8AttentionOptimizedMinWork()) {
    return true;
  }
  return q.size(3) >= Int8AttentionOptimizedSmallSeqMinHeadDim();
}

bool SupportsInt8AttentionTensorCorePath(const torch::Tensor& q) {
  if (Int8AttentionTensorCoreDisabled()) {
    return false;
  }
  const auto head_dim = q.size(3);
  if (head_dim < 16 || head_dim > 256 || (head_dim % 16) != 0) {
    return false;
  }
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, q.get_device()));
  return prop.major > 7 || (prop.major == 7 && prop.minor >= 5);
}

bool SupportsInt8AttentionSm90PipelinePath(const torch::Tensor& q, const torch::Tensor& k) {
  if (Int8AttentionSm90PipelineDisabled()) {
    return false;
  }
  if (!t10::cuda::DeviceIsSm90OrLater(q)) {
    return false;
  }
  const auto head_dim = q.size(3);
  if (head_dim < 16 || head_dim > 256 || (head_dim % 16) != 0) {
    return false;
  }
  const auto min_seq = Int8AttentionSm90PipelineMinSeq();
  return std::max<int64_t>(q.size(2), k.size(2)) >= min_seq;
}

bool SupportsInt8AttentionSm90BulkAsyncPath(const torch::Tensor& q, const torch::Tensor& k) {
  if (Int8AttentionSm90BulkAsyncDisabled()) {
    return false;
  }
  if (!t10::cuda::DeviceIsSm90OrLater(q)) {
    return false;
  }
  const auto head_dim = q.size(3);
  if (head_dim < 16 || head_dim > 256 || (head_dim % 16) != 0) {
    return false;
  }
  return std::max<int64_t>(q.size(2), k.size(2)) >= Int8AttentionSm90PipelineMinSeq();
}

bool SupportsInt8AttentionPersistentPath(const torch::Tensor& q, const torch::Tensor& k) {
  if (Int8AttentionPersistentDisabled() || !Int8AttentionPersistentEnabled()) {
    return false;
  }
  if (!t10::cuda::DeviceIsSm90OrLater(q)) {
    return false;
  }
  if (q.size(2) <= 1) {
    return false;
  }
  const auto head_dim = q.size(3);
  if (head_dim < 16 || head_dim > 256 || (head_dim % 16) != 0) {
    return false;
  }
  return ShouldPreferInt8AttentionOptimizedPath(q, k);
}

bool SupportsInt8AttentionWgmmaPath(const torch::Tensor& q, const torch::Tensor& k) {
  if (!t10::cuda::BuildRequestsSm90aExperimental()) {
    return false;
  }
  if (!Int8AttentionWgmmaEnabled() || Int8AttentionWgmmaDisabled()) {
    return false;
  }
  if (!t10::cuda::DeviceIsSm90OrLater(q)) {
    return false;
  }
  if (q.size(2) <= 1) {
    return false;
  }
  const auto head_dim = q.size(3);
  if (head_dim < kWgmmaAttentionTileK || head_dim > 128 || (head_dim % kWgmmaAttentionTileK) != 0) {
    return false;
  }
  if (k.size(2) < kWgmmaAttentionTileN) {
    return false;
  }
  const int64_t work = q.size(2) * k.size(2) * head_dim;
  return work >= Int8AttentionWgmmaMinWork() && ShouldPreferInt8AttentionOptimizedPath(q, k);
}

int64_t PersistentAttentionBlockCount(const torch::Tensor& q, int64_t total_blocks) {
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, q.get_device()));
  const int64_t target = static_cast<int64_t>(prop.multiProcessorCount) * Int8AttentionPersistentWaves();
  return std::max<int64_t>(1, std::min<int64_t>(total_blocks, target));
}

template <int HeadDim, int TileN, int Threads>
__device__ inline void async_copy_int8_attention_tile_generic(
    int8_t* __restrict__ k_dst,
    int8_t* __restrict__ v_dst,
    const int8_t* __restrict__ k_src,
    const int8_t* __restrict__ v_src,
    int valid_k) {
  const int lane = static_cast<int>(threadIdx.x);
  const int total_bytes = valid_k * HeadDim;
  for (int idx = lane; idx < TileN * HeadDim; idx += Threads) {
    if (idx >= total_bytes) {
      k_dst[idx] = static_cast<int8_t>(0);
      v_dst[idx] = static_cast<int8_t>(0);
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int kCopyBytes = 16;
  const int aligned = (total_bytes / kCopyBytes) * kCopyBytes;
  for (int byte_offset = lane * kCopyBytes; byte_offset < aligned; byte_offset += Threads * kCopyBytes) {
    __pipeline_memcpy_async(k_dst + byte_offset, k_src + byte_offset, kCopyBytes);
    __pipeline_memcpy_async(v_dst + byte_offset, v_src + byte_offset, kCopyBytes);
  }
  for (int idx = aligned + lane; idx < total_bytes; idx += Threads) {
    k_dst[idx] = k_src[idx];
    v_dst[idx] = v_src[idx];
  }
  __pipeline_commit();
#else
  for (int idx = lane; idx < total_bytes; idx += Threads) {
    k_dst[idx] = k_src[idx];
    v_dst[idx] = v_src[idx];
  }
#endif
}

template <int HeadDim>
__device__ inline void async_copy_int8_attention_tile(
    int8_t* __restrict__ k_dst,
    int8_t* __restrict__ v_dst,
    const int8_t* __restrict__ k_src,
    const int8_t* __restrict__ v_src,
    int valid_k) {
  async_copy_int8_attention_tile_generic<HeadDim, kTensorCoreTileN, kTensorCoreThreads>(
      k_dst, v_dst, k_src, v_src, valid_k);
}

bool SupportsInt8AttentionDecodeSpecializedPath(
    const torch::Tensor& q,
    bool has_bool_mask,
    bool has_additive_mask) {
  const char* env = std::getenv("MODEL_STACK_ENABLE_INT8_ATTENTION_DECODE_SPECIALIZED");
  if (Int8AttentionOptimizedDisabled() || env == nullptr || env[0] == '\0' || env[0] == '0') {
    return false;
  }
  if (has_bool_mask || has_additive_mask) {
    return false;
  }
  if (q.size(2) != 1) {
    return false;
  }
  const auto head_dim = q.size(3);
  if (head_dim < 4 || head_dim > 256 || (head_dim % 4) != 0) {
    return false;
  }
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, q.get_device()));
  return prop.major > 6 || (prop.major == 6 && prop.minor >= 1);
}

__device__ inline int DotInt8Packed4Attention(int32_t lhs_packed, int32_t rhs_packed, int acc) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(lhs_packed, rhs_packed, acc);
#else
  const int8_t* lhs = reinterpret_cast<const int8_t*>(&lhs_packed);
  const int8_t* rhs = reinterpret_cast<const int8_t*>(&rhs_packed);
  acc += static_cast<int>(lhs[0]) * static_cast<int>(rhs[0]);
  acc += static_cast<int>(lhs[1]) * static_cast<int>(rhs[1]);
  acc += static_cast<int>(lhs[2]) * static_cast<int>(rhs[2]);
  acc += static_cast<int>(lhs[3]) * static_cast<int>(rhs[3]);
  return acc;
#endif
}

__device__ inline float WarpReduceMaxFloat(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ inline float WarpReduceSumFloat(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void int8_attention_forward_generic_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ q_scale,
    const int8_t* __restrict__ k,
    const float* __restrict__ k_scale,
    const int8_t* __restrict__ v,
    const float* __restrict__ v_scale,
    const bool* __restrict__ bool_mask,
    const float* __restrict__ additive_mask,
    scalar_t* __restrict__ out,
    int64_t batch_size,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t tgt_len,
    int64_t src_len,
    int64_t head_dim,
    bool has_bool_mask,
    bool has_additive_mask,
    bool is_causal,
    float scale_value) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch_size * q_heads * tgt_len * head_dim;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % head_dim;
  const int64_t tmp0 = idx / head_dim;
  const int64_t t = tmp0 % tgt_len;
  const int64_t tmp1 = tmp0 / tgt_len;
  const int64_t h = tmp1 % q_heads;
  const int64_t b = tmp1 / q_heads;

  const int64_t head_repeat = q_heads / kv_heads;
  const int64_t kv_h = h / head_repeat;

  const int64_t q_row_idx = ((b * q_heads + h) * tgt_len) + t;
  const int8_t* q_row = q + (q_row_idx * head_dim);
  const bool use_packed_dp4a = (head_dim % 4) == 0;

  float max_score = -std::numeric_limits<float>::infinity();
  bool has_valid = false;
  for (int64_t s = 0; s < src_len; ++s) {
    if (is_causal && tgt_len > 1 && s > t) {
      continue;
    }
    const int64_t mask_idx = ((b * q_heads + h) * tgt_len + t) * src_len + s;
    if (has_bool_mask && bool_mask[mask_idx]) {
      continue;
    }
    const int64_t k_row_idx = ((b * kv_heads + kv_h) * src_len) + s;
    const int8_t* k_row = k + (k_row_idx * head_dim);
    int acc = 0;
    if (use_packed_dp4a) {
      const int64_t packed_chunks = head_dim / 4;
      const int32_t* q_packed = reinterpret_cast<const int32_t*>(q_row);
      const int32_t* k_packed = reinterpret_cast<const int32_t*>(k_row);
      for (int64_t i = 0; i < packed_chunks; ++i) {
        acc = DotInt8Packed4Attention(q_packed[i], k_packed[i], acc);
      }
    } else {
      for (int64_t i = 0; i < head_dim; ++i) {
        acc += static_cast<int>(q_row[i]) * static_cast<int>(k_row[i]);
      }
    }
    float score =
        static_cast<float>(acc) * q_scale[q_row_idx] * k_scale[k_row_idx] * scale_value;
    if (has_additive_mask) {
      score += additive_mask[mask_idx];
    }
    if (!::isfinite(score)) {
      continue;
    }
    has_valid = true;
    if (score > max_score) {
      max_score = score;
    }
  }

  if (!has_valid) {
    out[idx] = static_cast<scalar_t>(0.0f);
    return;
  }

  float denom = 0.0f;
  float weighted = 0.0f;
  for (int64_t s = 0; s < src_len; ++s) {
    if (is_causal && tgt_len > 1 && s > t) {
      continue;
    }
    const int64_t mask_idx = ((b * q_heads + h) * tgt_len + t) * src_len + s;
    if (has_bool_mask && bool_mask[mask_idx]) {
      continue;
    }
    const int64_t kv_row_idx = ((b * kv_heads + kv_h) * src_len) + s;
    const int8_t* k_row = k + (kv_row_idx * head_dim);
    int acc = 0;
    if (use_packed_dp4a) {
      const int64_t packed_chunks = head_dim / 4;
      const int32_t* q_packed = reinterpret_cast<const int32_t*>(q_row);
      const int32_t* k_packed = reinterpret_cast<const int32_t*>(k_row);
      for (int64_t i = 0; i < packed_chunks; ++i) {
        acc = DotInt8Packed4Attention(q_packed[i], k_packed[i], acc);
      }
    } else {
      for (int64_t i = 0; i < head_dim; ++i) {
        acc += static_cast<int>(q_row[i]) * static_cast<int>(k_row[i]);
      }
    }
    float score =
        static_cast<float>(acc) * q_scale[q_row_idx] * k_scale[kv_row_idx] * scale_value;
    if (has_additive_mask) {
      score += additive_mask[mask_idx];
    }
    if (!::isfinite(score)) {
      continue;
    }
    const float weight = expf(score - max_score);
    denom += weight;
    const int8_t* v_row = v + (kv_row_idx * head_dim);
    weighted += weight * (static_cast<float>(v_row[d]) * v_scale[kv_row_idx]);
  }

  const float out_value = weighted / fmaxf(denom, 1e-8f);
  out[idx] = static_cast<scalar_t>(out_value);
}

template <int Threads>
__device__ inline float BlockReduceMaxFloat(float value, float* warp_reduce) {
  constexpr int kWarpCount = Threads / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;

  value = WarpReduceMaxFloat(value);
  if (lane == 0) {
    warp_reduce[warp] = value;
  }
  __syncthreads();

  float block_value = -std::numeric_limits<float>::infinity();
  if (warp == 0) {
    block_value = lane < kWarpCount ? warp_reduce[lane] : -std::numeric_limits<float>::infinity();
    block_value = WarpReduceMaxFloat(block_value);
  }
  __syncthreads();
  return block_value;
}

template <int Threads>
__device__ inline float BlockReduceSumFloat(float value, float* warp_reduce) {
  constexpr int kWarpCount = Threads / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;

  value = WarpReduceSumFloat(value);
  if (lane == 0) {
    warp_reduce[warp] = value;
  }
  __syncthreads();

  float block_value = 0.0f;
  if (warp == 0) {
    block_value = lane < kWarpCount ? warp_reduce[lane] : 0.0f;
    block_value = WarpReduceSumFloat(block_value);
  }
  __syncthreads();
  return block_value;
}

template <typename scalar_t, int HeadDim, int TileN, int Threads>
__global__ void int8_attention_decode_nomask_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ q_scale,
    const int8_t* __restrict__ k,
    const float* __restrict__ k_scale,
    const int8_t* __restrict__ v,
    const float* __restrict__ v_scale,
    scalar_t* __restrict__ out,
    int64_t batch_size,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t src_len,
    float scale_value) {
  static_assert((Threads % kWarpSize) == 0, "decode threads must be warp-aligned");
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch_size * q_heads;
  if (row >= total_rows) {
    return;
  }

  const int64_t head_idx = row % q_heads;
  const int64_t batch_idx = row / q_heads;
  const int64_t head_repeat = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_repeat;

  const int64_t q_base = ((batch_idx * q_heads) + head_idx) * HeadDim;
  const int64_t kv_base = ((batch_idx * kv_heads) + kv_head_idx) * src_len * HeadDim;
  const int64_t scale_base = ((batch_idx * kv_heads) + kv_head_idx) * src_len;

  __shared__ int32_t q_packed[HeadDim / 4];
  __shared__ float q_scale_value;
  __shared__ float score_tile[TileN];
  __shared__ float weight_tile[TileN];
  __shared__ float v_scale_tile[TileN];
  __shared__ float warp_reduce[Threads / kWarpSize];
  __shared__ float max_score;
  __shared__ float denom;

  for (int idx = tid; idx < (HeadDim / 4); idx += Threads) {
    q_packed[idx] = reinterpret_cast<const int32_t*>(q + q_base)[idx];
  }
  if (tid == 0) {
    q_scale_value = q_scale[row];
    max_score = -std::numeric_limits<float>::infinity();
    denom = 0.0f;
  }
  __syncthreads();

  for (int64_t s_tile_start = 0; s_tile_start < src_len; s_tile_start += TileN) {
    const int valid_k = static_cast<int>(
        (src_len - s_tile_start) < TileN ? (src_len - s_tile_start) : TileN);

    float local_max = -std::numeric_limits<float>::infinity();
    if (tid < valid_k) {
      const int64_t kv_idx = s_tile_start + tid;
      const int8_t* k_row = k + kv_base + kv_idx * HeadDim;
      int acc = 0;
      #pragma unroll
      for (int chunk = 0; chunk < (HeadDim / 4); ++chunk) {
        const int32_t rhs_packed = reinterpret_cast<const int32_t*>(k_row)[chunk];
        acc = DotInt8Packed4Attention(q_packed[chunk], rhs_packed, acc);
      }
      local_max =
          static_cast<float>(acc) *
          q_scale_value *
          k_scale[scale_base + kv_idx] *
          scale_value;
    }
    const float tile_max = BlockReduceMaxFloat<Threads>(local_max, warp_reduce);
    if (tid == 0) {
      max_score = fmaxf(max_score, tile_max);
    }
    __syncthreads();
  }

  float accum = 0.0f;
  for (int64_t s_tile_start = 0; s_tile_start < src_len; s_tile_start += TileN) {
    const int valid_k = static_cast<int>(
        (src_len - s_tile_start) < TileN ? (src_len - s_tile_start) : TileN);

    float weight = 0.0f;
    if (tid < valid_k) {
      const int64_t kv_idx = s_tile_start + tid;
      const int8_t* k_row = k + kv_base + kv_idx * HeadDim;
      int acc = 0;
      #pragma unroll
      for (int chunk = 0; chunk < (HeadDim / 4); ++chunk) {
        const int32_t rhs_packed = reinterpret_cast<const int32_t*>(k_row)[chunk];
        acc = DotInt8Packed4Attention(q_packed[chunk], rhs_packed, acc);
      }
      const float score =
          static_cast<float>(acc) *
          q_scale_value *
          k_scale[scale_base + kv_idx] *
          scale_value;
      score_tile[tid] = score;
      weight = (::isfinite(score) && ::isfinite(max_score)) ? expf(score - max_score) : 0.0f;
      weight_tile[tid] = weight;
      v_scale_tile[tid] = v_scale[scale_base + kv_idx];
    } else if (tid < TileN) {
      score_tile[tid] = 0.0f;
      weight_tile[tid] = 0.0f;
      v_scale_tile[tid] = 0.0f;
    }
    __syncthreads();

    const float tile_sum = BlockReduceSumFloat<Threads>(weight, warp_reduce);
    if (tid == 0) {
      denom += tile_sum;
    }
    __syncthreads();

    if (tid < HeadDim) {
      #pragma unroll
      for (int col = 0; col < TileN; ++col) {
        if (col >= valid_k) {
          break;
        }
        const float v_value =
            static_cast<float>(v[kv_base + (s_tile_start + col) * HeadDim + tid]) *
            v_scale_tile[col];
        accum += weight_tile[col] * v_value;
      }
    }
    __syncthreads();
  }

  if (tid < HeadDim) {
    out[q_base + tid] = static_cast<scalar_t>(accum / fmaxf(denom, 1e-8f));
  }
}

template <int HeadDim>
__device__ inline void compute_int8_score_tile_tensorcore(
    const int8_t* q_tile,
    const int8_t* k_tile_col_major,
    int* score_tile) {
  wmma::fragment<wmma::accumulator, kTensorCoreTileM, kTensorCoreTileN, kTensorCoreTileK, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  #pragma unroll
  for (int chunk = 0; chunk < HeadDim; chunk += kTensorCoreTileK) {
    wmma::fragment<wmma::matrix_a, kTensorCoreTileM, kTensorCoreTileN, kTensorCoreTileK, signed char, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kTensorCoreTileM, kTensorCoreTileN, kTensorCoreTileK, signed char, wmma::col_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, reinterpret_cast<const signed char*>(q_tile + chunk), HeadDim);
    wmma::load_matrix_sync(
        b_frag,
        reinterpret_cast<const signed char*>(k_tile_col_major + chunk),
        HeadDim);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  wmma::store_matrix_sync(score_tile, acc_frag, kTensorCoreTileN, wmma::mem_row_major);
}

template <int HeadDim>
__device__ inline void compute_int8_score_tile_wgmma(
    const uint8_t* q_tile_rebased,
    const int8_t* k_tile_kmajor,
    const int* k_rebias_correction,
    int* score_tile,
    int valid_q,
    int valid_k) {
  static_assert((HeadDim % kWgmmaAttentionTileK) == 0, "WGMMA attention head_dim must be a multiple of 32");
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;

  int accum[4] = {0, 0, 0, 0};
  #pragma unroll
  for (int idx = 0; idx < 4; ++idx) {
    t10::cuda::WgmmaFenceOperand(accum[idx]);
  }
  t10::cuda::WgmmaFenceSyncAligned();

  for (int chunk = 0; chunk < HeadDim; chunk += kWgmmaAttentionTileK) {
    t10::cuda::AsyncProxyFenceSharedCta();
    const auto desc_a = static_cast<uint64_t>(t10::cuda::MakeWgmmaSmemDesc(
        q_tile_rebased + chunk,
        kWgmmaAttentionLayoutNoSwizzle,
        0,
        HeadDim * static_cast<int>(sizeof(uint8_t))));
    const auto desc_b = static_cast<uint64_t>(t10::cuda::MakeWgmmaSmemDesc(
        k_tile_kmajor + chunk * kWgmmaAttentionTileN,
        kWgmmaAttentionLayoutNoSwizzle,
        0,
        kWgmmaAttentionTileN * static_cast<int>(sizeof(int8_t))));
    t10::cuda::WgmmaM64N8K32S32U8S8(desc_a, desc_b, accum);
    t10::cuda::WgmmaCommitGroupSyncAligned();
    t10::cuda::WgmmaWaitGroupSyncAligned<0>();
    __syncthreads();
  }

  const int local_row = warp * 16 + (lane & 15);
  const int local_col_base = (lane >> 4) * 4;
  if (local_row >= valid_q || local_row >= kWgmmaAttentionActiveTileM) {
    return;
  }

  #pragma unroll
  for (int idx = 0; idx < 4; ++idx) {
    const int local_col = local_col_base + idx;
    if (local_col >= valid_k || local_col >= kWgmmaAttentionTileN) {
      continue;
    }
    score_tile[local_row * kWgmmaAttentionTileN + local_col] = accum[idx] - k_rebias_correction[local_col];
  }
}

template <typename scalar_t, int HeadDim, bool HasBoolMask, bool HasAdditiveMask, bool IsCausal, bool UseBulkAsync>
__global__ __launch_bounds__(kWgmmaAttentionThreads, 1) void int8_attention_forward_sm90a_wgmma_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ q_scale,
    const int8_t* __restrict__ k,
    const float* __restrict__ k_scale,
    const int8_t* __restrict__ v,
    const float* __restrict__ v_scale,
    const bool* __restrict__ bool_mask,
    const float* __restrict__ additive_mask,
    scalar_t* __restrict__ out,
    int64_t batch_size,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t tgt_len,
    int64_t src_len,
    float scale_value) {
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t tiles_per_head = DivUpInt64(tgt_len, kWgmmaAttentionActiveTileM);
  const int64_t total_blocks = batch_size * q_heads * tiles_per_head;

  __shared__ __align__(16) uint8_t q_tile[kWgmmaAttentionScoreTileM * HeadDim];
  __shared__ __align__(16) int8_t k_tile_row_major[kSm90PipelineStages][kWgmmaAttentionTileN * HeadDim];
  __shared__ __align__(16) int8_t k_tile_kmajor[kSm90PipelineStages][HeadDim * kWgmmaAttentionTileN];
  __shared__ __align__(16) int8_t v_tile[kSm90PipelineStages][kWgmmaAttentionTileN * HeadDim];
  __shared__ float q_row_scale[kWgmmaAttentionActiveTileM];
  __shared__ float k_row_scale[kSm90PipelineStages][kWgmmaAttentionTileN];
  __shared__ float v_row_scale[kSm90PipelineStages][kWgmmaAttentionTileN];
  __shared__ int k_rebias_correction[kSm90PipelineStages][kWgmmaAttentionTileN];
  __shared__ float row_max[kWgmmaAttentionActiveTileM];
  __shared__ float row_sum[kWgmmaAttentionActiveTileM];
  __shared__ float row_rescale[kWgmmaAttentionActiveTileM];
  __shared__ int score_tile[kWgmmaAttentionActiveTileM * kWgmmaAttentionTileN];
  __shared__ float weight_tile[kWgmmaAttentionActiveTileM * kWgmmaAttentionTileN];
  __shared__ float out_accum[kWgmmaAttentionActiveTileM * HeadDim];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier_t k_copy_bar[kSm90PipelineStages];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier_t v_copy_bar[kSm90PipelineStages];

  block_barrier_t::arrival_token k_tokens[kSm90PipelineStages];
  block_barrier_t::arrival_token v_tokens[kSm90PipelineStages];
  bool stage_pending[kSm90PipelineStages] = {false, false};

  for (int64_t logical_block = static_cast<int64_t>(blockIdx.x);
       logical_block < total_blocks;
       logical_block += static_cast<int64_t>(gridDim.x)) {
    const int64_t tile_idx = logical_block % tiles_per_head;
    const int64_t tmp = logical_block / tiles_per_head;
    const int64_t head_idx = tmp % q_heads;
    const int64_t batch_idx = tmp / q_heads;
    const int64_t q_tile_start = tile_idx * kWgmmaAttentionActiveTileM;
    const int valid_q = static_cast<int>(
        (tgt_len - q_tile_start) < kWgmmaAttentionActiveTileM ? (tgt_len - q_tile_start) : kWgmmaAttentionActiveTileM);
    if (valid_q <= 0) {
      continue;
    }

    const int64_t head_repeat = q_heads / kv_heads;
    const int64_t kv_head_idx = head_idx / head_repeat;

    if constexpr (UseBulkAsync) {
      if (tid < kSm90PipelineStages) {
        init(&k_copy_bar[tid], static_cast<ptrdiff_t>(kWgmmaAttentionThreads));
        init(&v_copy_bar[tid], static_cast<ptrdiff_t>(kWgmmaAttentionThreads));
      }
      if (tid == 0) {
        t10::cuda::Sm90FenceProxyAsyncSharedCta();
      }
      __syncthreads();
    }

    for (int idx = tid; idx < kWgmmaAttentionScoreTileM * HeadDim; idx += kWgmmaAttentionThreads) {
      const int row = idx / HeadDim;
      const int d = idx % HeadDim;
      if (row < valid_q) {
        const int64_t q_idx = q_tile_start + row;
        const int64_t q_base = (((batch_idx * q_heads) + head_idx) * tgt_len + q_idx) * HeadDim;
        q_tile[idx] = static_cast<uint8_t>(static_cast<int>(q[q_base + d]) + 128);
      } else {
        q_tile[idx] = 0;
      }
    }
    for (int row = tid; row < kWgmmaAttentionActiveTileM; row += kWgmmaAttentionThreads) {
      if (row < valid_q) {
        const int64_t q_idx = q_tile_start + row;
        q_row_scale[row] = q_scale[((batch_idx * q_heads) + head_idx) * tgt_len + q_idx];
        row_max[row] = -std::numeric_limits<float>::infinity();
        row_sum[row] = 0.0f;
        row_rescale[row] = 0.0f;
      } else {
        q_row_scale[row] = 0.0f;
        row_max[row] = -std::numeric_limits<float>::infinity();
        row_sum[row] = 0.0f;
        row_rescale[row] = 0.0f;
      }
    }
    for (int idx = tid; idx < kWgmmaAttentionActiveTileM * HeadDim; idx += kWgmmaAttentionThreads) {
      out_accum[idx] = 0.0f;
    }
    __syncthreads();

    const int first_valid_k = static_cast<int>(src_len < kWgmmaAttentionTileN ? src_len : kWgmmaAttentionTileN);
    const int first_copy_bytes = first_valid_k * HeadDim;
    if constexpr (UseBulkAsync) {
      if (tid == 0 && first_copy_bytes > 0) {
        const int64_t first_base = (((batch_idx * kv_heads) + kv_head_idx) * src_len) * HeadDim;
        t10::cuda::Sm90CpAsyncBulkGlobalToShared(
            k_tile_row_major[0],
            k + first_base,
            static_cast<uint32_t>(first_copy_bytes),
            k_copy_bar[0]);
        t10::cuda::Sm90CpAsyncBulkGlobalToShared(
            v_tile[0],
            v + first_base,
            static_cast<uint32_t>(first_copy_bytes),
            v_copy_bar[0]);
        k_tokens[0] = t10::cuda::Sm90BarrierArriveTx(k_copy_bar[0], 1, static_cast<uint32_t>(first_copy_bytes));
        v_tokens[0] = t10::cuda::Sm90BarrierArriveTx(v_copy_bar[0], 1, static_cast<uint32_t>(first_copy_bytes));
      } else {
        k_tokens[0] = k_copy_bar[0].arrive();
        v_tokens[0] = v_copy_bar[0].arrive();
      }
      stage_pending[0] = true;
    } else {
      async_copy_int8_attention_tile_generic<HeadDim, kWgmmaAttentionTileN, kWgmmaAttentionThreads>(
          k_tile_row_major[0],
          v_tile[0],
          k + (((batch_idx * kv_heads) + kv_head_idx) * src_len) * HeadDim,
          v + (((batch_idx * kv_heads) + kv_head_idx) * src_len) * HeadDim,
          first_valid_k);
    }
    for (int col = tid; col < kWgmmaAttentionTileN; col += kWgmmaAttentionThreads) {
      if (col < first_valid_k) {
        const int64_t scale_idx = (((batch_idx * kv_heads) + kv_head_idx) * src_len) + col;
        k_row_scale[0][col] = k_scale[scale_idx];
        v_row_scale[0][col] = v_scale[scale_idx];
      } else {
        k_row_scale[0][col] = 0.0f;
        v_row_scale[0][col] = 0.0f;
      }
    }
    if constexpr (UseBulkAsync) {
      __syncthreads();
    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      __pipeline_wait_prior(0);
#endif
    }
    __syncthreads();

    for (int64_t s_tile_start = 0; s_tile_start < src_len; s_tile_start += kWgmmaAttentionTileN) {
      const int current_stage = static_cast<int>((s_tile_start / kWgmmaAttentionTileN) % kSm90PipelineStages);
      const int next_stage = (current_stage + 1) % kSm90PipelineStages;
      const int valid_k = static_cast<int>(
          (src_len - s_tile_start) < kWgmmaAttentionTileN ? (src_len - s_tile_start) : kWgmmaAttentionTileN);
      const int64_t next_s_tile_start = s_tile_start + kWgmmaAttentionTileN;

      if constexpr (UseBulkAsync) {
        if (stage_pending[current_stage]) {
          k_copy_bar[current_stage].wait(std::move(k_tokens[current_stage]));
          v_copy_bar[current_stage].wait(std::move(v_tokens[current_stage]));
          stage_pending[current_stage] = false;
        }
        __syncthreads();
      }

      if (next_s_tile_start < src_len) {
        const int next_valid_k = static_cast<int>(
            (src_len - next_s_tile_start) < kWgmmaAttentionTileN ? (src_len - next_s_tile_start) : kWgmmaAttentionTileN);
        const int64_t next_base = ((((batch_idx * kv_heads) + kv_head_idx) * src_len) + next_s_tile_start) * HeadDim;
        if constexpr (UseBulkAsync) {
          const int next_copy_bytes = next_valid_k * HeadDim;
          if (tid == 0 && next_copy_bytes > 0) {
            t10::cuda::Sm90CpAsyncBulkGlobalToShared(
                k_tile_row_major[next_stage],
                k + next_base,
                static_cast<uint32_t>(next_copy_bytes),
                k_copy_bar[next_stage]);
            t10::cuda::Sm90CpAsyncBulkGlobalToShared(
                v_tile[next_stage],
                v + next_base,
                static_cast<uint32_t>(next_copy_bytes),
                v_copy_bar[next_stage]);
            k_tokens[next_stage] =
                t10::cuda::Sm90BarrierArriveTx(k_copy_bar[next_stage], 1, static_cast<uint32_t>(next_copy_bytes));
            v_tokens[next_stage] =
                t10::cuda::Sm90BarrierArriveTx(v_copy_bar[next_stage], 1, static_cast<uint32_t>(next_copy_bytes));
          } else {
            k_tokens[next_stage] = k_copy_bar[next_stage].arrive();
            v_tokens[next_stage] = v_copy_bar[next_stage].arrive();
          }
          stage_pending[next_stage] = true;
        } else {
          async_copy_int8_attention_tile_generic<HeadDim, kWgmmaAttentionTileN, kWgmmaAttentionThreads>(
              k_tile_row_major[next_stage],
              v_tile[next_stage],
              k + next_base,
              v + next_base,
              next_valid_k);
        }
        for (int col = tid; col < kWgmmaAttentionTileN; col += kWgmmaAttentionThreads) {
          if (col < next_valid_k) {
            const int64_t scale_idx = (((batch_idx * kv_heads) + kv_head_idx) * src_len) + next_s_tile_start + col;
            k_row_scale[next_stage][col] = k_scale[scale_idx];
            v_row_scale[next_stage][col] = v_scale[scale_idx];
          } else {
            k_row_scale[next_stage][col] = 0.0f;
            v_row_scale[next_stage][col] = 0.0f;
          }
        }
      }

      for (int idx = tid; idx < HeadDim * kWgmmaAttentionTileN; idx += kWgmmaAttentionThreads) {
        const int d = idx / kWgmmaAttentionTileN;
        const int col = idx % kWgmmaAttentionTileN;
        k_tile_kmajor[current_stage][idx] =
            col < valid_k ? k_tile_row_major[current_stage][col * HeadDim + d] : static_cast<int8_t>(0);
      }
      for (int col = tid; col < kWgmmaAttentionTileN; col += kWgmmaAttentionThreads) {
        int sum = 0;
        if (col < valid_k) {
          for (int d = 0; d < HeadDim; ++d) {
            sum += static_cast<int>(k_tile_kmajor[current_stage][d * kWgmmaAttentionTileN + col]);
          }
        }
        k_rebias_correction[current_stage][col] = sum * 128;
      }
      __syncthreads();

      compute_int8_score_tile_wgmma<HeadDim>(
          q_tile,
          k_tile_kmajor[current_stage],
          k_rebias_correction[current_stage],
          score_tile,
          valid_q,
          valid_k);
      __syncthreads();

      for (int row = tid; row < valid_q; row += kWgmmaAttentionThreads) {
        const float prev_row_max = row_max[row];
        const float prev_row_sum = row_sum[row];
        float tile_max = -std::numeric_limits<float>::infinity();
        const int64_t q_idx = q_tile_start + row;
        const int64_t mask_row_base = (((batch_idx * q_heads) + head_idx) * tgt_len + q_idx) * src_len;

        #pragma unroll
        for (int col = 0; col < kWgmmaAttentionTileN; ++col) {
          if (col >= valid_k) {
            break;
          }
          const int64_t kv_idx = s_tile_start + col;
          if constexpr (IsCausal) {
            if (tgt_len > 1 && kv_idx > q_idx) {
              continue;
            }
          }
          if constexpr (HasBoolMask) {
            if (bool_mask[mask_row_base + kv_idx]) {
              continue;
            }
          }
          float score =
              static_cast<float>(score_tile[row * kWgmmaAttentionTileN + col]) *
              q_row_scale[row] *
              k_row_scale[current_stage][col] *
              scale_value;
          if constexpr (HasAdditiveMask) {
            score += additive_mask[mask_row_base + kv_idx];
          }
          if (!::isfinite(score)) {
            continue;
          }
          tile_max = fmaxf(tile_max, score);
        }

        const float new_row_max = fmaxf(prev_row_max, tile_max);
        float prev_scale = 0.0f;
        if (::isfinite(prev_row_max) && ::isfinite(new_row_max)) {
          prev_scale = expf(prev_row_max - new_row_max);
        }
        float tile_sum = 0.0f;
        #pragma unroll
        for (int col = 0; col < kWgmmaAttentionTileN; ++col) {
          float weight = 0.0f;
          if (col < valid_k) {
            const int64_t kv_idx = s_tile_start + col;
            bool allowed = true;
            if constexpr (IsCausal) {
              if (tgt_len > 1 && kv_idx > q_idx) {
                allowed = false;
              }
            }
            if constexpr (HasBoolMask) {
              if (bool_mask[mask_row_base + kv_idx]) {
                allowed = false;
              }
            }
            if (allowed) {
              float score =
                  static_cast<float>(score_tile[row * kWgmmaAttentionTileN + col]) *
                  q_row_scale[row] *
                  k_row_scale[current_stage][col] *
                  scale_value;
              if constexpr (HasAdditiveMask) {
                score += additive_mask[mask_row_base + kv_idx];
              }
              if (::isfinite(score) && ::isfinite(new_row_max)) {
                weight = expf(score - new_row_max);
              }
            }
          }
          weight_tile[row * kWgmmaAttentionTileN + col] = weight;
          tile_sum += weight;
        }

        row_max[row] = new_row_max;
        row_sum[row] = prev_row_sum * prev_scale + tile_sum;
        row_rescale[row] = prev_scale;
      }
      __syncthreads();

      for (int idx = tid; idx < valid_q * HeadDim; idx += kWgmmaAttentionThreads) {
        const int row = idx / HeadDim;
        const int d = idx % HeadDim;
        float accum = out_accum[idx] * row_rescale[row];
        #pragma unroll
        for (int col = 0; col < kWgmmaAttentionTileN; ++col) {
          if (col >= valid_k) {
            break;
          }
          const float weight = weight_tile[row * kWgmmaAttentionTileN + col];
          if (weight != 0.0f) {
            accum += weight * (static_cast<float>(v_tile[current_stage][col * HeadDim + d]) * v_row_scale[current_stage][col]);
          }
        }
        out_accum[idx] = accum;
      }
      if constexpr (!UseBulkAsync) {
        if (next_s_tile_start < src_len) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
          __pipeline_wait_prior(0);
#endif
        }
      }
      __syncthreads();
    }

    for (int idx = tid; idx < valid_q * HeadDim; idx += kWgmmaAttentionThreads) {
      const int row = idx / HeadDim;
      const int d = idx % HeadDim;
      const float denom = fmaxf(row_sum[row], 1e-8f);
      const float value = out_accum[idx] / denom;
      const int64_t q_idx = q_tile_start + row;
      const int64_t out_base = (((batch_idx * q_heads) + head_idx) * tgt_len + q_idx) * HeadDim;
      out[out_base + d] = static_cast<scalar_t>(value);
    }
    __syncthreads();
  }
}

template <typename scalar_t, int HeadDim, bool HasBoolMask, bool HasAdditiveMask, bool IsCausal>
__global__ void int8_attention_forward_tensorcore_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ q_scale,
    const int8_t* __restrict__ k,
    const float* __restrict__ k_scale,
    const int8_t* __restrict__ v,
    const float* __restrict__ v_scale,
    const bool* __restrict__ bool_mask,
    const float* __restrict__ additive_mask,
    scalar_t* __restrict__ out,
    int64_t batch_size,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t tgt_len,
    int64_t src_len,
    float scale_value) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t tiles_per_head = DivUpInt64(tgt_len, kTensorCoreTileM);
  const int64_t total_blocks = batch_size * q_heads * tiles_per_head;

  __shared__ int8_t q_tile[kTensorCoreTileM * HeadDim];
  __shared__ int8_t k_tile_col_major[kTensorCoreTileN * HeadDim];
  __shared__ int8_t v_tile[kTensorCoreTileN * HeadDim];
  __shared__ float q_row_scale[kTensorCoreTileM];
  __shared__ float k_row_scale[kTensorCoreTileN];
  __shared__ float v_row_scale[kTensorCoreTileN];
  __shared__ float row_max[kTensorCoreTileM];
  __shared__ float row_sum[kTensorCoreTileM];
  __shared__ float row_rescale[kTensorCoreTileM];
  __shared__ int score_tile[kTensorCoreTileM * kTensorCoreTileN];
  __shared__ float weight_tile[kTensorCoreTileM * kTensorCoreTileN];
  __shared__ float out_accum[kTensorCoreTileM * HeadDim];

  for (int64_t logical_block = static_cast<int64_t>(blockIdx.x);
       logical_block < total_blocks;
       logical_block += static_cast<int64_t>(gridDim.x)) {
    const int64_t tile_idx = logical_block % tiles_per_head;
    const int64_t tmp = logical_block / tiles_per_head;
    const int64_t head_idx = tmp % q_heads;
    const int64_t batch_idx = tmp / q_heads;
    const int64_t q_tile_start = tile_idx * kTensorCoreTileM;
    const int valid_q = static_cast<int>(
        (tgt_len - q_tile_start) < kTensorCoreTileM ? (tgt_len - q_tile_start) : kTensorCoreTileM);
    if (valid_q <= 0) {
      continue;
    }

    const int64_t head_repeat = q_heads / kv_heads;
    const int64_t kv_head_idx = head_idx / head_repeat;

    for (int idx = lane; idx < kTensorCoreTileM * HeadDim; idx += kTensorCoreThreads) {
      const int row = idx / HeadDim;
      const int d = idx % HeadDim;
      if (row < valid_q) {
        const int64_t q_idx = q_tile_start + row;
        const int64_t q_base = (((batch_idx * q_heads) + head_idx) * tgt_len + q_idx) * HeadDim;
        q_tile[idx] = q[q_base + d];
        out_accum[idx] = 0.0f;
      } else {
        q_tile[idx] = 0;
        out_accum[idx] = 0.0f;
      }
    }
    if (lane < kTensorCoreTileM) {
      if (lane < valid_q) {
        const int64_t q_idx = q_tile_start + lane;
        q_row_scale[lane] = q_scale[((batch_idx * q_heads) + head_idx) * tgt_len + q_idx];
        row_max[lane] = -std::numeric_limits<float>::infinity();
        row_sum[lane] = 0.0f;
        row_rescale[lane] = 0.0f;
      } else {
        q_row_scale[lane] = 0.0f;
        row_max[lane] = -std::numeric_limits<float>::infinity();
        row_sum[lane] = 0.0f;
        row_rescale[lane] = 0.0f;
      }
    }
    __syncthreads();

    for (int64_t s_tile_start = 0; s_tile_start < src_len; s_tile_start += kTensorCoreTileN) {
      const int valid_k = static_cast<int>(
          (src_len - s_tile_start) < kTensorCoreTileN ? (src_len - s_tile_start) : kTensorCoreTileN);

      for (int idx = lane; idx < kTensorCoreTileN * HeadDim; idx += kTensorCoreThreads) {
        const int col = idx / HeadDim;
        const int d = idx % HeadDim;
        if (col < valid_k) {
          const int64_t kv_idx = s_tile_start + col;
          const int64_t kv_base = (((batch_idx * kv_heads) + kv_head_idx) * src_len + kv_idx) * HeadDim;
          k_tile_col_major[col * HeadDim + d] = k[kv_base + d];
          v_tile[col * HeadDim + d] = v[kv_base + d];
        } else {
          k_tile_col_major[col * HeadDim + d] = 0;
          v_tile[col * HeadDim + d] = 0;
        }
      }
      if (lane < kTensorCoreTileN) {
        if (lane < valid_k) {
          const int64_t kv_idx = s_tile_start + lane;
          const int64_t scale_idx = (((batch_idx * kv_heads) + kv_head_idx) * src_len) + kv_idx;
          k_row_scale[lane] = k_scale[scale_idx];
          v_row_scale[lane] = v_scale[scale_idx];
        } else {
          k_row_scale[lane] = 0.0f;
          v_row_scale[lane] = 0.0f;
        }
      }
      __syncthreads();

      compute_int8_score_tile_tensorcore<HeadDim>(q_tile, k_tile_col_major, score_tile);
      __syncthreads();

      if (lane < kTensorCoreTileM && lane < valid_q) {
        const float prev_row_max = row_max[lane];
        const float prev_row_sum = row_sum[lane];
        float tile_max = -std::numeric_limits<float>::infinity();
        const int64_t q_idx = q_tile_start + lane;
        const int64_t mask_row_base = (((batch_idx * q_heads) + head_idx) * tgt_len + q_idx) * src_len;

        #pragma unroll
        for (int col = 0; col < kTensorCoreTileN; ++col) {
          if (col >= valid_k) {
            break;
          }
          const int64_t kv_idx = s_tile_start + col;
          if constexpr (IsCausal) {
            if (tgt_len > 1 && kv_idx > q_idx) {
              continue;
            }
          }
          if constexpr (HasBoolMask) {
            if (bool_mask[mask_row_base + kv_idx]) {
              continue;
            }
          }
          float score =
              static_cast<float>(score_tile[lane * kTensorCoreTileN + col]) *
              q_row_scale[lane] *
              k_row_scale[col] *
              scale_value;
          if constexpr (HasAdditiveMask) {
            score += additive_mask[mask_row_base + kv_idx];
          }
          if (!::isfinite(score)) {
            continue;
          }
          tile_max = fmaxf(tile_max, score);
        }

        const float new_row_max = fmaxf(prev_row_max, tile_max);
        float prev_scale = 0.0f;
        if (::isfinite(prev_row_max) && ::isfinite(new_row_max)) {
          prev_scale = expf(prev_row_max - new_row_max);
        }
        float tile_sum = 0.0f;
        #pragma unroll
        for (int col = 0; col < kTensorCoreTileN; ++col) {
          float weight = 0.0f;
          if (col < valid_k) {
            const int64_t kv_idx = s_tile_start + col;
            bool allowed = true;
            if constexpr (IsCausal) {
              if (tgt_len > 1 && kv_idx > q_idx) {
                allowed = false;
              }
            }
            if constexpr (HasBoolMask) {
              if (bool_mask[mask_row_base + kv_idx]) {
                allowed = false;
              }
            }
            if (allowed) {
              float score =
                  static_cast<float>(score_tile[lane * kTensorCoreTileN + col]) *
                  q_row_scale[lane] *
                  k_row_scale[col] *
                  scale_value;
              if constexpr (HasAdditiveMask) {
                score += additive_mask[mask_row_base + kv_idx];
              }
              if (::isfinite(score) && ::isfinite(new_row_max)) {
                weight = expf(score - new_row_max);
              }
            }
          }
          weight_tile[lane * kTensorCoreTileN + col] = weight;
          tile_sum += weight;
        }

        row_max[lane] = new_row_max;
        row_sum[lane] = prev_row_sum * prev_scale + tile_sum;
        row_rescale[lane] = prev_scale;
      }
      __syncthreads();

      for (int idx = lane; idx < valid_q * HeadDim; idx += kTensorCoreThreads) {
        const int row = idx / HeadDim;
        const int d = idx % HeadDim;
        float accum = out_accum[idx] * row_rescale[row];
        #pragma unroll
        for (int col = 0; col < kTensorCoreTileN; ++col) {
          if (col >= valid_k) {
            break;
          }
          const float weight = weight_tile[row * kTensorCoreTileN + col];
          if (weight != 0.0f) {
            accum += weight * (static_cast<float>(v_tile[col * HeadDim + d]) * v_row_scale[col]);
          }
        }
        out_accum[idx] = accum;
      }
      __syncthreads();
    }

    for (int idx = lane; idx < valid_q * HeadDim; idx += kTensorCoreThreads) {
      const int row = idx / HeadDim;
      const int d = idx % HeadDim;
      const float denom = fmaxf(row_sum[row], 1e-8f);
      const float value = out_accum[idx] / denom;
      const int64_t q_idx = q_tile_start + row;
      const int64_t out_base = (((batch_idx * q_heads) + head_idx) * tgt_len + q_idx) * HeadDim;
      out[out_base + d] = static_cast<scalar_t>(value);
    }
    __syncthreads();
  }
}

template <typename scalar_t, int HeadDim, bool HasBoolMask, bool HasAdditiveMask, bool IsCausal, bool UseBulkAsync>
__global__ void int8_attention_forward_sm90_pipeline_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ q_scale,
    const int8_t* __restrict__ k,
    const float* __restrict__ k_scale,
    const int8_t* __restrict__ v,
    const float* __restrict__ v_scale,
    const bool* __restrict__ bool_mask,
    const float* __restrict__ additive_mask,
    scalar_t* __restrict__ out,
    int64_t batch_size,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t tgt_len,
    int64_t src_len,
    float scale_value) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t tiles_per_head = DivUpInt64(tgt_len, kTensorCoreTileM);
  const int64_t total_blocks = batch_size * q_heads * tiles_per_head;

  __shared__ alignas(16) int8_t q_tile[kTensorCoreTileM * HeadDim];
  __shared__ alignas(16) int8_t k_tile_col_major[kSm90PipelineStages][kTensorCoreTileN * HeadDim];
  __shared__ alignas(16) int8_t v_tile[kSm90PipelineStages][kTensorCoreTileN * HeadDim];
  __shared__ float q_row_scale[kTensorCoreTileM];
  __shared__ float k_row_scale[kSm90PipelineStages][kTensorCoreTileN];
  __shared__ float v_row_scale[kSm90PipelineStages][kTensorCoreTileN];
  __shared__ float row_max[kTensorCoreTileM];
  __shared__ float row_sum[kTensorCoreTileM];
  __shared__ float row_rescale[kTensorCoreTileM];
  __shared__ int score_tile[kTensorCoreTileM * kTensorCoreTileN];
  __shared__ float weight_tile[kTensorCoreTileM * kTensorCoreTileN];
  __shared__ float out_accum[kTensorCoreTileM * HeadDim];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier_t k_copy_bar[kSm90PipelineStages];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ block_barrier_t v_copy_bar[kSm90PipelineStages];

  block_barrier_t::arrival_token k_tokens[kSm90PipelineStages];
  block_barrier_t::arrival_token v_tokens[kSm90PipelineStages];
  bool stage_pending[kSm90PipelineStages] = {false, false};

  for (int64_t logical_block = static_cast<int64_t>(blockIdx.x);
       logical_block < total_blocks;
       logical_block += static_cast<int64_t>(gridDim.x)) {
    const int64_t tile_idx = logical_block % tiles_per_head;
    const int64_t tmp = logical_block / tiles_per_head;
    const int64_t head_idx = tmp % q_heads;
    const int64_t batch_idx = tmp / q_heads;
    const int64_t q_tile_start = tile_idx * kTensorCoreTileM;
    const int valid_q = static_cast<int>(
        (tgt_len - q_tile_start) < kTensorCoreTileM ? (tgt_len - q_tile_start) : kTensorCoreTileM);
    if (valid_q <= 0) {
      continue;
    }

    const int64_t head_repeat = q_heads / kv_heads;
    const int64_t kv_head_idx = head_idx / head_repeat;

    if constexpr (UseBulkAsync) {
      if (lane < kSm90PipelineStages) {
        init(&k_copy_bar[lane], static_cast<ptrdiff_t>(kTensorCoreThreads));
        init(&v_copy_bar[lane], static_cast<ptrdiff_t>(kTensorCoreThreads));
      }
      if (lane == 0) {
        t10::cuda::Sm90FenceProxyAsyncSharedCta();
      }
      __syncthreads();
    }

    for (int idx = lane; idx < kTensorCoreTileM * HeadDim; idx += kTensorCoreThreads) {
      const int row = idx / HeadDim;
      const int d = idx % HeadDim;
      if (row < valid_q) {
        const int64_t q_idx = q_tile_start + row;
        const int64_t q_base = (((batch_idx * q_heads) + head_idx) * tgt_len + q_idx) * HeadDim;
        q_tile[idx] = q[q_base + d];
        out_accum[idx] = 0.0f;
      } else {
        q_tile[idx] = 0;
        out_accum[idx] = 0.0f;
      }
    }
    if (lane < kTensorCoreTileM) {
      if (lane < valid_q) {
        const int64_t q_idx = q_tile_start + lane;
        q_row_scale[lane] = q_scale[((batch_idx * q_heads) + head_idx) * tgt_len + q_idx];
        row_max[lane] = -std::numeric_limits<float>::infinity();
        row_sum[lane] = 0.0f;
        row_rescale[lane] = 0.0f;
      } else {
        q_row_scale[lane] = 0.0f;
        row_max[lane] = -std::numeric_limits<float>::infinity();
        row_sum[lane] = 0.0f;
        row_rescale[lane] = 0.0f;
      }
    }
    __syncthreads();

    const int first_valid_k = static_cast<int>(src_len < kTensorCoreTileN ? src_len : kTensorCoreTileN);
    const int first_copy_bytes = first_valid_k * HeadDim;
    if constexpr (UseBulkAsync) {
      if (lane == 0 && first_copy_bytes > 0) {
        t10::cuda::Sm90CpAsyncBulkGlobalToShared(
            k_tile_col_major[0],
            k + (((batch_idx * kv_heads) + kv_head_idx) * src_len) * HeadDim,
            static_cast<uint32_t>(first_copy_bytes),
            k_copy_bar[0]);
        t10::cuda::Sm90CpAsyncBulkGlobalToShared(
            v_tile[0],
            v + (((batch_idx * kv_heads) + kv_head_idx) * src_len) * HeadDim,
            static_cast<uint32_t>(first_copy_bytes),
            v_copy_bar[0]);
        k_tokens[0] = t10::cuda::Sm90BarrierArriveTx(k_copy_bar[0], 1, static_cast<uint32_t>(first_copy_bytes));
        v_tokens[0] = t10::cuda::Sm90BarrierArriveTx(v_copy_bar[0], 1, static_cast<uint32_t>(first_copy_bytes));
      } else {
        k_tokens[0] = k_copy_bar[0].arrive();
        v_tokens[0] = v_copy_bar[0].arrive();
      }
      stage_pending[0] = true;
    } else {
      async_copy_int8_attention_tile<HeadDim>(
          k_tile_col_major[0],
          v_tile[0],
          k + (((batch_idx * kv_heads) + kv_head_idx) * src_len) * HeadDim,
          v + (((batch_idx * kv_heads) + kv_head_idx) * src_len) * HeadDim,
          first_valid_k);
    }
    if (lane < kTensorCoreTileN) {
      if (lane < first_valid_k) {
        const int64_t scale_idx = (((batch_idx * kv_heads) + kv_head_idx) * src_len) + lane;
        k_row_scale[0][lane] = k_scale[scale_idx];
        v_row_scale[0][lane] = v_scale[scale_idx];
      } else {
        k_row_scale[0][lane] = 0.0f;
        v_row_scale[0][lane] = 0.0f;
      }
    }
    if constexpr (UseBulkAsync) {
      __syncthreads();
    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      __pipeline_wait_prior(0);
#endif
    }
    __syncthreads();

    for (int64_t s_tile_start = 0; s_tile_start < src_len; s_tile_start += kTensorCoreTileN) {
      const int current_stage = static_cast<int>((s_tile_start / kTensorCoreTileN) % kSm90PipelineStages);
      const int next_stage = (current_stage + 1) % kSm90PipelineStages;
      const int valid_k = static_cast<int>(
          (src_len - s_tile_start) < kTensorCoreTileN ? (src_len - s_tile_start) : kTensorCoreTileN);
      const int64_t next_s_tile_start = s_tile_start + kTensorCoreTileN;

      if constexpr (UseBulkAsync) {
        if (stage_pending[current_stage]) {
          k_copy_bar[current_stage].wait(std::move(k_tokens[current_stage]));
          v_copy_bar[current_stage].wait(std::move(v_tokens[current_stage]));
          stage_pending[current_stage] = false;
        }
        __syncthreads();
      }

      if (next_s_tile_start < src_len) {
        const int next_valid_k = static_cast<int>(
            (src_len - next_s_tile_start) < kTensorCoreTileN ? (src_len - next_s_tile_start) : kTensorCoreTileN);
        const int64_t next_base = ((((batch_idx * kv_heads) + kv_head_idx) * src_len) + next_s_tile_start) * HeadDim;
        if constexpr (UseBulkAsync) {
          const int next_copy_bytes = next_valid_k * HeadDim;
          if (lane == 0 && next_copy_bytes > 0) {
            t10::cuda::Sm90CpAsyncBulkGlobalToShared(
                k_tile_col_major[next_stage],
                k + next_base,
                static_cast<uint32_t>(next_copy_bytes),
                k_copy_bar[next_stage]);
            t10::cuda::Sm90CpAsyncBulkGlobalToShared(
                v_tile[next_stage],
                v + next_base,
                static_cast<uint32_t>(next_copy_bytes),
                v_copy_bar[next_stage]);
            k_tokens[next_stage] = t10::cuda::Sm90BarrierArriveTx(
                k_copy_bar[next_stage], 1, static_cast<uint32_t>(next_copy_bytes));
            v_tokens[next_stage] = t10::cuda::Sm90BarrierArriveTx(
                v_copy_bar[next_stage], 1, static_cast<uint32_t>(next_copy_bytes));
          } else {
            k_tokens[next_stage] = k_copy_bar[next_stage].arrive();
            v_tokens[next_stage] = v_copy_bar[next_stage].arrive();
          }
          stage_pending[next_stage] = true;
        } else {
          async_copy_int8_attention_tile<HeadDim>(
              k_tile_col_major[next_stage],
              v_tile[next_stage],
              k + next_base,
              v + next_base,
              next_valid_k);
        }
        if (lane < kTensorCoreTileN) {
          if (lane < next_valid_k) {
            const int64_t scale_idx =
                (((batch_idx * kv_heads) + kv_head_idx) * src_len) + next_s_tile_start + lane;
            k_row_scale[next_stage][lane] = k_scale[scale_idx];
            v_row_scale[next_stage][lane] = v_scale[scale_idx];
          } else {
            k_row_scale[next_stage][lane] = 0.0f;
            v_row_scale[next_stage][lane] = 0.0f;
          }
        }
      }

      compute_int8_score_tile_tensorcore<HeadDim>(q_tile, k_tile_col_major[current_stage], score_tile);
      __syncthreads();

      if (lane < kTensorCoreTileM && lane < valid_q) {
        const float prev_row_max = row_max[lane];
        const float prev_row_sum = row_sum[lane];
        float tile_max = -std::numeric_limits<float>::infinity();
        const int64_t q_idx = q_tile_start + lane;
        const int64_t mask_row_base = (((batch_idx * q_heads) + head_idx) * tgt_len + q_idx) * src_len;

        #pragma unroll
        for (int col = 0; col < kTensorCoreTileN; ++col) {
          if (col >= valid_k) {
            break;
          }
          const int64_t kv_idx = s_tile_start + col;
          if constexpr (IsCausal) {
            if (tgt_len > 1 && kv_idx > q_idx) {
              continue;
            }
          }
          if constexpr (HasBoolMask) {
            if (bool_mask[mask_row_base + kv_idx]) {
              continue;
            }
          }
          float score =
              static_cast<float>(score_tile[lane * kTensorCoreTileN + col]) *
              q_row_scale[lane] *
              k_row_scale[current_stage][col] *
              scale_value;
          if constexpr (HasAdditiveMask) {
            score += additive_mask[mask_row_base + kv_idx];
          }
          if (!::isfinite(score)) {
            continue;
          }
          tile_max = fmaxf(tile_max, score);
        }

        const float new_row_max = fmaxf(prev_row_max, tile_max);
        float prev_scale = 0.0f;
        if (::isfinite(prev_row_max) && ::isfinite(new_row_max)) {
          prev_scale = expf(prev_row_max - new_row_max);
        }
        float tile_sum = 0.0f;
        #pragma unroll
        for (int col = 0; col < kTensorCoreTileN; ++col) {
          float weight = 0.0f;
          if (col < valid_k) {
            const int64_t kv_idx = s_tile_start + col;
            bool allowed = true;
            if constexpr (IsCausal) {
              if (tgt_len > 1 && kv_idx > q_idx) {
                allowed = false;
              }
            }
            if constexpr (HasBoolMask) {
              if (bool_mask[mask_row_base + kv_idx]) {
                allowed = false;
              }
            }
            if (allowed) {
              float score =
                  static_cast<float>(score_tile[lane * kTensorCoreTileN + col]) *
                  q_row_scale[lane] *
                  k_row_scale[current_stage][col] *
                  scale_value;
              if constexpr (HasAdditiveMask) {
                score += additive_mask[mask_row_base + kv_idx];
              }
              if (::isfinite(score) && ::isfinite(new_row_max)) {
                weight = expf(score - new_row_max);
              }
            }
          }
          weight_tile[lane * kTensorCoreTileN + col] = weight;
          tile_sum += weight;
        }

        row_max[lane] = new_row_max;
        row_sum[lane] = prev_row_sum * prev_scale + tile_sum;
        row_rescale[lane] = prev_scale;
      }
      __syncthreads();

      for (int idx = lane; idx < valid_q * HeadDim; idx += kTensorCoreThreads) {
        const int row = idx / HeadDim;
        const int d = idx % HeadDim;
        float accum = out_accum[idx] * row_rescale[row];
        #pragma unroll
        for (int col = 0; col < kTensorCoreTileN; ++col) {
          if (col >= valid_k) {
            break;
          }
          const float weight = weight_tile[row * kTensorCoreTileN + col];
          if (weight != 0.0f) {
            accum += weight *
                (static_cast<float>(v_tile[current_stage][col * HeadDim + d]) * v_row_scale[current_stage][col]);
          }
        }
        out_accum[idx] = accum;
      }
      if constexpr (!UseBulkAsync) {
        if (next_s_tile_start < src_len) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
          __pipeline_wait_prior(0);
#endif
        }
      }
      __syncthreads();
    }

    for (int idx = lane; idx < valid_q * HeadDim; idx += kTensorCoreThreads) {
      const int row = idx / HeadDim;
      const int d = idx % HeadDim;
      const float denom = fmaxf(row_sum[row], 1e-8f);
      const float value = out_accum[idx] / denom;
      const int64_t q_idx = q_tile_start + row;
      const int64_t out_base = (((batch_idx * q_heads) + head_idx) * tgt_len + q_idx) * HeadDim;
      out[out_base + d] = static_cast<scalar_t>(value);
    }
    __syncthreads();
  }
}

template <typename scalar_t>
inline void launch_int8_attention_generic(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    bool has_bool_mask,
    bool has_additive_mask,
    bool is_causal,
    float scale_value,
    cudaStream_t stream) {
  const int64_t total = out.numel();
  const dim3 threads(kGenericThreads);
  const dim3 blocks(static_cast<unsigned int>((total + kGenericThreads - 1) / kGenericThreads));
  int8_attention_forward_generic_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      q_contig.data_ptr<int8_t>(),
      q_scale_contig.data_ptr<float>(),
      k_contig.data_ptr<int8_t>(),
      k_scale_contig.data_ptr<float>(),
      v_contig.data_ptr<int8_t>(),
      v_scale_contig.data_ptr<float>(),
      bool_mask_ptr,
      additive_mask_ptr,
      out.data_ptr<scalar_t>(),
      q_contig.size(0),
      q_contig.size(1),
      k_contig.size(1),
      q_contig.size(2),
      k_contig.size(2),
      q_contig.size(3),
      has_bool_mask,
      has_additive_mask,
      is_causal,
      scale_value);
}

template <typename scalar_t, int HeadDim, int Threads>
inline void launch_int8_attention_decode_nomask(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const torch::Tensor& out,
    cudaStream_t stream,
    float scale_value) {
  const dim3 blocks(static_cast<unsigned int>(q_contig.size(0) * q_contig.size(1)));
  const dim3 threads(Threads);
  int8_attention_decode_nomask_kernel<scalar_t, HeadDim, kDecodeTileN, Threads><<<blocks, threads, 0, stream>>>(
      q_contig.data_ptr<int8_t>(),
      q_scale_contig.data_ptr<float>(),
      k_contig.data_ptr<int8_t>(),
      k_scale_contig.data_ptr<float>(),
      v_contig.data_ptr<int8_t>(),
      v_scale_contig.data_ptr<float>(),
      out.data_ptr<scalar_t>(),
      q_contig.size(0),
      q_contig.size(1),
      k_contig.size(1),
      k_contig.size(2),
      scale_value);
}

template <typename scalar_t, int HeadDim>
inline void launch_int8_attention_decode_nomask_for_head_dim(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const torch::Tensor& out,
    cudaStream_t stream,
    float scale_value) {
  if constexpr (HeadDim <= 128) {
    launch_int8_attention_decode_nomask<scalar_t, HeadDim, 128>(
        q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, out, stream, scale_value);
  } else {
    launch_int8_attention_decode_nomask<scalar_t, HeadDim, kDecodeMaxThreads>(
        q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, out, stream, scale_value);
  }
}

template <typename scalar_t>
inline bool launch_int8_attention_decode_if_supported(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const torch::Tensor& out,
    cudaStream_t stream,
    float scale_value) {
  switch (q_contig.size(3)) {
    case 32:
      launch_int8_attention_decode_nomask_for_head_dim<scalar_t, 32>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, out, stream, scale_value);
      return true;
    case 64:
      launch_int8_attention_decode_nomask_for_head_dim<scalar_t, 64>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, out, stream, scale_value);
      return true;
    case 96:
      launch_int8_attention_decode_nomask_for_head_dim<scalar_t, 96>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, out, stream, scale_value);
      return true;
    case 128:
      launch_int8_attention_decode_nomask_for_head_dim<scalar_t, 128>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, out, stream, scale_value);
      return true;
    case 192:
      launch_int8_attention_decode_nomask_for_head_dim<scalar_t, 192>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, out, stream, scale_value);
      return true;
    case 256:
      launch_int8_attention_decode_nomask_for_head_dim<scalar_t, 256>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, out, stream, scale_value);
      return true;
    default:
      return false;
  }
}

template <typename scalar_t, int HeadDim, bool HasBoolMask, bool HasAdditiveMask, bool IsCausal, bool UseBulkAsync>
inline void launch_int8_attention_sm90a_wgmma_specialized(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    float scale_value,
    cudaStream_t stream) {
  const int64_t tiles_per_head = DivUpInt64(q_contig.size(2), kWgmmaAttentionActiveTileM);
  const int64_t total_blocks = q_contig.size(0) * q_contig.size(1) * tiles_per_head;
  const int64_t launch_blocks = SupportsInt8AttentionPersistentPath(q_contig, k_contig)
      ? PersistentAttentionBlockCount(q_contig, total_blocks)
      : total_blocks;
  const dim3 blocks(static_cast<unsigned int>(launch_blocks));
  const dim3 threads(kWgmmaAttentionThreads);
  int8_attention_forward_sm90a_wgmma_kernel<scalar_t, HeadDim, HasBoolMask, HasAdditiveMask, IsCausal, UseBulkAsync>
      <<<blocks, threads, 0, stream>>>(
          q_contig.data_ptr<int8_t>(),
          q_scale_contig.data_ptr<float>(),
          k_contig.data_ptr<int8_t>(),
          k_scale_contig.data_ptr<float>(),
          v_contig.data_ptr<int8_t>(),
          v_scale_contig.data_ptr<float>(),
          bool_mask_ptr,
          additive_mask_ptr,
          out.data_ptr<scalar_t>(),
          q_contig.size(0),
          q_contig.size(1),
          k_contig.size(1),
          q_contig.size(2),
          k_contig.size(2),
          scale_value);
}

template <typename scalar_t, int HeadDim>
inline void launch_int8_attention_sm90a_wgmma(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    bool has_bool_mask,
    bool has_additive_mask,
    bool is_causal,
    bool use_bulk_async,
    float scale_value,
    cudaStream_t stream) {
  if (has_bool_mask) {
    if (is_causal) {
      if (use_bulk_async) {
        launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, true, false, true, true>(
            q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
            additive_mask_ptr, out, scale_value, stream);
      } else {
        launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, true, false, true, false>(
            q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
            additive_mask_ptr, out, scale_value, stream);
      }
      return;
    }
    if (use_bulk_async) {
      launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, true, false, false, true>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, scale_value, stream);
    } else {
      launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, true, false, false, false>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, scale_value, stream);
    }
    return;
  }
  if (has_additive_mask) {
    if (is_causal) {
      if (use_bulk_async) {
        launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, false, true, true, true>(
            q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
            additive_mask_ptr, out, scale_value, stream);
      } else {
        launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, false, true, true, false>(
            q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
            additive_mask_ptr, out, scale_value, stream);
      }
      return;
    }
    if (use_bulk_async) {
      launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, false, true, false, true>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, scale_value, stream);
    } else {
      launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, false, true, false, false>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, scale_value, stream);
    }
    return;
  }
  if (is_causal) {
    if (use_bulk_async) {
      launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, false, false, true, true>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, scale_value, stream);
    } else {
      launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, false, false, true, false>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, scale_value, stream);
    }
    return;
  }
  if (use_bulk_async) {
    launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, false, false, false, true>(
        q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
        additive_mask_ptr, out, scale_value, stream);
  } else {
    launch_int8_attention_sm90a_wgmma_specialized<scalar_t, HeadDim, false, false, false, false>(
        q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
        additive_mask_ptr, out, scale_value, stream);
  }
}

template <typename scalar_t>
inline bool launch_int8_attention_sm90a_wgmma_if_supported(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    bool has_bool_mask,
    bool has_additive_mask,
    bool is_causal,
    bool use_bulk_async,
    float scale_value,
    cudaStream_t stream) {
  switch (q_contig.size(3)) {
    case 32:
      launch_int8_attention_sm90a_wgmma<scalar_t, 32>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    case 64:
      launch_int8_attention_sm90a_wgmma<scalar_t, 64>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    case 96:
      launch_int8_attention_sm90a_wgmma<scalar_t, 96>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    case 128:
      launch_int8_attention_sm90a_wgmma<scalar_t, 128>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    default:
      return false;
  }
}

template <typename scalar_t, int HeadDim, bool HasBoolMask, bool HasAdditiveMask, bool IsCausal>
inline void launch_int8_attention_tensorcore_specialized(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    float scale_value,
    cudaStream_t stream) {
  const int64_t tiles_per_head = DivUpInt64(q_contig.size(2), kTensorCoreTileM);
  const int64_t total_blocks = q_contig.size(0) * q_contig.size(1) * tiles_per_head;
  const int64_t launch_blocks = SupportsInt8AttentionPersistentPath(q_contig, k_contig)
      ? PersistentAttentionBlockCount(q_contig, total_blocks)
      : total_blocks;
  const dim3 blocks(static_cast<unsigned int>(launch_blocks));
  const dim3 threads(kTensorCoreThreads);
  int8_attention_forward_tensorcore_kernel<scalar_t, HeadDim, HasBoolMask, HasAdditiveMask, IsCausal>
      <<<blocks, threads, 0, stream>>>(
      q_contig.data_ptr<int8_t>(),
      q_scale_contig.data_ptr<float>(),
      k_contig.data_ptr<int8_t>(),
      k_scale_contig.data_ptr<float>(),
      v_contig.data_ptr<int8_t>(),
      v_scale_contig.data_ptr<float>(),
      bool_mask_ptr,
      additive_mask_ptr,
      out.data_ptr<scalar_t>(),
      q_contig.size(0),
      q_contig.size(1),
      k_contig.size(1),
      q_contig.size(2),
      k_contig.size(2),
      scale_value);
}

template <typename scalar_t, int HeadDim>
inline void launch_int8_attention_tensorcore(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    bool has_bool_mask,
    bool has_additive_mask,
    bool is_causal,
    float scale_value,
    cudaStream_t stream) {
  if (has_bool_mask) {
    if (is_causal) {
      launch_int8_attention_tensorcore_specialized<scalar_t, HeadDim, true, false, true>(
          q_contig,
          q_scale_contig,
          k_contig,
          k_scale_contig,
          v_contig,
          v_scale_contig,
          bool_mask_ptr,
          additive_mask_ptr,
          out,
          scale_value,
          stream);
      return;
    }
    launch_int8_attention_tensorcore_specialized<scalar_t, HeadDim, true, false, false>(
        q_contig,
        q_scale_contig,
        k_contig,
        k_scale_contig,
        v_contig,
        v_scale_contig,
        bool_mask_ptr,
        additive_mask_ptr,
        out,
        scale_value,
        stream);
    return;
  }
  if (has_additive_mask) {
    if (is_causal) {
      launch_int8_attention_tensorcore_specialized<scalar_t, HeadDim, false, true, true>(
          q_contig,
          q_scale_contig,
          k_contig,
          k_scale_contig,
          v_contig,
          v_scale_contig,
          bool_mask_ptr,
          additive_mask_ptr,
          out,
          scale_value,
          stream);
      return;
    }
    launch_int8_attention_tensorcore_specialized<scalar_t, HeadDim, false, true, false>(
        q_contig,
        q_scale_contig,
        k_contig,
        k_scale_contig,
        v_contig,
        v_scale_contig,
        bool_mask_ptr,
        additive_mask_ptr,
        out,
        scale_value,
        stream);
    return;
  }
  if (is_causal) {
    launch_int8_attention_tensorcore_specialized<scalar_t, HeadDim, false, false, true>(
        q_contig,
        q_scale_contig,
        k_contig,
        k_scale_contig,
        v_contig,
        v_scale_contig,
        bool_mask_ptr,
        additive_mask_ptr,
        out,
        scale_value,
        stream);
    return;
  }
  launch_int8_attention_tensorcore_specialized<scalar_t, HeadDim, false, false, false>(
      q_contig,
      q_scale_contig,
      k_contig,
      k_scale_contig,
      v_contig,
      v_scale_contig,
      bool_mask_ptr,
      additive_mask_ptr,
      out,
      scale_value,
      stream);
}

template <typename scalar_t, int HeadDim, bool HasBoolMask, bool HasAdditiveMask, bool IsCausal, bool UseBulkAsync>
inline void launch_int8_attention_sm90_pipeline_specialized(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    bool has_bool_mask,
    bool has_additive_mask,
    bool is_causal,
    float scale_value,
    cudaStream_t stream) {
  const int64_t tiles_per_head = DivUpInt64(q_contig.size(2), kTensorCoreTileM);
  const int64_t total_blocks = q_contig.size(0) * q_contig.size(1) * tiles_per_head;
  const int64_t launch_blocks = SupportsInt8AttentionPersistentPath(q_contig, k_contig)
      ? PersistentAttentionBlockCount(q_contig, total_blocks)
      : total_blocks;
  const dim3 blocks(static_cast<unsigned int>(launch_blocks));
  const dim3 threads(kTensorCoreThreads);
  int8_attention_forward_sm90_pipeline_kernel<scalar_t, HeadDim, HasBoolMask, HasAdditiveMask, IsCausal, UseBulkAsync>
      <<<blocks, threads, 0, stream>>>(
          q_contig.data_ptr<int8_t>(),
          q_scale_contig.data_ptr<float>(),
          k_contig.data_ptr<int8_t>(),
          k_scale_contig.data_ptr<float>(),
          v_contig.data_ptr<int8_t>(),
          v_scale_contig.data_ptr<float>(),
          bool_mask_ptr,
          additive_mask_ptr,
          out.data_ptr<scalar_t>(),
          q_contig.size(0),
          q_contig.size(1),
          k_contig.size(1),
          q_contig.size(2),
          k_contig.size(2),
          scale_value);
}

template <typename scalar_t, int HeadDim>
inline void launch_int8_attention_sm90_pipeline(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    bool has_bool_mask,
    bool has_additive_mask,
    bool is_causal,
    bool use_bulk_async,
    float scale_value,
    cudaStream_t stream) {
  if (has_bool_mask) {
    if (is_causal) {
      if (use_bulk_async) {
        launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, true, false, true, true>(
            q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
            additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
      } else {
        launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, true, false, true, false>(
            q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
            additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
      }
      return;
    }
    if (use_bulk_async) {
      launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, true, false, false, true>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
    } else {
      launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, true, false, false, false>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
    }
    return;
  }
  if (has_additive_mask) {
    if (is_causal) {
      if (use_bulk_async) {
        launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, false, true, true, true>(
            q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
            additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
      } else {
        launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, false, true, true, false>(
            q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
            additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
      }
      return;
    }
    if (use_bulk_async) {
      launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, false, true, false, true>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
    } else {
      launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, false, true, false, false>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
    }
    return;
  }
  if (is_causal) {
    if (use_bulk_async) {
      launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, false, false, true, true>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
    } else {
      launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, false, false, true, false>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
    }
    return;
  }
  if (use_bulk_async) {
    launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, false, false, false, true>(
        q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
        additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
  } else {
    launch_int8_attention_sm90_pipeline_specialized<scalar_t, HeadDim, false, false, false, false>(
        q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
        additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, scale_value, stream);
  }
}

template <typename scalar_t>
inline bool launch_int8_attention_sm90_pipeline_if_supported(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    bool has_bool_mask,
    bool has_additive_mask,
    bool is_causal,
    bool use_bulk_async,
    float scale_value,
    cudaStream_t stream) {
  switch (q_contig.size(3)) {
    case 32:
      launch_int8_attention_sm90_pipeline<scalar_t, 32>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    case 64:
      launch_int8_attention_sm90_pipeline<scalar_t, 64>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    case 96:
      launch_int8_attention_sm90_pipeline<scalar_t, 96>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    case 128:
      launch_int8_attention_sm90_pipeline<scalar_t, 128>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    case 192:
      launch_int8_attention_sm90_pipeline<scalar_t, 192>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    case 256:
      launch_int8_attention_sm90_pipeline<scalar_t, 256>(
          q_contig, q_scale_contig, k_contig, k_scale_contig, v_contig, v_scale_contig, bool_mask_ptr,
          additive_mask_ptr, out, has_bool_mask, has_additive_mask, is_causal, use_bulk_async, scale_value, stream);
      return true;
    default:
      return false;
  }
}

template <typename scalar_t>
inline bool launch_int8_attention_tensorcore_if_supported(
    const torch::Tensor& q_contig,
    const torch::Tensor& q_scale_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& k_scale_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& v_scale_contig,
    const bool* bool_mask_ptr,
    const float* additive_mask_ptr,
    const torch::Tensor& out,
    bool has_bool_mask,
    bool has_additive_mask,
    bool is_causal,
    float scale_value,
    cudaStream_t stream) {
  switch (q_contig.size(3)) {
    case 32:
      launch_int8_attention_tensorcore<scalar_t, 32>(
          q_contig,
          q_scale_contig,
          k_contig,
          k_scale_contig,
          v_contig,
          v_scale_contig,
          bool_mask_ptr,
          additive_mask_ptr,
          out,
          has_bool_mask,
          has_additive_mask,
          is_causal,
          scale_value,
          stream);
      return true;
    case 64:
      launch_int8_attention_tensorcore<scalar_t, 64>(
          q_contig,
          q_scale_contig,
          k_contig,
          k_scale_contig,
          v_contig,
          v_scale_contig,
          bool_mask_ptr,
          additive_mask_ptr,
          out,
          has_bool_mask,
          has_additive_mask,
          is_causal,
          scale_value,
          stream);
      return true;
    case 96:
      launch_int8_attention_tensorcore<scalar_t, 96>(
          q_contig,
          q_scale_contig,
          k_contig,
          k_scale_contig,
          v_contig,
          v_scale_contig,
          bool_mask_ptr,
          additive_mask_ptr,
          out,
          has_bool_mask,
          has_additive_mask,
          is_causal,
          scale_value,
          stream);
      return true;
    case 128:
      launch_int8_attention_tensorcore<scalar_t, 128>(
          q_contig,
          q_scale_contig,
          k_contig,
          k_scale_contig,
          v_contig,
          v_scale_contig,
          bool_mask_ptr,
          additive_mask_ptr,
          out,
          has_bool_mask,
          has_additive_mask,
          is_causal,
          scale_value,
          stream);
      return true;
    case 192:
      launch_int8_attention_tensorcore<scalar_t, 192>(
          q_contig,
          q_scale_contig,
          k_contig,
          k_scale_contig,
          v_contig,
          v_scale_contig,
          bool_mask_ptr,
          additive_mask_ptr,
          out,
          has_bool_mask,
          has_additive_mask,
          is_causal,
          scale_value,
          stream);
      return true;
    case 256:
      launch_int8_attention_tensorcore<scalar_t, 256>(
          q_contig,
          q_scale_contig,
          k_contig,
          k_scale_contig,
          v_contig,
          v_scale_contig,
          bool_mask_ptr,
          additive_mask_ptr,
          out,
          has_bool_mask,
          has_additive_mask,
          is_causal,
          scale_value,
          stream);
      return true;
    default:
      return false;
  }
}

}  // namespace

bool HasCudaInt8AttentionKernel() {
  return true;
}

torch::Tensor CudaInt8AttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& q_scale,
    const torch::Tensor& k,
    const torch::Tensor& k_scale,
    const torch::Tensor& v,
    const torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(q.is_cuda() && q_scale.is_cuda() && k.is_cuda() && k_scale.is_cuda() && v.is_cuda() && v_scale.is_cuda(),
              "CudaInt8AttentionForward: q, q_scale, k, k_scale, v, and v_scale must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && k.scalar_type() == torch::kInt8 && v.scalar_type() == torch::kInt8,
              "CudaInt8AttentionForward: q, k, and v must use int8 storage");
  TORCH_CHECK(q_scale.scalar_type() == torch::kFloat32 && k_scale.scalar_type() == torch::kFloat32 &&
                  v_scale.scalar_type() == torch::kFloat32,
              "CudaInt8AttentionForward: q_scale, k_scale, and v_scale must use float32 storage");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "CudaInt8AttentionForward: q, k, and v must be rank-4");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "CudaInt8AttentionForward: batch size mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "CudaInt8AttentionForward: kv head mismatch");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "CudaInt8AttentionForward: q heads must be a multiple of kv heads");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "CudaInt8AttentionForward: head_dim mismatch");
  TORCH_CHECK(q_scale.numel() == q.size(0) * q.size(1) * q.size(2), "CudaInt8AttentionForward: q_scale size mismatch");
  TORCH_CHECK(k_scale.numel() == k.size(0) * k.size(1) * k.size(2), "CudaInt8AttentionForward: k_scale size mismatch");
  TORCH_CHECK(v_scale.numel() == v.size(0) * v.size(1) * v.size(2), "CudaInt8AttentionForward: v_scale size mismatch");

  const auto output_dtype = out_dtype.has_value() ? out_dtype.value() : torch::kFloat32;
  TORCH_CHECK(IsSupportedInt8AttentionOutDtype(output_dtype), "CudaInt8AttentionForward: unsupported output dtype");

  c10::cuda::CUDAGuard device_guard{q.device()};

  auto q_contig = q.contiguous();
  auto q_scale_contig = q_scale.reshape({-1}).contiguous();
  auto k_contig = k.contiguous();
  auto k_scale_contig = k_scale.reshape({-1}).contiguous();
  auto v_contig = v.contiguous();
  auto v_scale_contig = v_scale.reshape({-1}).contiguous();
  torch::Tensor bool_mask_contig;
  torch::Tensor additive_mask_contig;
  const bool* bool_mask_ptr = nullptr;
  const float* additive_mask_ptr = nullptr;
  bool has_bool_mask = false;
  bool has_additive_mask = false;
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    auto mask = attn_mask.value();
    TORCH_CHECK(mask.is_cuda(), "CudaInt8AttentionForward: attn_mask must be a CUDA tensor");
    TORCH_CHECK(mask.dim() == 4, "CudaInt8AttentionForward: attn_mask must be rank-4");
    TORCH_CHECK(mask.size(0) == q_contig.size(0), "CudaInt8AttentionForward: attn_mask batch size mismatch");
    TORCH_CHECK(mask.size(1) == 1 || mask.size(1) == q_contig.size(1),
                "CudaInt8AttentionForward: attn_mask head dimension must be 1 or q_heads");
    TORCH_CHECK(mask.size(2) == q_contig.size(2), "CudaInt8AttentionForward: attn_mask tgt_len mismatch");
    TORCH_CHECK(mask.size(3) == k_contig.size(2), "CudaInt8AttentionForward: attn_mask src_len mismatch");
    if (mask.size(1) == 1 && q_contig.size(1) != 1) {
      mask = mask.expand({q_contig.size(0), q_contig.size(1), q_contig.size(2), k_contig.size(2)});
    }
    if (mask.scalar_type() == torch::kBool) {
      bool_mask_contig = mask.contiguous();
      bool_mask_ptr = bool_mask_contig.data_ptr<bool>();
      has_bool_mask = true;
    } else {
      additive_mask_contig = mask.to(torch::kFloat32).contiguous();
      additive_mask_ptr = additive_mask_contig.data_ptr<float>();
      has_additive_mask = true;
    }
  }

  auto out = torch::empty(
      {q_contig.size(0), q_contig.size(1), q_contig.size(2), q_contig.size(3)},
      q_contig.options().dtype(output_dtype));
  if (out.numel() == 0) {
    return out;
  }

  const float scale_value = static_cast<float>(
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(static_cast<double>(q_contig.size(3)))));
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
  const bool prefer_optimized = ShouldPreferInt8AttentionOptimizedPath(q_contig, k_contig);
  const bool prefer_decode_specialized =
      SupportsInt8AttentionDecodeSpecializedPath(q_contig, has_bool_mask, has_additive_mask);
  const bool prefer_wgmma =
      !prefer_decode_specialized && prefer_optimized && SupportsInt8AttentionWgmmaPath(q_contig, k_contig);
  const bool prefer_tensorcore = prefer_optimized && SupportsInt8AttentionTensorCorePath(q_contig);
  const bool prefer_sm90_pipeline =
      !prefer_decode_specialized && !prefer_wgmma && prefer_tensorcore &&
      SupportsInt8AttentionSm90PipelinePath(q_contig, k_contig);
  const bool prefer_sm90_bulk_async =
      (prefer_wgmma || prefer_sm90_pipeline) && SupportsInt8AttentionSm90BulkAsyncPath(q_contig, k_contig);

  switch (output_dtype) {
    case torch::kFloat32: {
      const bool launched_decode =
          prefer_decode_specialized &&
          launch_int8_attention_decode_if_supported<float>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              out,
              stream.stream(),
              scale_value);
      const bool launched_wgmma =
          !launched_decode &&
          prefer_wgmma &&
          launch_int8_attention_sm90a_wgmma_if_supported<float>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              bool_mask_ptr,
              additive_mask_ptr,
              out,
              has_bool_mask,
              has_additive_mask,
              is_causal,
              prefer_sm90_bulk_async,
              scale_value,
              stream.stream());
      const bool launched_sm90_pipeline =
          !launched_decode &&
          !launched_wgmma &&
          prefer_sm90_pipeline &&
          launch_int8_attention_sm90_pipeline_if_supported<float>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              bool_mask_ptr,
              additive_mask_ptr,
              out,
              has_bool_mask,
              has_additive_mask,
              is_causal,
              prefer_sm90_bulk_async,
              scale_value,
              stream.stream());
      const bool launched_tensorcore =
          !launched_decode &&
          !launched_wgmma &&
          !launched_sm90_pipeline &&
          prefer_tensorcore &&
          launch_int8_attention_tensorcore_if_supported<float>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              bool_mask_ptr,
              additive_mask_ptr,
              out,
              has_bool_mask,
              has_additive_mask,
              is_causal,
              scale_value,
              stream.stream());
      if (!launched_decode && !launched_wgmma && !launched_sm90_pipeline && !launched_tensorcore) {
        launch_int8_attention_generic<float>(
            q_contig,
            q_scale_contig,
            k_contig,
            k_scale_contig,
            v_contig,
            v_scale_contig,
            bool_mask_ptr,
            additive_mask_ptr,
            out,
            has_bool_mask,
            has_additive_mask,
            is_causal,
            scale_value,
            stream.stream());
      }
      break;
    }
    case torch::kFloat16: {
      const bool launched_decode =
          prefer_decode_specialized &&
          launch_int8_attention_decode_if_supported<at::Half>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              out,
              stream.stream(),
              scale_value);
      const bool launched_wgmma =
          !launched_decode &&
          prefer_wgmma &&
          launch_int8_attention_sm90a_wgmma_if_supported<at::Half>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              bool_mask_ptr,
              additive_mask_ptr,
              out,
              has_bool_mask,
              has_additive_mask,
              is_causal,
              prefer_sm90_bulk_async,
              scale_value,
              stream.stream());
      const bool launched_sm90_pipeline =
          !launched_decode &&
          !launched_wgmma &&
          prefer_sm90_pipeline &&
          launch_int8_attention_sm90_pipeline_if_supported<at::Half>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              bool_mask_ptr,
              additive_mask_ptr,
              out,
              has_bool_mask,
              has_additive_mask,
              is_causal,
              prefer_sm90_bulk_async,
              scale_value,
              stream.stream());
      const bool launched_tensorcore =
          !launched_decode &&
          !launched_wgmma &&
          !launched_sm90_pipeline &&
          prefer_tensorcore &&
          launch_int8_attention_tensorcore_if_supported<at::Half>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              bool_mask_ptr,
              additive_mask_ptr,
              out,
              has_bool_mask,
              has_additive_mask,
              is_causal,
              scale_value,
              stream.stream());
      if (!launched_decode && !launched_wgmma && !launched_sm90_pipeline && !launched_tensorcore) {
        launch_int8_attention_generic<at::Half>(
            q_contig,
            q_scale_contig,
            k_contig,
            k_scale_contig,
            v_contig,
            v_scale_contig,
            bool_mask_ptr,
            additive_mask_ptr,
            out,
            has_bool_mask,
            has_additive_mask,
            is_causal,
            scale_value,
            stream.stream());
      }
      break;
    }
    case torch::kBFloat16: {
      const bool launched_decode =
          prefer_decode_specialized &&
          launch_int8_attention_decode_if_supported<at::BFloat16>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              out,
              stream.stream(),
              scale_value);
      const bool launched_wgmma =
          !launched_decode &&
          prefer_wgmma &&
          launch_int8_attention_sm90a_wgmma_if_supported<at::BFloat16>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              bool_mask_ptr,
              additive_mask_ptr,
              out,
              has_bool_mask,
              has_additive_mask,
              is_causal,
              prefer_sm90_bulk_async,
              scale_value,
              stream.stream());
      const bool launched_sm90_pipeline =
          !launched_decode &&
          !launched_wgmma &&
          prefer_sm90_pipeline &&
          launch_int8_attention_sm90_pipeline_if_supported<at::BFloat16>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              bool_mask_ptr,
              additive_mask_ptr,
              out,
              has_bool_mask,
              has_additive_mask,
              is_causal,
              prefer_sm90_bulk_async,
              scale_value,
              stream.stream());
      const bool launched_tensorcore =
          !launched_decode &&
          !launched_wgmma &&
          !launched_sm90_pipeline &&
          prefer_tensorcore &&
          launch_int8_attention_tensorcore_if_supported<at::BFloat16>(
              q_contig,
              q_scale_contig,
              k_contig,
              k_scale_contig,
              v_contig,
              v_scale_contig,
              bool_mask_ptr,
              additive_mask_ptr,
              out,
              has_bool_mask,
              has_additive_mask,
              is_causal,
              scale_value,
              stream.stream());
      if (!launched_decode && !launched_wgmma && !launched_sm90_pipeline && !launched_tensorcore) {
        launch_int8_attention_generic<at::BFloat16>(
            q_contig,
            q_scale_contig,
            k_contig,
            k_scale_contig,
            v_contig,
            v_scale_contig,
            bool_mask_ptr,
            additive_mask_ptr,
            out,
            has_bool_mask,
            has_additive_mask,
            is_causal,
            scale_value,
            stream.stream());
      }
      break;
    }
    default:
      TORCH_CHECK(false, "CudaInt8AttentionForward: unsupported output dtype");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
