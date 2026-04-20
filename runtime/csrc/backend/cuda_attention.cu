#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "attention/cuda_attention_common.cuh"
#include "attention/cuda_attention_dispatch.cuh"
#include "../descriptors/attention_desc.h"
#include "../policy/attention_policy.h"
#include "../reference/aten_reference.h"

#include <cmath>

namespace {

bool UseCudaAttentionKernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask) {
  if (!q.is_cuda() || !k.is_cuda() || !v.is_cuda()) {
    return false;
  }
  if (!(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16)) {
    return false;
  }
  if (q.scalar_type() != k.scalar_type() || q.scalar_type() != v.scalar_type()) {
    return false;
  }
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    const auto& mask = attn_mask.value();
    if (!mask.is_cuda() || mask.dim() != 4) {
      return false;
    }
    if (!(mask.scalar_type() == torch::kBool ||
          mask.scalar_type() == q.scalar_type() ||
          mask.scalar_type() == torch::kFloat32)) {
      return false;
    }
    if (mask.size(0) != q.size(0) || mask.size(1) != q.size(1) || mask.size(2) != q.size(2) || mask.size(3) != k.size(2)) {
      return false;
    }
  }
  if (k.size(1) != v.size(1)) {
    return false;
  }
  if (q.size(1) % k.size(1) != 0) {
    return false;
  }
  return true;
}

bool TryBuildAttentionDesc(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    t10::desc::AttentionDesc* out_desc) {
  if (!UseCudaAttentionKernel(q, k, v, attn_mask)) {
    return false;
  }
  *out_desc = t10::desc::AttentionDesc{
      q.scalar_type(),
      q.size(0),
      q.size(1),
      k.size(1),
      q.size(2),
      k.size(2),
      q.size(3),
      t10::desc::ResolveAttentionMaskKind(attn_mask, q.scalar_type()),
      t10::desc::ResolveAttentionPhase(q.size(2)),
      t10::desc::ResolveAttentionHeadMode(q.size(1), k.size(1)),
      t10::desc::AttentionLayoutKind::kBHSD,
      t10::desc::AttentionLayoutKind::kBHSD,
      is_causal};
  return true;
}

void LaunchAttentionKernel(
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
  switch (desc.phase) {
    case t10::desc::AttentionPhase::kDecode:
      t10::cuda::attention::LaunchPlannedAttentionDecodeDispatcher(
          q_contig,
          k_contig,
          v_contig,
          mask_ptr,
          mask_kind,
          out,
          desc,
          plan,
          scale_value,
          stream);
      return;
    default:
      t10::cuda::attention::LaunchPlannedAttentionPrefillDispatcher(
          q_contig,
          k_contig,
          v_contig,
          mask_ptr,
          mask_kind,
          out,
          desc,
          plan,
          scale_value,
          stream);
      return;
  }
}

}  // namespace

bool HasCudaAttentionKernel() {
  return true;
}

torch::Tensor CudaAttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale) {
  t10::desc::AttentionDesc desc;
  if (!TryBuildAttentionDesc(q, k, v, attn_mask, is_causal, &desc)) {
    return ReferenceAttentionForward(q, k, v, attn_mask, is_causal, scale);
  }
  const auto plan = t10::policy::ResolveAttentionPlan(desc);

  c10::cuda::CUDAGuard device_guard{q.device()};

  auto q_contig = q.contiguous();
  auto k_contig = k.contiguous();
  auto v_contig = v.contiguous();
  torch::Tensor mask_contig;
  int mask_kind = 0;
  const void* mask_ptr = nullptr;
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    mask_contig = attn_mask.value().contiguous();
    if (mask_contig.scalar_type() == torch::kBool) {
      mask_kind = 1;
    } else if (mask_contig.scalar_type() == q_contig.scalar_type()) {
      mask_kind = 2;
    } else {
      mask_kind = 3;
    }
    mask_ptr = mask_contig.data_ptr();
  }

  auto out = torch::empty_like(q_contig);
  const float scale_value = static_cast<float>(
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(static_cast<double>(desc.head_dim))));
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());

  LaunchAttentionKernel(
      q_contig,
      k_contig,
      v_contig,
      mask_ptr,
      mask_kind,
      out,
      desc,
      plan,
      scale_value,
      stream.stream());

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

namespace {

bool UseCudaPagedAttentionDecodeKernel(
    const torch::Tensor& q,
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    const c10::optional<torch::Tensor>& attn_mask) {
  if (!q.is_cuda() || !k_pages.is_cuda() || !v_pages.is_cuda() || !block_table.is_cuda() || !lengths.is_cuda()) {
    return false;
  }
  if (q.dim() != 4 || q.size(2) != 1 || k_pages.dim() != 4 || v_pages.dim() != 4 || block_table.dim() != 2 || lengths.dim() != 1) {
    return false;
  }
  if (!(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16)) {
    return false;
  }
  if (q.scalar_type() != k_pages.scalar_type() || q.scalar_type() != v_pages.scalar_type()) {
    return false;
  }
  if (q.size(0) != block_table.size(0) || q.size(0) != lengths.size(0)) {
    return false;
  }
  if (k_pages.size(1) != v_pages.size(1) || q.size(1) % k_pages.size(1) != 0 || q.size(3) != k_pages.size(3)) {
    return false;
  }
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    const auto& mask = attn_mask.value();
    if (!mask.is_cuda() || mask.dim() != 4 || mask.size(0) != q.size(0) || mask.size(1) != q.size(1) || mask.size(2) != 1) {
      return false;
    }
    if (!(mask.scalar_type() == torch::kBool || mask.scalar_type() == q.scalar_type() || mask.scalar_type() == torch::kFloat32)) {
      return false;
    }
  }
  return true;
}

int SelectPagedDecodeThreads(int64_t seq_len, int64_t head_dim) {
  if (seq_len <= 64 && head_dim <= 64) {
    return t10::cuda::attention::kSmallRowThreads;
  }
  if (seq_len <= 128 && head_dim <= 128) {
    return t10::cuda::attention::kMediumRowThreads;
  }
  return t10::cuda::attention::kLargeRowThreads;
}

template <typename scalar_t, int Threads, bool HasMask>
__global__ __launch_bounds__(Threads, 2) void paged_decode_attention_q1_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pages,
    const scalar_t* __restrict__ v_pages,
    const int64_t* __restrict__ block_table,
    const int64_t* __restrict__ lengths,
    const void* __restrict__ mask,
    int mask_kind,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t max_blocks,
    int64_t page_size,
    int64_t mask_seq,
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
  const int64_t row_len = lengths[batch_idx];
  const int64_t q_base = ((batch_idx * q_heads) + head_idx) * dh;
  if (row_len <= 0) {
    for (int64_t d = threadIdx.x; d < dh; d += blockDim.x) {
      out[q_base + d] = static_cast<scalar_t>(0);
    }
    return;
  }

  const int64_t head_group = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_group;
  const int64_t mask_base = ((batch_idx * q_heads) + head_idx) * mask_seq;

  float local_max = -INFINITY;
  for (int64_t s = threadIdx.x; s < row_len; s += blockDim.x) {
    float score = 0.0f;
    const int64_t block_idx = s / page_size;
    const int64_t offset = s % page_size;
    const int64_t page_id = block_table[batch_idx * max_blocks + block_idx];
    const int64_t page_base = ((((page_id * kv_heads) + kv_head_idx) * page_size) + offset) * dh;
    for (int64_t d = 0; d < dh; ++d) {
      score += static_cast<float>(q[q_base + d]) * static_cast<float>(k_pages[page_base + d]);
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
  const float row_max = t10::cuda::attention::BlockReduceMax<Threads>(local_max, shared);
  if (row_max == -INFINITY) {
    for (int64_t d = threadIdx.x; d < dh; d += blockDim.x) {
      out[q_base + d] = static_cast<scalar_t>(0);
    }
    return;
  }

  float local_sum = 0.0f;
  for (int64_t s = threadIdx.x; s < row_len; s += blockDim.x) {
    float score = 0.0f;
    const int64_t block_idx = s / page_size;
    const int64_t offset = s % page_size;
    const int64_t page_id = block_table[batch_idx * max_blocks + block_idx];
    const int64_t page_base = ((((page_id * kv_heads) + kv_head_idx) * page_size) + offset) * dh;
    for (int64_t d = 0; d < dh; ++d) {
      score += static_cast<float>(q[q_base + d]) * static_cast<float>(k_pages[page_base + d]);
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
  const float denom = fmaxf(t10::cuda::attention::BlockReduceSum<Threads>(local_sum, shared), 1e-20f);

  for (int64_t d = threadIdx.x; d < dh; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t s = 0; s < row_len; ++s) {
      float score = 0.0f;
      const int64_t block_idx = s / page_size;
      const int64_t offset = s % page_size;
      const int64_t page_id = block_table[batch_idx * max_blocks + block_idx];
      const int64_t page_base = ((((page_id * kv_heads) + kv_head_idx) * page_size) + offset) * dh;
      for (int64_t kk = 0; kk < dh; ++kk) {
        score += static_cast<float>(q[q_base + kk]) * static_cast<float>(k_pages[page_base + kk]);
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
      acc += (expf(score - row_max) / denom) * static_cast<float>(v_pages[page_base + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, bool HasMask>
__global__ __launch_bounds__(t10::cuda::attention::kSmallRowThreads, 4) void paged_decode_attention_q1_smallseq_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pages,
    const scalar_t* __restrict__ v_pages,
    const int64_t* __restrict__ block_table,
    const int64_t* __restrict__ lengths,
    const void* __restrict__ mask,
    int mask_kind,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t max_blocks,
    int64_t page_size,
    int64_t mask_seq,
    int64_t dh,
    float scale) {
  constexpr int Threads = t10::cuda::attention::kSmallRowThreads;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t total_rows = batch * q_heads;
  if (row >= total_rows) {
    return;
  }

  __shared__ float reduce_shared[Threads];
  __shared__ float score_shared[Threads];
  __shared__ float q_shared[Threads];
  __shared__ int64_t page_base_shared[Threads];

  const int lane = static_cast<int>(threadIdx.x);
  const int64_t head_idx = row % q_heads;
  const int64_t batch_idx = row / q_heads;
  const int64_t row_len = lengths[batch_idx];
  const int64_t q_base = ((batch_idx * q_heads) + head_idx) * dh;
  if (row_len <= 0) {
    for (int64_t d = lane; d < dh; d += Threads) {
      out[q_base + d] = static_cast<scalar_t>(0);
    }
    return;
  }

  if (lane < dh) {
    q_shared[lane] = static_cast<float>(q[q_base + lane]);
  }
  __syncthreads();

  const int64_t head_group = q_heads / kv_heads;
  const int64_t kv_head_idx = head_idx / head_group;
  const int64_t mask_base = ((batch_idx * q_heads) + head_idx) * mask_seq;

  float local_score = -INFINITY;
  if (lane < row_len) {
    const int64_t s = lane;
    const int64_t block_idx = s / page_size;
    const int64_t offset = s % page_size;
    const int64_t page_id = block_table[batch_idx * max_blocks + block_idx];
    const int64_t page_base = ((((page_id * kv_heads) + kv_head_idx) * page_size) + offset) * dh;
    page_base_shared[lane] = page_base;
    float score = 0.0f;
    for (int64_t d = 0; d < dh; ++d) {
      score += q_shared[d] * static_cast<float>(k_pages[page_base + d]);
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
    local_score = score;
    score_shared[lane] = score;
  } else {
    score_shared[lane] = -INFINITY;
  }

  const float row_max = t10::cuda::attention::BlockReduceMax<Threads>(local_score, reduce_shared);
  if (row_max == -INFINITY) {
    for (int64_t d = lane; d < dh; d += Threads) {
      out[q_base + d] = static_cast<scalar_t>(0);
    }
    return;
  }

  float local_sum = 0.0f;
  if (lane < row_len) {
    const float weight = expf(score_shared[lane] - row_max);
    score_shared[lane] = weight;
    local_sum = weight;
  } else {
    score_shared[lane] = 0.0f;
  }
  const float denom = fmaxf(t10::cuda::attention::BlockReduceSum<Threads>(local_sum, reduce_shared), 1.0e-20f);
  if (lane < row_len) {
    score_shared[lane] /= denom;
  }
  __syncthreads();

  for (int64_t d = lane; d < dh; d += Threads) {
    float acc = 0.0f;
    for (int64_t s = 0; s < row_len; ++s) {
      acc += score_shared[s] * static_cast<float>(v_pages[page_base_shared[s] + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, bool HasMask>
void LaunchPagedDecodeAttentionQ1(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_pages_contig,
    const torch::Tensor& v_pages_contig,
    const torch::Tensor& block_table_contig,
    const torch::Tensor& lengths_contig,
    const void* mask_ptr,
    int mask_kind,
    const torch::Tensor& out,
    int threads,
    int64_t mask_seq,
    float scale_value,
    cudaStream_t stream) {
  const dim3 blocks(static_cast<unsigned int>(q_contig.size(0) * q_contig.size(1)));
  switch (threads) {
    case t10::cuda::attention::kSmallRowThreads:
      if (mask_seq <= t10::cuda::attention::kSmallRowThreads && q_contig.size(3) <= t10::cuda::attention::kSmallRowThreads) {
        paged_decode_attention_q1_smallseq_forward_kernel<scalar_t, HasMask>
            <<<blocks, t10::cuda::attention::kSmallRowThreads, 0, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_pages_contig.data_ptr<scalar_t>(),
                v_pages_contig.data_ptr<scalar_t>(),
                block_table_contig.data_ptr<int64_t>(),
                lengths_contig.data_ptr<int64_t>(),
                mask_ptr,
                mask_kind,
                out.data_ptr<scalar_t>(),
                q_contig.size(0),
                q_contig.size(1),
                k_pages_contig.size(1),
                block_table_contig.size(1),
                k_pages_contig.size(2),
                mask_seq,
                q_contig.size(3),
                scale_value);
      } else {
        paged_decode_attention_q1_forward_kernel<scalar_t, t10::cuda::attention::kSmallRowThreads, HasMask>
            <<<blocks, t10::cuda::attention::kSmallRowThreads, 0, stream>>>(
                q_contig.data_ptr<scalar_t>(),
                k_pages_contig.data_ptr<scalar_t>(),
                v_pages_contig.data_ptr<scalar_t>(),
                block_table_contig.data_ptr<int64_t>(),
                lengths_contig.data_ptr<int64_t>(),
                mask_ptr,
                mask_kind,
                out.data_ptr<scalar_t>(),
                q_contig.size(0),
                q_contig.size(1),
                k_pages_contig.size(1),
                block_table_contig.size(1),
                k_pages_contig.size(2),
                mask_seq,
                q_contig.size(3),
                scale_value);
      }
      return;
    case t10::cuda::attention::kMediumRowThreads:
      paged_decode_attention_q1_forward_kernel<scalar_t, t10::cuda::attention::kMediumRowThreads, HasMask>
          <<<blocks, t10::cuda::attention::kMediumRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_pages_contig.data_ptr<scalar_t>(),
              v_pages_contig.data_ptr<scalar_t>(),
              block_table_contig.data_ptr<int64_t>(),
              lengths_contig.data_ptr<int64_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              q_contig.size(0),
              q_contig.size(1),
              k_pages_contig.size(1),
              block_table_contig.size(1),
              k_pages_contig.size(2),
              mask_seq,
              q_contig.size(3),
              scale_value);
      return;
    default:
      paged_decode_attention_q1_forward_kernel<scalar_t, t10::cuda::attention::kLargeRowThreads, HasMask>
          <<<blocks, t10::cuda::attention::kLargeRowThreads, 0, stream>>>(
              q_contig.data_ptr<scalar_t>(),
              k_pages_contig.data_ptr<scalar_t>(),
              v_pages_contig.data_ptr<scalar_t>(),
              block_table_contig.data_ptr<int64_t>(),
              lengths_contig.data_ptr<int64_t>(),
              mask_ptr,
              mask_kind,
              out.data_ptr<scalar_t>(),
              q_contig.size(0),
              q_contig.size(1),
              k_pages_contig.size(1),
              block_table_contig.size(1),
              k_pages_contig.size(2),
              mask_seq,
              q_contig.size(3),
              scale_value);
      return;
  }
}

}  // namespace

bool HasCudaPagedAttentionDecodeKernel() {
  return true;
}

torch::Tensor CudaPagedAttentionDecodeForward(
    const torch::Tensor& q,
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    const c10::optional<torch::Tensor>& attn_mask,
    const c10::optional<double>& scale,
    int64_t known_mask_seq) {
  TORCH_CHECK(UseCudaPagedAttentionDecodeKernel(q, k_pages, v_pages, block_table, lengths, attn_mask),
              "CudaPagedAttentionDecodeForward: unsupported tensor configuration");

  c10::cuda::CUDAGuard device_guard{q.device()};

  auto q_contig = q.contiguous();
  auto k_pages_contig = k_pages.contiguous();
  auto v_pages_contig = v_pages.contiguous();
  auto block_table_contig = block_table.to(torch::kLong).contiguous();
  auto lengths_contig = lengths.to(torch::kLong).contiguous();
  torch::Tensor mask_contig;
  int mask_kind = 0;
  const void* mask_ptr = nullptr;
  int64_t mask_seq = known_mask_seq >= 0 ? known_mask_seq : (lengths_contig.numel() > 0 ? lengths_contig.max().item<int64_t>() : 0);
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    mask_contig = attn_mask.value().contiguous();
    mask_seq = mask_contig.size(3);
    if (mask_contig.scalar_type() == torch::kBool) {
      mask_kind = 1;
    } else if (mask_contig.scalar_type() == q_contig.scalar_type()) {
      mask_kind = 2;
    } else {
      mask_kind = 3;
    }
    mask_ptr = mask_contig.data_ptr();
  }

  auto out = torch::zeros_like(q_contig);
  const float scale_value = static_cast<float>(
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(static_cast<double>(q_contig.size(3)))));
  const int threads = SelectPagedDecodeThreads(mask_seq, q_contig.size(3));
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q_contig.scalar_type(),
      "model_stack_cuda_paged_attention_decode_forward",
      [&] {
        if (mask_ptr == nullptr) {
          LaunchPagedDecodeAttentionQ1<scalar_t, false>(
              q_contig,
              k_pages_contig,
              v_pages_contig,
              block_table_contig,
              lengths_contig,
              nullptr,
              0,
              out,
              threads,
              mask_seq,
              scale_value,
              stream.stream());
        } else {
          LaunchPagedDecodeAttentionQ1<scalar_t, true>(
              q_contig,
              k_pages_contig,
              v_pages_contig,
              block_table_contig,
              lengths_contig,
              mask_ptr,
              mask_kind,
              out,
              threads,
              mask_seq,
              scale_value,
              stream.stream());
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
