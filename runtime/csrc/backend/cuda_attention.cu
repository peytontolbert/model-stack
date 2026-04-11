#include <torch/extension.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void attention_forward_kernel(
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

  __shared__ float shared[kThreads];

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
      const float weight = expf(score - row_max) / denom;
      acc += weight * static_cast<float>(v[kv_base + s * dh + d]);
    }
    out[q_base + d] = static_cast<scalar_t>(acc);
  }
}

bool UseCudaAttentionKernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask) {
  if (!q.is_cuda() || !k.is_cuda() || !v.is_cuda()) {
    return false;
  }
  if (!(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16)) {
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
  auto k_all = k;
  auto v_all = v;
  if (q.size(1) != k.size(1)) {
    const auto repeat = q.size(1) / k.size(1);
    auto head_index = torch::arange(k.size(1), torch::TensorOptions().dtype(torch::kLong).device(k.device()))
                          .repeat_interleave(repeat);
    k_all = torch::index_select(k, 1, head_index);
    v_all = torch::index_select(v, 1, head_index);
  }

  if (!UseCudaAttentionKernel(q, k, v, attn_mask)) {
    const double scale_value = scale.has_value() ? scale.value() : (1.0 / std::sqrt(static_cast<double>(q.size(3))));
    auto scores = torch::matmul(q, k_all.transpose(2, 3)) * scale_value;
    if (attn_mask.has_value() && attn_mask.value().defined()) {
      if (attn_mask.value().scalar_type() == torch::kBool) {
        scores = scores.masked_fill(attn_mask.value(), -INFINITY);
      } else {
        scores = scores + attn_mask.value().to(scores.scalar_type());
      }
    }
    if (is_causal) {
      auto causal = torch::ones({q.size(2), k_all.size(2)}, torch::TensorOptions().dtype(torch::kBool).device(q.device())).triu(1);
      scores = scores.masked_fill(causal.view({1, 1, q.size(2), k_all.size(2)}), -INFINITY);
    }
    auto probs = torch::softmax(scores.to(torch::kFloat32), -1).to(q.scalar_type());
    return torch::matmul(probs, v_all);
  }

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
  const auto batch = q_contig.size(0);
  const auto q_heads = q_contig.size(1);
  const auto kv_heads = k_contig.size(1);
  const auto tq = q_contig.size(2);
  const auto sk = k_contig.size(2);
  const auto dh = q_contig.size(3);
  const float scale_value = static_cast<float>(
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(static_cast<double>(dh))));
  const dim3 blocks(static_cast<unsigned int>(batch * q_heads * tq));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      q_contig.scalar_type(),
      "model_stack_cuda_attention_forward",
      [&] {
        attention_forward_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            q_contig.data_ptr<scalar_t>(),
            k_contig.data_ptr<scalar_t>(),
            v_contig.data_ptr<scalar_t>(),
            mask_ptr,
            mask_kind,
            out.data_ptr<scalar_t>(),
            batch,
            q_heads,
            kv_heads,
            tq,
            sk,
            dh,
            scale_value,
            is_causal);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
