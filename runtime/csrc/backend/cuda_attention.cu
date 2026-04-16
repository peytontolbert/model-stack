#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

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
