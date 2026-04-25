#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "cuda_attention_sm80_flash_prefill.cuh"

#include <cmath>
#include <unordered_map>

#ifdef MODEL_STACK_WITH_LOCAL_FLASH_STYLE_PREFILL
#include "../../../../other_repos/flash-attention/csrc/flash_attn/src/flash.h"
#include "../../../../other_repos/flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h"
#include <cutlass/numeric_types.h>
#endif

namespace t10::cuda::attention {

#ifdef MODEL_STACK_WITH_LOCAL_FLASH_STYLE_PREFILL

template <typename scalar_t>
struct ModelStackSm80FlashScalar;

template <>
struct ModelStackSm80FlashScalar<at::Half> {
  using type = cutlass::half_t;
};

template <>
struct ModelStackSm80FlashScalar<at::BFloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename scalar_t>
using ModelStackSm80FlashScalarT = typename ModelStackSm80FlashScalar<scalar_t>::type;

struct ModelStackSm80FlashWorkspaceKey {
  int device_index = -1;
  uintptr_t stream_handle = 0;

  bool operator==(const ModelStackSm80FlashWorkspaceKey& other) const {
    return device_index == other.device_index && stream_handle == other.stream_handle;
  }
};

struct ModelStackSm80FlashWorkspaceKeyHash {
  size_t operator()(const ModelStackSm80FlashWorkspaceKey& key) const {
    return (static_cast<size_t>(static_cast<uint32_t>(key.device_index)) << 32) ^
        static_cast<size_t>(key.stream_handle);
  }
};

inline int RoundUpToMultiple(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

inline torch::Tensor AcquireModelStackSm80FlashSoftmaxLseWorkspace(
    const torch::Tensor& reference,
    int64_t batch,
    int64_t heads,
    int64_t q_len) {
  const auto current_stream = at::cuda::getCurrentCUDAStream(reference.get_device());
  const int64_t needed = batch * heads * q_len;
  static thread_local std::unordered_map<
      ModelStackSm80FlashWorkspaceKey,
      torch::Tensor,
      ModelStackSm80FlashWorkspaceKeyHash>
      cache;

  const ModelStackSm80FlashWorkspaceKey key{
      reference.get_device(),
      reinterpret_cast<uintptr_t>(current_stream.stream())};
  auto& buffer = cache[key];
  if (!buffer.defined() ||
      buffer.device() != reference.device() ||
      buffer.scalar_type() != torch::kFloat32 ||
      buffer.numel() < needed) {
    buffer = torch::empty({needed}, reference.options().dtype(torch::kFloat32));
  }
  c10::cuda::CUDACachingAllocator::recordStream(buffer.storage().data_ptr(), current_stream);
  return buffer.narrow(0, 0, needed).view({batch, heads, q_len});
}

template <typename scalar_t>
inline void SetModelStackSm80FlashFwdParams(
    FLASH_NAMESPACE::Flash_fwd_params& params,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const torch::Tensor& softmax_lse,
    const t10::desc::AttentionDesc& desc,
    float scale_value) {
  params = {};

  params.is_bf16 = std::is_same_v<scalar_t, at::BFloat16>;

  params.q_ptr = const_cast<scalar_t*>(q_contig.data_ptr<scalar_t>());
  params.k_ptr = const_cast<scalar_t*>(k_contig.data_ptr<scalar_t>());
  params.v_ptr = const_cast<scalar_t*>(v_contig.data_ptr<scalar_t>());
  params.o_ptr = out.data_ptr<scalar_t>();

  // Our runtime uses BHSD, and the flash-style kernel contract is stride-based,
  // so we can pass the contiguous BHSD tensors directly.
  params.q_batch_stride = q_contig.stride(0);
  params.k_batch_stride = k_contig.stride(0);
  params.v_batch_stride = v_contig.stride(0);
  params.o_batch_stride = out.stride(0);

  params.q_row_stride = q_contig.stride(2);
  params.k_row_stride = k_contig.stride(2);
  params.v_row_stride = v_contig.stride(2);
  params.o_row_stride = out.stride(2);

  params.q_head_stride = q_contig.stride(1);
  params.k_head_stride = k_contig.stride(1);
  params.v_head_stride = v_contig.stride(1);
  params.o_head_stride = out.stride(1);

  params.p_ptr = nullptr;
  params.softmax_lse_ptr = softmax_lse.data_ptr<float>();

  params.b = static_cast<int>(desc.batch);
  params.h = static_cast<int>(desc.q_heads);
  params.h_k = static_cast<int>(desc.kv_heads);
  params.h_h_k_ratio = static_cast<int>(desc.q_heads / desc.kv_heads);
  params.seqlen_q = static_cast<int>(desc.q_len);
  params.seqlen_k = static_cast<int>(desc.kv_len);
  params.seqlen_knew = 0;
  params.seqlen_q_rounded = RoundUpToMultiple(static_cast<int>(desc.q_len), 128);
  params.seqlen_k_rounded = RoundUpToMultiple(static_cast<int>(desc.kv_len), 128);
  params.d = static_cast<int>(desc.head_dim);
  params.d_rounded =
      RoundUpToMultiple(static_cast<int>(desc.head_dim), desc.head_dim <= 128 ? 32 : 64);
  params.rotary_dim = 0;
  params.total_q = static_cast<int>(desc.batch * desc.q_len);

  params.scale_softmax = scale_value;
  params.scale_softmax_log2 = scale_value * static_cast<float>(M_LOG2E);
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = scale_value;

  params.is_causal = true;
  params.window_size_left = -1;
  params.window_size_right = 0;
  params.softcap = 0.0f;
  params.is_seqlens_k_cumulative = true;
  params.is_rotary_interleaved = false;
  params.num_splits = 0;
  params.unpadded_lse = false;
  params.seqlenq_ngroups_swapped = false;
}

template <typename scalar_t>
inline bool TryLaunchModelStackSm80FlashAttentionPrefillImpl(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  c10::cuda::CUDAGuard device_guard{q_contig.device()};

  auto softmax_lse = AcquireModelStackSm80FlashSoftmaxLseWorkspace(
      q_contig,
      desc.batch,
      desc.q_heads,
      desc.q_len);

  FLASH_NAMESPACE::Flash_fwd_params params;
  SetModelStackSm80FlashFwdParams<scalar_t>(
      params,
      q_contig,
      k_contig,
      v_contig,
      out,
      softmax_lse,
      desc,
      scale_value);

  FLASH_NAMESPACE::run_mha_fwd_hdim64<ModelStackSm80FlashScalarT<scalar_t>, true>(params, stream);
  return true;
}

bool TryLaunchModelStackSm80FlashAttentionPrefillF16(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  return TryLaunchModelStackSm80FlashAttentionPrefillImpl<at::Half>(
      q_contig,
      k_contig,
      v_contig,
      out,
      desc,
      scale_value,
      stream);
}

bool TryLaunchModelStackSm80FlashAttentionPrefillBF16(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  return TryLaunchModelStackSm80FlashAttentionPrefillImpl<at::BFloat16>(
      q_contig,
      k_contig,
      v_contig,
      out,
      desc,
      scale_value,
      stream);
}

#else

bool TryLaunchModelStackSm80FlashAttentionPrefillF16(
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const t10::desc::AttentionDesc&,
    float,
    cudaStream_t) {
  return false;
}

bool TryLaunchModelStackSm80FlashAttentionPrefillBF16(
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const t10::desc::AttentionDesc&,
    float,
    cudaStream_t) {
  return false;
}

#endif

}  // namespace t10::cuda::attention
