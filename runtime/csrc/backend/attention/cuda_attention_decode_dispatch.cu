#include <torch/extension.h>

#include "cuda_attention_decode.cuh"

namespace t10::cuda::attention {

void LaunchPlannedAttentionDecodeDispatcher(
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
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q_contig.scalar_type(),
      "model_stack_cuda_attention_decode_dispatch",
      [&] {
        LaunchPlannedAttentionDecode<scalar_t>(
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
      });
}

}  // namespace t10::cuda::attention
