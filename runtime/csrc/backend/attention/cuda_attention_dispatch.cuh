#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "../../descriptors/attention_desc.h"
#include "../../policy/attention_policy.h"

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
    cudaStream_t stream);

void LaunchPlannedAttentionPrefillDispatcher(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const void* mask_ptr,
    int mask_kind,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    const t10::policy::AttentionPlan& plan,
    float scale_value,
    cudaStream_t stream);

}  // namespace t10::cuda::attention
