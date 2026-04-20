#include <torch/extension.h>

#include "bitnet_attention_common.cuh"

namespace t10::bitnet {

bool HasCudaBitNetFusedQkvKernel() {
  return true;
}

std::vector<torch::Tensor> CudaBitNetFusedQkvPackedHeadsProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& packed_bias,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads) {
  TORCH_CHECK(x.defined() && packed_weight.defined() && scale_values.defined() && layout_header.defined() &&
                  segment_offsets.defined(),
              "CudaBitNetFusedQkvPackedHeadsProjectionForward: tensors must be defined");
  TORCH_CHECK(x.is_cuda() && packed_weight.is_cuda() && scale_values.is_cuda() && layout_header.is_cuda() &&
                  segment_offsets.is_cuda(),
              "CudaBitNetFusedQkvPackedHeadsProjectionForward: tensors must be CUDA");
  TORCH_CHECK(x.dim() == 3, "CudaBitNetFusedQkvPackedHeadsProjectionForward: x must have shape (B, T, D)");
  TORCH_CHECK(q_size > 0 && k_size > 0 && v_size > 0,
              "CudaBitNetFusedQkvPackedHeadsProjectionForward: q/k/v sizes must be positive");
  TORCH_CHECK(q_heads > 0 && kv_heads > 0,
              "CudaBitNetFusedQkvPackedHeadsProjectionForward: q_heads and kv_heads must be positive");
  const auto rows = x.size(0) * x.size(1);
  if (rows <= 8) {
    return LaunchBitNetAttentionDecodePackedQkv(
        x,
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        packed_bias,
        q_size,
        k_size,
        v_size,
        q_heads,
        kv_heads);
  }
  return LaunchBitNetAttentionPrefillPackedQkv(
      x,
      packed_weight,
      scale_values,
      layout_header,
      segment_offsets,
      packed_bias,
      q_size,
      k_size,
      v_size,
      q_heads,
      kv_heads);
}

}  // namespace t10::bitnet
