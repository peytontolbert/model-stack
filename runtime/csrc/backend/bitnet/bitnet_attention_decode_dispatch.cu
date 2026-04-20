#include <torch/extension.h>

#include "bitnet_attention_common.cuh"

namespace t10::bitnet {

std::vector<torch::Tensor> LaunchBitNetAttentionDecodePackedQkv(
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
  auto fused = CudaBitNetLinearForward(
      x,
      packed_weight,
      scale_values,
      layout_header,
      segment_offsets,
      packed_bias,
      c10::nullopt);
  return SplitBitNetFusedQkv(fused, q_size, k_size, v_size, q_heads, kv_heads);
}

}  // namespace t10::bitnet
