#pragma once

#include <torch/extension.h>

#include "bitnet_common.cuh"

#include <vector>

namespace t10::bitnet {

inline torch::Tensor SplitBitNetHeads(const torch::Tensor& x, int64_t num_heads) {
  TORCH_CHECK(x.defined(), "SplitBitNetHeads: x must be defined");
  TORCH_CHECK(x.dim() == 3, "SplitBitNetHeads: x must have shape (B, T, D)");
  TORCH_CHECK(num_heads > 0, "SplitBitNetHeads: num_heads must be positive");
  TORCH_CHECK(x.size(2) % num_heads == 0, "SplitBitNetHeads: model dim must be divisible by num_heads");
  const auto head_dim = x.size(2) / num_heads;
  return x.contiguous().view({x.size(0), x.size(1), num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous();
}

inline std::vector<torch::Tensor> SplitBitNetFusedQkv(
    const torch::Tensor& fused,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads) {
  TORCH_CHECK(fused.defined(), "SplitBitNetFusedQkv: fused tensor must be defined");
  TORCH_CHECK(fused.dim() == 3, "SplitBitNetFusedQkv: fused tensor must have shape (B, T, D)");
  TORCH_CHECK(q_size > 0 && k_size > 0 && v_size > 0, "SplitBitNetFusedQkv: q/k/v sizes must be positive");
  TORCH_CHECK(fused.size(2) == q_size + k_size + v_size, "SplitBitNetFusedQkv: fused width mismatch");
  return {
      SplitBitNetHeads(fused.slice(-1, 0, q_size), q_heads),
      SplitBitNetHeads(fused.slice(-1, q_size, q_size + k_size), kv_heads),
      SplitBitNetHeads(fused.slice(-1, q_size + k_size, q_size + k_size + v_size), kv_heads),
  };
}

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
    int64_t kv_heads);

std::vector<torch::Tensor> LaunchBitNetAttentionPrefillPackedQkv(
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
    int64_t kv_heads);

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
    int64_t kv_heads);

bool HasCudaBitNetFusedQkvKernel();

}  // namespace t10::bitnet
