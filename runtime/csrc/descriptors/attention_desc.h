#pragma once

#include <torch/extension.h>

#include <c10/util/Optional.h>

#include <cstdint>

namespace t10::desc {

enum class AttentionMaskKind : int {
  kNone = 0,
  kBool = 1,
  kAdditiveSameDtype = 2,
  kAdditiveFloat32 = 3,
};

enum class AttentionPhase : int {
  kPrefill = 0,
  kDecode = 1,
};

enum class AttentionHeadMode : int {
  kMHA = 0,
  kGQA = 1,
  kMQA = 2,
};

enum class AttentionLayoutKind : int {
  kBHSD = 0,
};

struct AttentionDesc {
  torch::ScalarType dtype;
  int64_t batch;
  int64_t q_heads;
  int64_t kv_heads;
  int64_t q_len;
  int64_t kv_len;
  int64_t head_dim;
  AttentionMaskKind mask_kind;
  AttentionPhase phase;
  AttentionHeadMode head_mode;
  AttentionLayoutKind q_layout;
  AttentionLayoutKind kv_layout;
  bool causal;
};

inline bool IsSupportedAttentionDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

inline AttentionPhase ResolveAttentionPhase(int64_t q_len) {
  return q_len == 1 ? AttentionPhase::kDecode : AttentionPhase::kPrefill;
}

inline AttentionHeadMode ResolveAttentionHeadMode(int64_t q_heads, int64_t kv_heads) {
  if (kv_heads == 1) {
    return AttentionHeadMode::kMQA;
  }
  if (q_heads == kv_heads) {
    return AttentionHeadMode::kMHA;
  }
  return AttentionHeadMode::kGQA;
}

inline AttentionMaskKind ResolveAttentionMaskKind(
    const c10::optional<torch::Tensor>& attn_mask,
    torch::ScalarType q_dtype) {
  if (!attn_mask.has_value() || !attn_mask.value().defined()) {
    return AttentionMaskKind::kNone;
  }
  const auto& mask = attn_mask.value();
  if (mask.scalar_type() == torch::kBool) {
    return AttentionMaskKind::kBool;
  }
  if (mask.scalar_type() == q_dtype) {
    return AttentionMaskKind::kAdditiveSameDtype;
  }
  return AttentionMaskKind::kAdditiveFloat32;
}

}  // namespace t10::desc
