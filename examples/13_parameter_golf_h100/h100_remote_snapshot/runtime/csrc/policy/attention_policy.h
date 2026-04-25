#pragma once

#include <algorithm>
#include <cstdlib>
#include <cmath>

#include "../descriptors/attention_desc.h"

namespace t10::policy {

enum class AttentionKernelKind : int {
  kGenericPrefill = 0,
  kGenericDecode = 1,
  kDecodeQ1 = 2,
  kDecodeQ1Hdim32 = 3,
  kDecodeQ1Hdim64 = 4,
  kDecodeQ1Hdim96 = 5,
  kDecodeQ1Hdim128 = 6,
  kPrefillHdim32 = 7,
  kPrefillHdim64 = 8,
  kPrefillHdim96 = 9,
  kPrefillHdim128 = 10,
  kDecodeQ1MHA = 11,
  kPrefillMHA = 12,
  kDecodeQ1NoMask = 13,
  kDecodeQ1MHANoMask = 14,
  kPrefillMHANoMask = 15,
  kGenericPrefillNoMask = 16,
};

struct AttentionPlan {
  AttentionKernelKind kernel;
  int row_reduce_threads;
  int head_dim_bucket;
  t10::desc::AttentionPhase phase;
  t10::desc::AttentionHeadMode head_mode;
};

inline int BucketAttentionHeadDim(int64_t head_dim) {
  if (head_dim <= 32) {
    return 32;
  }
  if (head_dim <= 64) {
    return 64;
  }
  if (head_dim <= 96) {
    return 96;
  }
  if (head_dim <= 128) {
    return 128;
  }
  if (head_dim <= 192) {
    return 192;
  }
  if (head_dim <= 256) {
    return 256;
  }
  return -1;
}

inline int SelectAttentionRowThreads(const t10::desc::AttentionDesc& desc) {
  if (desc.phase == t10::desc::AttentionPhase::kPrefill) {
    if (desc.head_dim <= 64) {
      if (desc.kv_len >= 512) {
        return 256;
      }
      if (desc.kv_len >= 256) {
        return 128;
      }
      return 64;
    }
    if (desc.head_dim <= 128) {
      return 128;
    }
    return 256;
  }
  if (desc.kv_len <= 64 && desc.head_dim <= 64) {
    return 64;
  }
  if (desc.kv_len <= 128 && desc.head_dim <= 128) {
    return 128;
  }
  return 256;
}

inline bool AttentionSm80FlashPrefillDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_FLASH");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline int64_t AttentionSm80FlashPrefillMinSeq() {
  const char* env = std::getenv("MODEL_STACK_SM80_FLASH_PREFILL_MIN_SEQ");
  if (env == nullptr || env[0] == '\0') {
    return 4096;
  }
  char* end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed <= 0) {
    return 4096;
  }
  return static_cast<int64_t>(parsed);
}

inline bool SupportsAttentionSm80FlashPrefill(const t10::desc::AttentionDesc& desc) {
  return desc.phase == t10::desc::AttentionPhase::kPrefill &&
      desc.causal &&
      desc.mask_kind == t10::desc::AttentionMaskKind::kNone &&
      desc.head_mode == t10::desc::AttentionHeadMode::kMHA &&
      desc.q_layout == t10::desc::AttentionLayoutKind::kBHSD &&
      desc.kv_layout == t10::desc::AttentionLayoutKind::kBHSD &&
      desc.head_dim == 64 &&
      desc.q_len > 0 &&
      desc.q_len == desc.kv_len;
}

inline bool PreferAttentionSm80FlashPrefill(const t10::desc::AttentionDesc& desc) {
  return !AttentionSm80FlashPrefillDisabled() &&
      SupportsAttentionSm80FlashPrefill(desc) &&
      desc.q_len >= AttentionSm80FlashPrefillMinSeq();
}

inline bool AttentionHasDeadTopLeftCausalKvTail(const t10::desc::AttentionDesc& desc) {
  return desc.phase == t10::desc::AttentionPhase::kPrefill &&
      desc.causal &&
      desc.mask_kind == t10::desc::AttentionMaskKind::kNone &&
      desc.q_len > 0 &&
      desc.q_len < desc.kv_len;
}

inline int64_t AttentionEffectiveKvLen(const t10::desc::AttentionDesc& desc) {
  return AttentionHasDeadTopLeftCausalKvTail(desc)
      ? desc.q_len
      : desc.kv_len;
}

inline int AttentionSplitKvBlockN(const t10::desc::AttentionDesc& desc) {
  if (desc.head_dim <= 64) {
    return 256;
  }
  if (desc.head_dim <= 128) {
    return 128;
  }
  return 64;
}

inline int AttentionSplitKvNumMBlocks(const t10::desc::AttentionDesc& desc) {
  return static_cast<int>((desc.q_len + 64 - 1) / 64);
}

inline int AttentionSplitKvNumNBlocks(const t10::desc::AttentionDesc& desc) {
  const int block_n = AttentionSplitKvBlockN(desc);
  const int64_t effective_kv_len = AttentionEffectiveKvLen(desc);
  return static_cast<int>((effective_kv_len + block_n - 1) / block_n);
}

inline bool SupportsAttentionSplitKv(const t10::desc::AttentionDesc& desc) {
  return desc.phase == t10::desc::AttentionPhase::kPrefill &&
      desc.causal &&
      desc.mask_kind == t10::desc::AttentionMaskKind::kNone &&
      !AttentionHasDeadTopLeftCausalKvTail(desc) &&
      desc.head_mode == t10::desc::AttentionHeadMode::kMHA &&
      desc.q_layout == t10::desc::AttentionLayoutKind::kBHSD &&
      desc.kv_layout == t10::desc::AttentionLayoutKind::kBHSD &&
      desc.head_dim > 0 &&
      desc.head_dim <= 128 &&
      desc.q_len > 0 &&
      desc.kv_len > 0;
}

inline int SelectAttentionSplitKvSplits(
    const t10::desc::AttentionDesc& desc,
    int effective_sms,
    int max_splits = 128) {
  if (!SupportsAttentionSplitKv(desc) || effective_sms <= 0) {
    return 1;
  }

  const int num_n_blocks = AttentionSplitKvNumNBlocks(desc);
  const int num_m_blocks = AttentionSplitKvNumMBlocks(desc);
  const int batch_nheads_mblocks =
      static_cast<int>(desc.batch * desc.q_heads * num_m_blocks);
  if (batch_nheads_mblocks >= static_cast<int>(0.8f * effective_sms)) {
    return 1;
  }

  max_splits = std::min({max_splits, effective_sms, num_n_blocks});
  if (max_splits <= 1) {
    return 1;
  }

  auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
  auto is_split_eligible = [&](int num_splits) {
    return num_splits == 1 ||
        ceil_div(num_n_blocks, num_splits) != ceil_div(num_n_blocks, num_splits - 1);
  };

  float max_efficiency = 0.0f;
  int best_split = 1;
  for (int num_splits = 1; num_splits <= max_splits; ++num_splits) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    const float waves =
        static_cast<float>(batch_nheads_mblocks * num_splits) /
        static_cast<float>(effective_sms);
    const float efficiency = waves / std::ceil(waves);
    if (efficiency > max_efficiency) {
      max_efficiency = efficiency;
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; ++num_splits) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    const float waves =
        static_cast<float>(batch_nheads_mblocks * num_splits) /
        static_cast<float>(effective_sms);
    const float efficiency = waves / std::ceil(waves);
    if (efficiency >= 0.85f * max_efficiency) {
      best_split = num_splits;
      break;
    }
  }
  return best_split;
}

inline AttentionPlan ResolveAttentionPlan(const t10::desc::AttentionDesc& desc) {
  if (desc.phase == t10::desc::AttentionPhase::kDecode && desc.q_len == 1) {
    if (desc.head_dim == 32) {
      return AttentionPlan{
          AttentionKernelKind::kDecodeQ1Hdim32,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.head_dim == 64) {
      return AttentionPlan{
          AttentionKernelKind::kDecodeQ1Hdim64,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.head_dim == 96) {
      return AttentionPlan{
          AttentionKernelKind::kDecodeQ1Hdim96,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.head_dim == 128) {
      return AttentionPlan{
          AttentionKernelKind::kDecodeQ1Hdim128,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.mask_kind == t10::desc::AttentionMaskKind::kNone) {
      if (desc.head_mode == t10::desc::AttentionHeadMode::kMHA) {
        return AttentionPlan{
            AttentionKernelKind::kDecodeQ1MHANoMask,
            SelectAttentionRowThreads(desc),
            BucketAttentionHeadDim(desc.head_dim),
            desc.phase,
            desc.head_mode};
      }
      return AttentionPlan{
          AttentionKernelKind::kDecodeQ1NoMask,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.head_mode == t10::desc::AttentionHeadMode::kMHA) {
      return AttentionPlan{
          AttentionKernelKind::kDecodeQ1MHA,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    return AttentionPlan{
        AttentionKernelKind::kDecodeQ1,
        SelectAttentionRowThreads(desc),
        BucketAttentionHeadDim(desc.head_dim),
        desc.phase,
        desc.head_mode};
  }
  if (desc.phase == t10::desc::AttentionPhase::kPrefill) {
    if (desc.head_dim == 32) {
      return AttentionPlan{
          AttentionKernelKind::kPrefillHdim32,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.head_dim == 64) {
      return AttentionPlan{
          AttentionKernelKind::kPrefillHdim64,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.head_dim == 96) {
      return AttentionPlan{
          AttentionKernelKind::kPrefillHdim96,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.head_dim == 128) {
      return AttentionPlan{
          AttentionKernelKind::kPrefillHdim128,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.head_mode == t10::desc::AttentionHeadMode::kMHA) {
      if (desc.mask_kind == t10::desc::AttentionMaskKind::kNone) {
        return AttentionPlan{
            AttentionKernelKind::kPrefillMHANoMask,
            SelectAttentionRowThreads(desc),
            BucketAttentionHeadDim(desc.head_dim),
            desc.phase,
            desc.head_mode};
      }
      return AttentionPlan{
          AttentionKernelKind::kPrefillMHA,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
    if (desc.mask_kind == t10::desc::AttentionMaskKind::kNone) {
      return AttentionPlan{
          AttentionKernelKind::kGenericPrefillNoMask,
          SelectAttentionRowThreads(desc),
          BucketAttentionHeadDim(desc.head_dim),
          desc.phase,
          desc.head_mode};
    }
  }
  return AttentionPlan{
      desc.phase == t10::desc::AttentionPhase::kDecode
          ? AttentionKernelKind::kGenericDecode
          : AttentionKernelKind::kGenericPrefill,
      SelectAttentionRowThreads(desc),
      BucketAttentionHeadDim(desc.head_dim),
      desc.phase,
      desc.head_mode};
}

}  // namespace t10::policy
