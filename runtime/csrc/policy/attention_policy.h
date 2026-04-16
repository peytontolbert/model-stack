#pragma once

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
  if (desc.kv_len <= 64 && desc.head_dim <= 64) {
    return 64;
  }
  if (desc.kv_len <= 128 && desc.head_dim <= 128) {
    return 128;
  }
  return 256;
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
