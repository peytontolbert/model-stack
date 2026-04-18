#pragma once

#include <torch/extension.h>

namespace t10::bitnet {

constexpr int64_t kLayoutHeaderLen = 13;
constexpr int64_t kIdxFormatVersion = 0;
constexpr int64_t kIdxTileN = 1;
constexpr int64_t kIdxTileK = 2;
constexpr int64_t kIdxLogicalOut = 3;
constexpr int64_t kIdxLogicalIn = 4;
constexpr int64_t kIdxPaddedOut = 5;
constexpr int64_t kIdxPaddedIn = 6;
constexpr int64_t kIdxScaleGranularity = 7;
constexpr int64_t kIdxScaleGroupSize = 8;
constexpr int64_t kIdxInterleaveMode = 9;
constexpr int64_t kIdxArchMin = 10;
constexpr int64_t kIdxSegmentCount = 11;
constexpr int64_t kIdxFlags = 12;

struct LayoutInfo {
  int64_t format_version = 0;
  int64_t tile_n = 0;
  int64_t tile_k = 0;
  int64_t logical_out_features = 0;
  int64_t logical_in_features = 0;
  int64_t padded_out_features = 0;
  int64_t padded_in_features = 0;
  int64_t scale_granularity = 0;
  int64_t scale_group_size = 0;
  int64_t interleave_mode = 0;
  int64_t arch_min = 0;
  int64_t segment_count = 0;
  int64_t flags = 0;
};

inline LayoutInfo ParseLayoutHeader(const torch::Tensor& layout_header) {
  TORCH_CHECK(layout_header.defined(), "bitnet layout_header must be defined");
  TORCH_CHECK(layout_header.dim() == 1, "bitnet layout_header must be rank-1");
  TORCH_CHECK(layout_header.numel() >= kLayoutHeaderLen,
              "bitnet layout_header must have at least ",
              kLayoutHeaderLen,
              " entries");
  auto header_cpu = layout_header.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kLong)).contiguous();
  auto acc = header_cpu.accessor<int64_t, 1>();
  LayoutInfo info;
  info.format_version = acc[kIdxFormatVersion];
  info.tile_n = acc[kIdxTileN];
  info.tile_k = acc[kIdxTileK];
  info.logical_out_features = acc[kIdxLogicalOut];
  info.logical_in_features = acc[kIdxLogicalIn];
  info.padded_out_features = acc[kIdxPaddedOut];
  info.padded_in_features = acc[kIdxPaddedIn];
  info.scale_granularity = acc[kIdxScaleGranularity];
  info.scale_group_size = acc[kIdxScaleGroupSize];
  info.interleave_mode = acc[kIdxInterleaveMode];
  info.arch_min = acc[kIdxArchMin];
  info.segment_count = acc[kIdxSegmentCount];
  info.flags = acc[kIdxFlags];
  TORCH_CHECK(info.format_version == 1,
              "bitnet layout_header format_version must be 1 (got ",
              info.format_version,
              ")");
  TORCH_CHECK(info.logical_out_features > 0 && info.logical_in_features > 0,
              "bitnet logical dimensions must be positive");
  TORCH_CHECK(info.padded_out_features >= info.logical_out_features &&
                  info.padded_in_features >= info.logical_in_features,
              "bitnet padded dimensions must be at least the logical dimensions");
  TORCH_CHECK(info.segment_count > 0, "bitnet segment_count must be positive");
  return info;
}

inline void ValidateSegmentOffsets(
    const torch::Tensor& segment_offsets,
    const LayoutInfo& layout) {
  TORCH_CHECK(segment_offsets.defined(), "bitnet segment_offsets must be defined");
  TORCH_CHECK(segment_offsets.dim() == 1, "bitnet segment_offsets must be rank-1");
  TORCH_CHECK(segment_offsets.numel() == layout.segment_count + 1,
              "bitnet segment_offsets length mismatch");
  auto offsets_cpu = segment_offsets.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kLong)).contiguous();
  auto acc = offsets_cpu.accessor<int64_t, 1>();
  TORCH_CHECK(acc[0] == 0, "bitnet segment_offsets must start at 0");
  TORCH_CHECK(acc[layout.segment_count] == layout.logical_out_features,
              "bitnet segment_offsets must end at logical_out_features");
  for (int64_t idx = 1; idx < offsets_cpu.size(0); ++idx) {
    TORCH_CHECK(acc[idx] >= acc[idx - 1], "bitnet segment_offsets must be non-decreasing");
  }
}

inline float ResolveRowScale(
    int64_t out_idx,
    const LayoutInfo& layout,
    const torch::Tensor& scale_values,
    const torch::Tensor& segment_offsets) {
  auto scale_cpu = scale_values.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
  auto scale_acc = scale_cpu.accessor<float, 1>();
  if (layout.scale_granularity == 0) {
    TORCH_CHECK(scale_cpu.numel() >= 1, "bitnet per-matrix scaling requires at least one value");
    return scale_acc[0];
  }
  if (layout.scale_granularity == 1) {
    TORCH_CHECK(scale_cpu.numel() == layout.segment_count, "bitnet per-segment scaling size mismatch");
    auto offsets_cpu = segment_offsets.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kLong)).contiguous();
    auto offsets_acc = offsets_cpu.accessor<int64_t, 1>();
    for (int64_t seg_idx = 0; seg_idx < layout.segment_count; ++seg_idx) {
      if (out_idx >= offsets_acc[seg_idx] && out_idx < offsets_acc[seg_idx + 1]) {
        return scale_acc[seg_idx];
      }
    }
    return 0.0f;
  }
  if (layout.scale_granularity == 2) {
    TORCH_CHECK(layout.scale_group_size > 0,
                "bitnet per-output-group scaling requires a positive scale_group_size");
    const int64_t group_idx = out_idx / layout.scale_group_size;
    const int64_t expected_groups =
        (layout.logical_out_features + layout.scale_group_size - 1) / layout.scale_group_size;
    TORCH_CHECK(scale_cpu.numel() == expected_groups, "bitnet per-output-group scaling size mismatch");
    return scale_acc[group_idx];
  }
  TORCH_CHECK(false, "Unsupported bitnet scale granularity: ", layout.scale_granularity);
  return 0.0f;
}

}  // namespace t10::bitnet
