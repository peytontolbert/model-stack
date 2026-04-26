#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "bitnet_common.cuh"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace t10::bitnet {
namespace {

std::string NormalizeModeName(const std::string& mode) {
  std::string normalized = mode;
  std::transform(
      normalized.begin(),
      normalized.end(),
      normalized.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (normalized.empty() || normalized == "off") {
    return "none";
  }
  return normalized;
}

}  // namespace

bool HasCudaBitNetLinearKernel() {
  return true;
}

torch::Tensor CudaBitNetLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined() && packed_weight.defined() && scale_values.defined() && layout_header.defined() &&
                  segment_offsets.defined(),
              "CudaBitNetLinearForward: tensors must be defined");
  TORCH_CHECK(x.is_cuda() && packed_weight.is_cuda() && scale_values.is_cuda() && layout_header.is_cuda() &&
                  segment_offsets.is_cuda(),
              "CudaBitNetLinearForward: tensors must be CUDA");
  TORCH_CHECK(x.dim() >= 2, "CudaBitNetLinearForward: x must have rank >= 2");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetLinearForward: x must use float32, float16, or bfloat16");

  c10::cuda::CUDAGuard device_guard(x.device());

  const auto layout = ParseLayoutHeader(layout_header);
  ValidateSegmentOffsets(segment_offsets, layout);
  ValidateScaleValues(scale_values, layout);
  TORCH_CHECK(SupportsScaleGranularity(layout),
              "CudaBitNetLinearForward: unsupported scale granularity");
  TORCH_CHECK(packed_weight.dim() == 2, "CudaBitNetLinearForward: packed_weight must be rank-2");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
              "CudaBitNetLinearForward: packed_weight must use uint8 storage");
  TORCH_CHECK(x.size(-1) == layout.logical_in_features,
              "CudaBitNetLinearForward: input feature size mismatch");
  TORCH_CHECK(packed_weight.size(0) == layout.padded_out_features,
              "CudaBitNetLinearForward: packed_weight row count mismatch");
  TORCH_CHECK(packed_weight.size(1) == (layout.padded_in_features + 3) / 4,
              "CudaBitNetLinearForward: packed_weight column count mismatch");

  auto x_contig = x.contiguous();
  const auto rows = x_contig.numel() / layout.logical_in_features;
  const auto compute_dtype = x.scalar_type();
  auto x_2d = x_contig.view({rows, layout.logical_in_features});
  const auto plan = ResolvePlan(x_2d, layout);
  auto packed_contig = packed_weight.contiguous();
  auto scale_contig = scale_values.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
  auto offsets_contig = segment_offsets.to(torch::TensorOptions().device(x.device()).dtype(torch::kInt32)).contiguous();

  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().dim() == 1, "CudaBitNetLinearForward: bias must be rank-1");
    TORCH_CHECK(bias.value().size(0) == layout.logical_out_features,
                "CudaBitNetLinearForward: bias size mismatch");
    bias_cast = bias.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  auto out_dtype_resolved = out_dtype.has_value() ? out_dtype.value() : compute_dtype;
  TORCH_CHECK(IsSupportedLinearDtype(out_dtype_resolved),
              "CudaBitNetLinearForward: out_dtype must use float32, float16, or bfloat16");

  auto out_2d = torch::empty(
      {rows, layout.logical_out_features},
      torch::TensorOptions().device(x.device()).dtype(compute_dtype));
  if (rows == 0 || layout.logical_out_features == 0) {
    std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
    out_sizes.back() = layout.logical_out_features;
    auto empty_out = out_2d.view(out_sizes);
    return out_dtype_resolved == compute_dtype ? empty_out : empty_out.to(out_dtype_resolved);
  }

  if (plan.kind == KernelKind::kDecodePersistent) {
    LaunchBitNetDecodeKernel(out_2d, x_2d, packed_contig, scale_contig, offsets_contig, bias_cast, layout, plan);
  } else if (plan.kind == KernelKind::kPrefillSplitK) {
    LaunchBitNetPrefillSplitKKernel(out_2d, x_2d, packed_contig, scale_contig, offsets_contig, bias_cast, layout, plan);
  } else {
    LaunchBitNetPrefillKernel(out_2d, x_2d, packed_contig, scale_contig, offsets_contig, bias_cast, layout, plan);
  }

  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = layout.logical_out_features;
  auto out = out_2d.view(out_sizes);
  return out_dtype_resolved == compute_dtype ? out : out.to(out_dtype_resolved);
}

torch::Tensor CudaBitNetLinearForwardComputePacked(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined() && packed_weight.defined() && scale_values.defined() && layout_header.defined() &&
                  segment_offsets.defined() && compute_packed_words.defined() && compute_row_scales.defined(),
              "CudaBitNetLinearForwardComputePacked: tensors must be defined");
  TORCH_CHECK(x.is_cuda() && compute_packed_words.is_cuda() && compute_row_scales.is_cuda(),
              "CudaBitNetLinearForwardComputePacked: x and compute-packed tensors must be CUDA");
  TORCH_CHECK(x.dim() >= 2, "CudaBitNetLinearForwardComputePacked: x must have rank >= 2");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetLinearForwardComputePacked: x must use float32, float16, or bfloat16");

  c10::cuda::CUDAGuard device_guard(x.device());

  const auto layout = ParseLayoutHeader(layout_header);
  ValidateSegmentOffsets(segment_offsets, layout);
  ValidateScaleValues(scale_values, layout);
  TORCH_CHECK(SupportsScaleGranularity(layout),
              "CudaBitNetLinearForwardComputePacked: unsupported scale granularity");
  TORCH_CHECK(x.size(-1) == layout.logical_in_features,
              "CudaBitNetLinearForwardComputePacked: input feature size mismatch");

  auto x_contig = x.contiguous();
  const auto rows = x_contig.numel() / layout.logical_in_features;
  const auto compute_dtype = x.scalar_type();
  auto x_2d = x_contig.view({rows, layout.logical_in_features});
  const auto plan = ResolvePlan(x_2d, layout);

  const int64_t expected_word_cols = (layout.padded_in_features + 15) / 16;
  const int64_t expected_decode_chunk_cols = (layout.logical_in_features + 31) / 32;
  const bool compute_layout_valid =
      compute_packed_words.dim() == 3 &&
      compute_packed_words.scalar_type() == torch::kInt32 &&
      compute_row_scales.dim() == 2 &&
      compute_row_scales.scalar_type() == torch::kFloat32 &&
      compute_packed_words.device() == x.device() &&
      compute_row_scales.device() == x.device() &&
      compute_packed_words.size(1) == expected_word_cols &&
      compute_packed_words.size(0) == compute_row_scales.size(0) &&
      compute_packed_words.size(2) == compute_row_scales.size(1) &&
      compute_row_scales.size(1) > 0 &&
      compute_row_scales.size(0) * compute_row_scales.size(1) >= layout.padded_out_features;
  const bool decode_layout_valid =
      decode_nz_masks.defined() &&
      decode_sign_masks.defined() &&
      decode_row_scales.defined() &&
      decode_nz_masks.dim() == 3 &&
      decode_sign_masks.dim() == 3 &&
      decode_row_scales.dim() == 2 &&
      decode_nz_masks.scalar_type() == torch::kInt32 &&
      decode_sign_masks.scalar_type() == torch::kInt32 &&
      decode_row_scales.scalar_type() == torch::kFloat32 &&
      decode_nz_masks.device() == x.device() &&
      decode_sign_masks.device() == x.device() &&
      decode_row_scales.device() == x.device() &&
      decode_nz_masks.sizes() == decode_sign_masks.sizes() &&
      decode_nz_masks.size(1) == expected_decode_chunk_cols &&
      decode_nz_masks.size(0) == decode_row_scales.size(0) &&
      decode_nz_masks.size(2) == decode_row_scales.size(1) &&
      decode_row_scales.size(1) > 0 &&
      decode_row_scales.size(0) * decode_row_scales.size(1) >= layout.padded_out_features;

  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().dim() == 1, "CudaBitNetLinearForwardComputePacked: bias must be rank-1");
    TORCH_CHECK(bias.value().size(0) == layout.logical_out_features,
                "CudaBitNetLinearForwardComputePacked: bias size mismatch");
    bias_cast = bias.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  auto out_dtype_resolved = out_dtype.has_value() ? out_dtype.value() : compute_dtype;
  TORCH_CHECK(IsSupportedLinearDtype(out_dtype_resolved),
              "CudaBitNetLinearForwardComputePacked: out_dtype must use float32, float16, or bfloat16");

  auto out_2d = torch::empty(
      {rows, layout.logical_out_features},
      torch::TensorOptions().device(x.device()).dtype(compute_dtype));
  if (rows == 0 || layout.logical_out_features == 0) {
    std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
    out_sizes.back() = layout.logical_out_features;
    auto empty_out = out_2d.view(out_sizes);
    return out_dtype_resolved == compute_dtype ? empty_out : empty_out.to(out_dtype_resolved);
  }

  if (plan.kind == KernelKind::kDecodePersistent && rows == 1 && decode_layout_valid) {
    auto decode_nz_masks_contig = decode_nz_masks.contiguous();
    auto decode_sign_masks_contig = decode_sign_masks.contiguous();
    auto decode_scales_contig =
        decode_row_scales.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
    LaunchBitNetDecodeKernelBitplaneRow1(
        out_2d,
        x_2d,
        decode_nz_masks_contig,
        decode_sign_masks_contig,
        decode_scales_contig,
        bias_cast,
        layout,
        plan);
  } else {
    if (!compute_layout_valid) {
      return CudaBitNetLinearForward(
          x_contig,
          packed_weight,
          scale_values,
          layout_header,
          segment_offsets,
          bias,
          out_dtype);
    }
    auto compute_words_contig = compute_packed_words.contiguous();
    auto compute_scales_contig =
        compute_row_scales.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
    if (plan.kind == KernelKind::kDecodePersistent) {
      LaunchBitNetDecodeKernelComputePacked(
          out_2d,
          x_2d,
          compute_words_contig,
          compute_scales_contig,
          bias_cast,
          layout,
          plan);
    } else if (plan.kind == KernelKind::kPrefillSplitK) {
      LaunchBitNetPrefillSplitKKernelComputePacked(
          out_2d,
          x_2d,
          compute_words_contig,
          compute_scales_contig,
          bias_cast,
          layout,
          plan);
    } else {
      LaunchBitNetPrefillKernelComputePacked(
          out_2d,
          x_2d,
          compute_words_contig,
          compute_scales_contig,
          bias_cast,
          layout,
          plan);
    }
  }

  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = layout.logical_out_features;
  auto out = out_2d.view(out_sizes);
  return out_dtype_resolved == compute_dtype ? out : out.to(out_dtype_resolved);
}

torch::Tensor CudaBitNetRmsNormLinearForwardDecodeRows(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& rms_weight,
    double eps,
    const torch::Tensor& layout_header,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined() && layout_header.defined(),
              "CudaBitNetRmsNormLinearForwardDecodeRows: x and layout_header must be defined");
  TORCH_CHECK(x.is_cuda(), "CudaBitNetRmsNormLinearForwardDecodeRows: x must be CUDA");
  TORCH_CHECK(x.dim() >= 2, "CudaBitNetRmsNormLinearForwardDecodeRows: x must have rank >= 2");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetRmsNormLinearForwardDecodeRows: x must use float32, float16, or bfloat16");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0,
              "CudaBitNetRmsNormLinearForwardDecodeRows: eps must be positive and finite");

  c10::cuda::CUDAGuard device_guard(x.device());

  const auto layout = ParseLayoutHeader(layout_header);
  TORCH_CHECK(x.size(-1) == layout.logical_in_features,
              "CudaBitNetRmsNormLinearForwardDecodeRows: input feature size mismatch");

  auto x_contig = x.contiguous();
  const auto rows = x_contig.numel() / layout.logical_in_features;
  TORCH_CHECK(rows > 0 && rows <= 8,
              "CudaBitNetRmsNormLinearForwardDecodeRows only supports decode rows in [1, 8]");
  const auto compute_dtype = x.scalar_type();
  auto x_2d = x_contig.view({rows, layout.logical_in_features});
  const auto plan = ResolvePlan(x_2d, layout);
  TORCH_CHECK(plan.kind == KernelKind::kDecodePersistent,
              "CudaBitNetRmsNormLinearForwardDecodeRows only supports decode-persistent plans");

  const int64_t expected_word_cols = (layout.padded_in_features + 15) / 16;
  const int64_t expected_decode_chunk_cols = (layout.logical_in_features + 31) / 32;
  const bool compute_layout_valid =
      compute_packed_words.defined() &&
      compute_row_scales.defined() &&
      compute_packed_words.dim() == 3 &&
      compute_packed_words.scalar_type() == torch::kInt32 &&
      compute_row_scales.dim() == 2 &&
      compute_row_scales.scalar_type() == torch::kFloat32 &&
      compute_packed_words.device() == x.device() &&
      compute_row_scales.device() == x.device() &&
      compute_packed_words.size(1) == expected_word_cols &&
      compute_packed_words.size(0) == compute_row_scales.size(0) &&
      compute_packed_words.size(2) == compute_row_scales.size(1) &&
      compute_row_scales.size(1) > 0 &&
      compute_row_scales.size(0) * compute_row_scales.size(1) >= layout.padded_out_features;
  const bool decode_layout_valid =
      decode_nz_masks.defined() &&
      decode_sign_masks.defined() &&
      decode_row_scales.defined() &&
      decode_nz_masks.dim() == 3 &&
      decode_sign_masks.dim() == 3 &&
      decode_row_scales.dim() == 2 &&
      decode_nz_masks.scalar_type() == torch::kInt32 &&
      decode_sign_masks.scalar_type() == torch::kInt32 &&
      decode_row_scales.scalar_type() == torch::kFloat32 &&
      decode_nz_masks.device() == x.device() &&
      decode_sign_masks.device() == x.device() &&
      decode_row_scales.device() == x.device() &&
      decode_nz_masks.sizes() == decode_sign_masks.sizes() &&
      decode_nz_masks.size(1) == expected_decode_chunk_cols &&
      decode_nz_masks.size(0) == decode_row_scales.size(0) &&
      decode_nz_masks.size(2) == decode_row_scales.size(1) &&
      decode_row_scales.size(1) > 0 &&
      decode_row_scales.size(0) * decode_row_scales.size(1) >= layout.padded_out_features;

  c10::optional<torch::Tensor> rms_weight_cast = c10::nullopt;
  if (rms_weight.has_value() && rms_weight.value().defined()) {
    TORCH_CHECK(rms_weight.value().dim() == 1,
                "CudaBitNetRmsNormLinearForwardDecodeRows: rms_weight must be rank-1");
    TORCH_CHECK(rms_weight.value().size(0) == layout.logical_in_features,
                "CudaBitNetRmsNormLinearForwardDecodeRows: rms_weight size mismatch");
    rms_weight_cast =
        rms_weight.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().dim() == 1, "CudaBitNetRmsNormLinearForwardDecodeRows: bias must be rank-1");
    TORCH_CHECK(bias.value().size(0) == layout.logical_out_features,
                "CudaBitNetRmsNormLinearForwardDecodeRows: bias size mismatch");
    bias_cast = bias.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  auto out_dtype_resolved = out_dtype.has_value() ? out_dtype.value() : compute_dtype;
  TORCH_CHECK(IsSupportedLinearDtype(out_dtype_resolved),
              "CudaBitNetRmsNormLinearForwardDecodeRows: out_dtype must use float32, float16, or bfloat16");

  auto out_2d = torch::empty(
      {rows, layout.logical_out_features},
      torch::TensorOptions().device(x.device()).dtype(compute_dtype));
  if (rows == 1 && decode_layout_valid) {
    auto decode_nz_masks_contig = decode_nz_masks.contiguous();
    auto decode_sign_masks_contig = decode_sign_masks.contiguous();
    auto decode_scales_contig =
        decode_row_scales.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
    LaunchBitNetDecodeKernelBitplaneRow1RmsNorm(
        out_2d,
        x_2d,
        rms_weight_cast,
        eps,
        decode_nz_masks_contig,
        decode_sign_masks_contig,
        decode_scales_contig,
        bias_cast,
        layout,
        plan);
  } else {
    TORCH_CHECK(compute_layout_valid,
                "CudaBitNetRmsNormLinearForwardDecodeRows: invalid compute-packed decode layout");
    auto compute_words_contig = compute_packed_words.contiguous();
    auto compute_scales_contig =
        compute_row_scales.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
    LaunchBitNetDecodeKernelComputePackedRmsNorm(
        out_2d,
        x_2d,
        rms_weight_cast,
        eps,
        compute_words_contig,
        compute_scales_contig,
        bias_cast,
        layout,
        plan);
  }

  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = layout.logical_out_features;
  auto out = out_2d.view(out_sizes);
  return out_dtype_resolved == compute_dtype ? out : out.to(out_dtype_resolved);
}

std::vector<torch::Tensor> CudaBitNetAddRmsNormLinearForwardDecodeRows(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& rms_weight,
    double residual_scale,
    double eps,
    const torch::Tensor& layout_header,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined() && update.defined() && layout_header.defined(),
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: x, update, and layout_header must be defined");
  TORCH_CHECK(x.is_cuda() && update.is_cuda(),
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: x and update must be CUDA");
  TORCH_CHECK(x.dim() >= 2 && update.dim() >= 2,
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: x and update must have rank >= 2");
  TORCH_CHECK(x.sizes() == update.sizes(),
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: x and update shape mismatch");
  TORCH_CHECK(x.scalar_type() == update.scalar_type(),
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: x and update dtype mismatch");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: x must use float32, float16, or bfloat16");
  TORCH_CHECK(std::isfinite(residual_scale),
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: residual_scale must be finite");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0,
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: eps must be positive and finite");

  c10::cuda::CUDAGuard device_guard(x.device());

  const auto layout = ParseLayoutHeader(layout_header);
  TORCH_CHECK(x.size(-1) == layout.logical_in_features,
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: input feature size mismatch");

  auto x_contig = x.contiguous();
  auto update_contig = update.contiguous();
  const auto rows = x_contig.numel() / layout.logical_in_features;
  TORCH_CHECK(rows > 0 && rows <= 8,
              "CudaBitNetAddRmsNormLinearForwardDecodeRows only supports decode rows in [1, 8]");
  const auto compute_dtype = x.scalar_type();
  auto x_2d = x_contig.view({rows, layout.logical_in_features});
  auto update_2d = update_contig.view({rows, layout.logical_in_features});
  const auto plan = ResolvePlan(x_2d, layout);
  TORCH_CHECK(plan.kind == KernelKind::kDecodePersistent,
              "CudaBitNetAddRmsNormLinearForwardDecodeRows only supports decode-persistent plans");

  const int64_t expected_word_cols = (layout.padded_in_features + 15) / 16;
  const int64_t expected_decode_chunk_cols = (layout.logical_in_features + 31) / 32;
  const bool compute_layout_valid =
      compute_packed_words.defined() &&
      compute_row_scales.defined() &&
      compute_packed_words.dim() == 3 &&
      compute_packed_words.scalar_type() == torch::kInt32 &&
      compute_row_scales.dim() == 2 &&
      compute_row_scales.scalar_type() == torch::kFloat32 &&
      compute_packed_words.device() == x.device() &&
      compute_row_scales.device() == x.device() &&
      compute_packed_words.size(1) == expected_word_cols &&
      compute_packed_words.size(0) == compute_row_scales.size(0) &&
      compute_packed_words.size(2) == compute_row_scales.size(1) &&
      compute_row_scales.size(1) > 0 &&
      compute_row_scales.size(0) * compute_row_scales.size(1) >= layout.padded_out_features;
  const bool decode_layout_valid =
      decode_nz_masks.defined() &&
      decode_sign_masks.defined() &&
      decode_row_scales.defined() &&
      decode_nz_masks.dim() == 3 &&
      decode_sign_masks.dim() == 3 &&
      decode_row_scales.dim() == 2 &&
      decode_nz_masks.scalar_type() == torch::kInt32 &&
      decode_sign_masks.scalar_type() == torch::kInt32 &&
      decode_row_scales.scalar_type() == torch::kFloat32 &&
      decode_nz_masks.device() == x.device() &&
      decode_sign_masks.device() == x.device() &&
      decode_row_scales.device() == x.device() &&
      decode_nz_masks.sizes() == decode_sign_masks.sizes() &&
      decode_nz_masks.size(1) == expected_decode_chunk_cols &&
      decode_nz_masks.size(0) == decode_row_scales.size(0) &&
      decode_nz_masks.size(2) == decode_row_scales.size(1) &&
      decode_row_scales.size(1) > 0 &&
      decode_row_scales.size(0) * decode_row_scales.size(1) >= layout.padded_out_features;

  c10::optional<torch::Tensor> rms_weight_cast = c10::nullopt;
  if (rms_weight.has_value() && rms_weight.value().defined()) {
    TORCH_CHECK(rms_weight.value().dim() == 1,
                "CudaBitNetAddRmsNormLinearForwardDecodeRows: rms_weight must be rank-1");
    TORCH_CHECK(rms_weight.value().size(0) == layout.logical_in_features,
                "CudaBitNetAddRmsNormLinearForwardDecodeRows: rms_weight size mismatch");
    rms_weight_cast =
        rms_weight.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().dim() == 1,
                "CudaBitNetAddRmsNormLinearForwardDecodeRows: bias must be rank-1");
    TORCH_CHECK(bias.value().size(0) == layout.logical_out_features,
                "CudaBitNetAddRmsNormLinearForwardDecodeRows: bias size mismatch");
    bias_cast = bias.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  auto out_dtype_resolved = out_dtype.has_value() ? out_dtype.value() : compute_dtype;
  TORCH_CHECK(IsSupportedLinearDtype(out_dtype_resolved),
              "CudaBitNetAddRmsNormLinearForwardDecodeRows: out_dtype must use float32, float16, or bfloat16");

  auto combined_2d = torch::empty(
      {rows, layout.logical_in_features},
      torch::TensorOptions().device(x.device()).dtype(compute_dtype));
  auto out_2d = torch::empty(
      {rows, layout.logical_out_features},
      torch::TensorOptions().device(x.device()).dtype(compute_dtype));
  if (rows == 1 && decode_layout_valid) {
    auto decode_nz_masks_contig = decode_nz_masks.contiguous();
    auto decode_sign_masks_contig = decode_sign_masks.contiguous();
    auto decode_scales_contig =
        decode_row_scales.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
    LaunchBitNetDecodeKernelBitplaneRow1AddRmsNorm(
        combined_2d,
        out_2d,
        x_2d,
        update_2d,
        rms_weight_cast,
        residual_scale,
        eps,
        decode_nz_masks_contig,
        decode_sign_masks_contig,
        decode_scales_contig,
        bias_cast,
        layout,
        plan);
  } else {
    TORCH_CHECK(compute_layout_valid,
                "CudaBitNetAddRmsNormLinearForwardDecodeRows: invalid compute-packed decode layout");
    auto compute_words_contig = compute_packed_words.contiguous();
    auto compute_scales_contig =
        compute_row_scales.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
    LaunchBitNetDecodeKernelComputePackedAddRmsNorm(
        combined_2d,
        out_2d,
        x_2d,
        update_2d,
        rms_weight_cast,
        residual_scale,
        eps,
        compute_words_contig,
        compute_scales_contig,
        bias_cast,
        layout,
        plan);
  }

  std::vector<int64_t> combined_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  auto combined = combined_2d.view(combined_sizes);
  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = layout.logical_out_features;
  auto out = out_2d.view(out_sizes);
  if (out_dtype_resolved != compute_dtype) {
    out = out.to(out_dtype_resolved);
  }
  return {combined, out};
}

torch::Tensor CudaBitNetRmsNormLinearForwardRow1(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& rms_weight,
    double eps,
    const torch::Tensor& layout_header,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined() && layout_header.defined() && decode_nz_masks.defined() &&
                  decode_sign_masks.defined() && decode_row_scales.defined(),
              "CudaBitNetRmsNormLinearForwardRow1: tensors must be defined");
  TORCH_CHECK(x.is_cuda() && decode_nz_masks.is_cuda() && decode_sign_masks.is_cuda() && decode_row_scales.is_cuda(),
              "CudaBitNetRmsNormLinearForwardRow1: x and decode backend tensors must be CUDA");
  TORCH_CHECK(x.dim() >= 2, "CudaBitNetRmsNormLinearForwardRow1: x must have rank >= 2");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetRmsNormLinearForwardRow1: x must use float32, float16, or bfloat16");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0,
              "CudaBitNetRmsNormLinearForwardRow1: eps must be positive and finite");

  c10::cuda::CUDAGuard device_guard(x.device());

  const auto layout = ParseLayoutHeader(layout_header);
  TORCH_CHECK(x.size(-1) == layout.logical_in_features,
              "CudaBitNetRmsNormLinearForwardRow1: input feature size mismatch");

  auto x_contig = x.contiguous();
  const auto rows = x_contig.numel() / layout.logical_in_features;
  TORCH_CHECK(rows == 1, "CudaBitNetRmsNormLinearForwardRow1 only supports a single decode row");
  const auto compute_dtype = x.scalar_type();
  auto x_2d = x_contig.view({rows, layout.logical_in_features});
  const auto plan = ResolvePlan(x_2d, layout);
  TORCH_CHECK(plan.kind == KernelKind::kDecodePersistent,
              "CudaBitNetRmsNormLinearForwardRow1 only supports decode-persistent plans");

  const int64_t expected_decode_chunk_cols = (layout.logical_in_features + 31) / 32;
  const bool decode_layout_valid =
      decode_nz_masks.dim() == 3 &&
      decode_sign_masks.dim() == 3 &&
      decode_row_scales.dim() == 2 &&
      decode_nz_masks.scalar_type() == torch::kInt32 &&
      decode_sign_masks.scalar_type() == torch::kInt32 &&
      decode_row_scales.scalar_type() == torch::kFloat32 &&
      decode_nz_masks.device() == x.device() &&
      decode_sign_masks.device() == x.device() &&
      decode_row_scales.device() == x.device() &&
      decode_nz_masks.sizes() == decode_sign_masks.sizes() &&
      decode_nz_masks.size(1) == expected_decode_chunk_cols &&
      decode_nz_masks.size(0) == decode_row_scales.size(0) &&
      decode_nz_masks.size(2) == decode_row_scales.size(1) &&
      decode_row_scales.size(1) > 0 &&
      decode_row_scales.size(0) * decode_row_scales.size(1) >= layout.padded_out_features;
  TORCH_CHECK(decode_layout_valid,
              "CudaBitNetRmsNormLinearForwardRow1: invalid decode backend layout");

  c10::optional<torch::Tensor> rms_weight_cast = c10::nullopt;
  if (rms_weight.has_value() && rms_weight.value().defined()) {
    TORCH_CHECK(rms_weight.value().dim() == 1,
                "CudaBitNetRmsNormLinearForwardRow1: rms_weight must be rank-1");
    TORCH_CHECK(rms_weight.value().size(0) == layout.logical_in_features,
                "CudaBitNetRmsNormLinearForwardRow1: rms_weight size mismatch");
    rms_weight_cast =
        rms_weight.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().dim() == 1, "CudaBitNetRmsNormLinearForwardRow1: bias must be rank-1");
    TORCH_CHECK(bias.value().size(0) == layout.logical_out_features,
                "CudaBitNetRmsNormLinearForwardRow1: bias size mismatch");
    bias_cast = bias.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  auto out_dtype_resolved = out_dtype.has_value() ? out_dtype.value() : compute_dtype;
  TORCH_CHECK(IsSupportedLinearDtype(out_dtype_resolved),
              "CudaBitNetRmsNormLinearForwardRow1: out_dtype must use float32, float16, or bfloat16");

  auto out_2d = torch::empty(
      {rows, layout.logical_out_features},
      torch::TensorOptions().device(x.device()).dtype(compute_dtype));
  auto decode_nz_masks_contig = decode_nz_masks.contiguous();
  auto decode_sign_masks_contig = decode_sign_masks.contiguous();
  auto decode_scales_contig =
      decode_row_scales.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
  LaunchBitNetDecodeKernelBitplaneRow1RmsNorm(
      out_2d,
      x_2d,
      rms_weight_cast,
      eps,
      decode_nz_masks_contig,
      decode_sign_masks_contig,
      decode_scales_contig,
      bias_cast,
      layout,
      plan);

  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = layout.logical_out_features;
  auto out = out_2d.view(out_sizes);
  return out_dtype_resolved == compute_dtype ? out : out.to(out_dtype_resolved);
}

std::vector<torch::Tensor> CudaBitNetAddRmsNormLinearForwardRow1(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& rms_weight,
    double residual_scale,
    double eps,
    const torch::Tensor& layout_header,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined() && update.defined() && layout_header.defined() && decode_nz_masks.defined() &&
                  decode_sign_masks.defined() && decode_row_scales.defined(),
              "CudaBitNetAddRmsNormLinearForwardRow1: tensors must be defined");
  TORCH_CHECK(x.is_cuda() && update.is_cuda() && decode_nz_masks.is_cuda() &&
                  decode_sign_masks.is_cuda() && decode_row_scales.is_cuda(),
              "CudaBitNetAddRmsNormLinearForwardRow1: tensors must be CUDA");
  TORCH_CHECK(x.dim() >= 2 && update.dim() >= 2,
              "CudaBitNetAddRmsNormLinearForwardRow1: x and update must have rank >= 2");
  TORCH_CHECK(x.sizes() == update.sizes(),
              "CudaBitNetAddRmsNormLinearForwardRow1: x and update shape mismatch");
  TORCH_CHECK(x.scalar_type() == update.scalar_type(),
              "CudaBitNetAddRmsNormLinearForwardRow1: x and update dtype mismatch");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetAddRmsNormLinearForwardRow1: x must use float32, float16, or bfloat16");
  TORCH_CHECK(std::isfinite(residual_scale),
              "CudaBitNetAddRmsNormLinearForwardRow1: residual_scale must be finite");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0,
              "CudaBitNetAddRmsNormLinearForwardRow1: eps must be positive and finite");

  c10::cuda::CUDAGuard device_guard(x.device());

  const auto layout = ParseLayoutHeader(layout_header);
  TORCH_CHECK(x.size(-1) == layout.logical_in_features,
              "CudaBitNetAddRmsNormLinearForwardRow1: input feature size mismatch");

  auto x_contig = x.contiguous();
  auto update_contig = update.contiguous();
  const auto rows = x_contig.numel() / layout.logical_in_features;
  TORCH_CHECK(rows == 1, "CudaBitNetAddRmsNormLinearForwardRow1 only supports a single decode row");
  const auto compute_dtype = x.scalar_type();
  auto x_2d = x_contig.view({rows, layout.logical_in_features});
  auto update_2d = update_contig.view({rows, layout.logical_in_features});
  const auto plan = ResolvePlan(x_2d, layout);
  TORCH_CHECK(plan.kind == KernelKind::kDecodePersistent,
              "CudaBitNetAddRmsNormLinearForwardRow1 only supports decode-persistent plans");

  const int64_t expected_decode_chunk_cols = (layout.logical_in_features + 31) / 32;
  const bool decode_layout_valid =
      decode_nz_masks.dim() == 3 &&
      decode_sign_masks.dim() == 3 &&
      decode_row_scales.dim() == 2 &&
      decode_nz_masks.scalar_type() == torch::kInt32 &&
      decode_sign_masks.scalar_type() == torch::kInt32 &&
      decode_row_scales.scalar_type() == torch::kFloat32 &&
      decode_nz_masks.device() == x.device() &&
      decode_sign_masks.device() == x.device() &&
      decode_row_scales.device() == x.device() &&
      decode_nz_masks.sizes() == decode_sign_masks.sizes() &&
      decode_nz_masks.size(1) == expected_decode_chunk_cols &&
      decode_nz_masks.size(0) == decode_row_scales.size(0) &&
      decode_nz_masks.size(2) == decode_row_scales.size(1) &&
      decode_row_scales.size(1) > 0 &&
      decode_row_scales.size(0) * decode_row_scales.size(1) >= layout.padded_out_features;
  TORCH_CHECK(decode_layout_valid,
              "CudaBitNetAddRmsNormLinearForwardRow1: invalid decode backend layout");

  c10::optional<torch::Tensor> rms_weight_cast = c10::nullopt;
  if (rms_weight.has_value() && rms_weight.value().defined()) {
    TORCH_CHECK(rms_weight.value().dim() == 1,
                "CudaBitNetAddRmsNormLinearForwardRow1: rms_weight must be rank-1");
    TORCH_CHECK(rms_weight.value().size(0) == layout.logical_in_features,
                "CudaBitNetAddRmsNormLinearForwardRow1: rms_weight size mismatch");
    rms_weight_cast =
        rms_weight.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().dim() == 1,
                "CudaBitNetAddRmsNormLinearForwardRow1: bias must be rank-1");
    TORCH_CHECK(bias.value().size(0) == layout.logical_out_features,
                "CudaBitNetAddRmsNormLinearForwardRow1: bias size mismatch");
    bias_cast = bias.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  auto out_dtype_resolved = out_dtype.has_value() ? out_dtype.value() : compute_dtype;
  TORCH_CHECK(IsSupportedLinearDtype(out_dtype_resolved),
              "CudaBitNetAddRmsNormLinearForwardRow1: out_dtype must use float32, float16, or bfloat16");

  auto combined_2d = torch::empty(
      {rows, layout.logical_in_features},
      torch::TensorOptions().device(x.device()).dtype(compute_dtype));
  auto out_2d = torch::empty(
      {rows, layout.logical_out_features},
      torch::TensorOptions().device(x.device()).dtype(compute_dtype));
  auto decode_nz_masks_contig = decode_nz_masks.contiguous();
  auto decode_sign_masks_contig = decode_sign_masks.contiguous();
  auto decode_scales_contig =
      decode_row_scales.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
  LaunchBitNetDecodeKernelBitplaneRow1AddRmsNorm(
      combined_2d,
      out_2d,
      x_2d,
      update_2d,
      rms_weight_cast,
      residual_scale,
      eps,
      decode_nz_masks_contig,
      decode_sign_masks_contig,
      decode_scales_contig,
      bias_cast,
      layout,
      plan);

  std::vector<int64_t> combined_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  auto combined = combined_2d.view(combined_sizes);
  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = layout.logical_out_features;
  auto out = out_2d.view(out_sizes);
  if (out_dtype_resolved != compute_dtype) {
    out = out.to(out_dtype_resolved);
  }
  return {combined, out};
}

torch::Tensor CudaBitNetLinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined() && packed_weight.defined() && scale_values.defined() && layout_header.defined() &&
                  segment_offsets.defined(),
              "CudaBitNetLinearFromFloatForward: tensors must be defined");
  TORCH_CHECK(x.is_cuda() && packed_weight.is_cuda() && scale_values.is_cuda() && layout_header.is_cuda() &&
                  segment_offsets.is_cuda(),
              "CudaBitNetLinearFromFloatForward: tensors must be CUDA");
  TORCH_CHECK(x.dim() >= 2, "CudaBitNetLinearFromFloatForward: x must have rank >= 2");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "CudaBitNetLinearFromFloatForward: x must use float32, float16, or bfloat16");

  c10::cuda::CUDAGuard device_guard(x.device());

  const auto layout = ParseLayoutHeader(layout_header);
  ValidateSegmentOffsets(segment_offsets, layout);
  ValidateScaleValues(scale_values, layout);
  TORCH_CHECK(SupportsScaleGranularity(layout),
              "CudaBitNetLinearFromFloatForward: unsupported scale granularity");
  TORCH_CHECK(packed_weight.dim() == 2, "CudaBitNetLinearFromFloatForward: packed_weight must be rank-2");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
              "CudaBitNetLinearFromFloatForward: packed_weight must use uint8 storage");
  TORCH_CHECK(x.size(-1) == layout.logical_in_features,
              "CudaBitNetLinearFromFloatForward: input feature size mismatch");
  TORCH_CHECK(packed_weight.size(0) == layout.padded_out_features,
              "CudaBitNetLinearFromFloatForward: packed_weight row count mismatch");
  TORCH_CHECK(packed_weight.size(1) == (layout.padded_in_features + 3) / 4,
              "CudaBitNetLinearFromFloatForward: packed_weight column count mismatch");

  auto x_contig = x.contiguous();
  const auto rows = x_contig.numel() / layout.logical_in_features;
  const auto compute_dtype = x.scalar_type();
  auto x_2d = x_contig.view({rows, layout.logical_in_features});
  const auto plan = ResolvePlan(x_2d, layout);
  auto packed_contig = packed_weight.contiguous();
  auto scale_contig = scale_values.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous();
  auto offsets_contig = segment_offsets.to(torch::TensorOptions().device(x.device()).dtype(torch::kInt32)).contiguous();

  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().dim() == 1, "CudaBitNetLinearFromFloatForward: bias must be rank-1");
    TORCH_CHECK(bias.value().size(0) == layout.logical_out_features,
                "CudaBitNetLinearFromFloatForward: bias size mismatch");
    bias_cast = bias.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).contiguous();
  }

  c10::optional<torch::Tensor> pre_scale_cast = c10::nullopt;
  if (pre_scale.has_value() && pre_scale.value().defined() && pre_scale.value().numel() > 0) {
    auto scale = pre_scale.value().to(torch::TensorOptions().device(x.device()).dtype(compute_dtype)).reshape({-1}).contiguous();
    TORCH_CHECK(scale.numel() == layout.logical_in_features,
                "CudaBitNetLinearFromFloatForward: pre_scale must match x last dimension");
    pre_scale_cast = scale;
  }

  auto out_dtype_resolved = out_dtype.has_value() ? out_dtype.value() : compute_dtype;
  TORCH_CHECK(IsSupportedLinearDtype(out_dtype_resolved),
              "CudaBitNetLinearFromFloatForward: out_dtype must use float32, float16, or bfloat16");

  auto out_2d = torch::empty(
      {rows, layout.logical_out_features},
      torch::TensorOptions().device(x.device()).dtype(compute_dtype));
  if (rows == 0 || layout.logical_out_features == 0) {
    std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
    out_sizes.back() = layout.logical_out_features;
    auto empty_out = out_2d.view(out_sizes);
    return out_dtype_resolved == compute_dtype ? empty_out : empty_out.to(out_dtype_resolved);
  }

  const auto mode = NormalizeModeName(act_quant_mode);
  const auto method = NormalizeModeName(act_quant_method);
  const bool needs_prescale = pre_scale_cast.has_value() && pre_scale_cast.value().defined();

  if (mode == "none" && !needs_prescale) {
    return CudaBitNetLinearForward(
        x_contig,
        packed_contig,
        scale_contig,
        layout_header,
        offsets_contig,
        bias_cast,
        out_dtype);
  }

  c10::optional<torch::Tensor> input_scale = c10::nullopt;
  if (mode == "static_int8" && act_quant_bits >= 2 && act_quant_bits <= 8 && act_scale.has_value() &&
      act_scale.value().defined()) {
    auto scale = act_scale.value().to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).reshape({-1}).contiguous();
    TORCH_CHECK(scale.numel() == 1 || scale.numel() == rows,
                "CudaBitNetLinearFromFloatForward: act_scale must have 1 or rows elements");
    input_scale = scale;
  } else if (mode == "dynamic_int8" && act_quant_bits >= 2 && act_quant_bits <= 8) {
    TORCH_CHECK(
        method.empty() || method == "absmax" || method == "mse",
        "CudaBitNetLinearFromFloatForward: unsupported dynamic activation calibration method: ",
        act_quant_method);
    input_scale = CudaBitNetCalibrateInputScaleForward(x_contig, pre_scale_cast, act_quant_bits);
  }

  if (input_scale.has_value()) {
    if (plan.kind == KernelKind::kDecodePersistent) {
      LaunchBitNetDecodeKernelStaticInput(
          out_2d,
          x_2d,
          pre_scale_cast,
          input_scale.value(),
          act_quant_bits,
          packed_contig,
          scale_contig,
          offsets_contig,
          bias_cast,
          layout,
          plan);
    } else if (plan.kind == KernelKind::kPrefillSplitK) {
      LaunchBitNetPrefillSplitKKernelStaticInput(
          out_2d,
          x_2d,
          pre_scale_cast,
          input_scale.value(),
          act_quant_bits,
          packed_contig,
          scale_contig,
          offsets_contig,
          bias_cast,
          layout,
          plan);
    } else {
      LaunchBitNetPrefillKernelStaticInput(
          out_2d,
          x_2d,
          pre_scale_cast,
          input_scale.value(),
          act_quant_bits,
          packed_contig,
          scale_contig,
          offsets_contig,
          bias_cast,
          layout,
          plan);
    }

    std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
    out_sizes.back() = layout.logical_out_features;
    auto out = out_2d.view(out_sizes);
    return out_dtype_resolved == compute_dtype ? out : out.to(out_dtype_resolved);
  }

  TORCH_CHECK(
      HasCudaBitNetInputFrontendKernel(),
      "CudaBitNetLinearFromFloatForward: unsupported BitNet frontend mode without CUDA transform kernel");
  TORCH_CHECK(
      mode == "none" || mode == "static_int8",
      "CudaBitNetLinearFromFloatForward: unsupported activation quant mode: ",
      act_quant_mode);
  auto transformed = CudaBitNetTransformInputForward(
      x_contig,
      pre_scale_cast,
      mode,
      method,
      act_quant_bits,
      act_scale);
  return CudaBitNetLinearForward(
      transformed,
      packed_contig,
      scale_contig,
      layout_header,
      offsets_contig,
      bias_cast,
      out_dtype);
}

}  // namespace t10::bitnet
