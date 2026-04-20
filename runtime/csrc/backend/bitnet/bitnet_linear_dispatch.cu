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
