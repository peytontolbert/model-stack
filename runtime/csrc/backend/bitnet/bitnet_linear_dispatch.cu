#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "bitnet_common.cuh"

#include <vector>

namespace t10::bitnet {

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
  TORCH_CHECK(SupportsScaleGranularity(layout),
              "CudaBitNetLinearForward: unsupported scale granularity");
  TORCH_CHECK(packed_weight.dim() == 2, "CudaBitNetLinearForward: packed_weight must be rank-2");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
              "CudaBitNetLinearForward: packed_weight must use uint8 storage");
  TORCH_CHECK(scale_values.dim() == 1, "CudaBitNetLinearForward: scale_values must be rank-1");
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

}  // namespace t10::bitnet
