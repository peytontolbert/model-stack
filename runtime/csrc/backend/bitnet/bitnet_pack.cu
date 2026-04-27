#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include "bitnet_common.cuh"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace t10::bitnet {
namespace {

constexpr int kPackThreads = 256;

template <typename scalar_t>
__global__ void bitnet_pack_weight_kernel(
    const scalar_t* __restrict__ weight,
    uint8_t* __restrict__ packed_weight,
    float scale,
    int64_t logical_out,
    int64_t logical_in,
    int64_t padded_out,
    int64_t packed_cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = padded_out * packed_cols;
  if (idx >= total) {
    return;
  }
  const int64_t out_idx = idx / packed_cols;
  const int64_t packed_col = idx % packed_cols;
  const int64_t in_base = packed_col * 4;
  uint8_t packed = 0;
  #pragma unroll
  for (int offset = 0; offset < 4; ++offset) {
    uint8_t code = 1;
    const int64_t in_idx = in_base + offset;
    if (out_idx < logical_out && in_idx < logical_in) {
      const float normalized = static_cast<float>(weight[out_idx * logical_in + in_idx]) / scale;
      const int rounded = static_cast<int>(nearbyintf(normalized));
      const int clamped = max(-1, min(1, rounded));
      code = static_cast<uint8_t>(clamped + 1);
    }
    packed |= static_cast<uint8_t>(code << (offset * 2));
  }
  packed_weight[idx] = packed;
}

template <typename scalar_t>
__global__ void bitnet_runtime_row_quantize_kernel(
    const scalar_t* __restrict__ weight,
    int8_t* __restrict__ qweight,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols,
    float eps) {
  extern __shared__ float scratch[];
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  if (row >= rows) {
    return;
  }

  float local_sum = 0.0f;
  const int64_t row_offset = row * cols;
  for (int64_t col = thread; col < cols; col += blockDim.x) {
    local_sum += fabsf(static_cast<float>(weight[row_offset + col]));
  }
  scratch[thread] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (thread < stride) {
      scratch[thread] += scratch[thread + stride];
    }
    __syncthreads();
  }

  const float scale = fmaxf(scratch[0] / static_cast<float>(cols), eps);
  if (thread == 0) {
    row_scale[row] = scale;
  }
  __syncthreads();

  for (int64_t col = thread; col < cols; col += blockDim.x) {
    const float normalized = static_cast<float>(weight[row_offset + col]) / scale;
    const int rounded = static_cast<int>(nearbyintf(normalized));
    const int clamped = max(-1, min(1, rounded));
    qweight[row_offset + col] = static_cast<int8_t>(clamped);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CudaPackBitNetWeightForward(
    const torch::Tensor& weight) {
  TORCH_CHECK(weight.defined(), "CudaPackBitNetWeightForward: weight must be defined");
  TORCH_CHECK(weight.is_cuda(), "CudaPackBitNetWeightForward: weight must be CUDA");
  TORCH_CHECK(weight.dim() == 2, "CudaPackBitNetWeightForward: weight must be rank-2");
  TORCH_CHECK(IsSupportedLinearDtype(weight.scalar_type()),
              "CudaPackBitNetWeightForward: unsupported weight dtype");

  c10::cuda::CUDAGuard device_guard(weight.device());
  auto weight_contig = weight.contiguous();
  const auto logical_out = weight_contig.size(0);
  const auto logical_in = weight_contig.size(1);
  const auto padded_out = ((logical_out + 15) / 16) * 16;
  const auto padded_in = ((logical_in + 31) / 32) * 32;
  const auto packed_cols = (padded_in + 3) / 4;

  const auto max_abs = weight_contig.abs().amax().item<float>();
  const float scale = std::max(max_abs, 1.0e-8f);

  auto packed_weight = torch::empty(
      {padded_out, packed_cols},
      torch::TensorOptions().device(weight.device()).dtype(torch::kUInt8));
  auto scale_values = torch::tensor(
      {scale},
      torch::TensorOptions().device(weight.device()).dtype(torch::kFloat32));
  auto layout_header = torch::zeros(
      {kLayoutHeaderLen},
      torch::TensorOptions().device(weight.device()).dtype(torch::kInt32));
  auto segment_offsets = torch::tensor(
      {0, static_cast<int32_t>(logical_out)},
      torch::TensorOptions().device(weight.device()).dtype(torch::kInt32));

  auto header_cpu = torch::zeros(
      {kLayoutHeaderLen},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
  auto header_acc = header_cpu.accessor<int32_t, 1>();
  header_acc[kIdxFormatVersion] = 1;
  header_acc[kIdxTileN] = 16;
  header_acc[kIdxTileK] = 32;
  header_acc[kIdxLogicalOut] = static_cast<int32_t>(logical_out);
  header_acc[kIdxLogicalIn] = static_cast<int32_t>(logical_in);
  header_acc[kIdxPaddedOut] = static_cast<int32_t>(padded_out);
  header_acc[kIdxPaddedIn] = static_cast<int32_t>(padded_in);
  header_acc[kIdxScaleGranularity] = 0;
  header_acc[kIdxScaleGroupSize] = static_cast<int32_t>(logical_out);
  header_acc[kIdxInterleaveMode] = 1;
  header_acc[kIdxArchMin] = 80;
  header_acc[kIdxSegmentCount] = 1;
  header_acc[kIdxFlags] = 0;
  layout_header.copy_(header_cpu.to(layout_header.device()));

  const int64_t total = padded_out * packed_cols;
  const dim3 blocks(static_cast<unsigned int>((total + kPackThreads - 1) / kPackThreads));
  const dim3 threads(kPackThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(weight.device().index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weight_contig.scalar_type(),
      "bitnet_pack_weight_cuda",
      [&] {
        bitnet_pack_weight_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            weight_contig.data_ptr<scalar_t>(),
            packed_weight.data_ptr<uint8_t>(),
            scale,
            logical_out,
            logical_in,
            padded_out,
            packed_cols);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(packed_weight, scale_values, layout_header, segment_offsets);
}

std::tuple<torch::Tensor, torch::Tensor> CudaBitNetRuntimeRowQuantizeForward(
    const torch::Tensor& weight,
    double eps) {
  TORCH_CHECK(weight.defined(), "CudaBitNetRuntimeRowQuantizeForward: weight must be defined");
  TORCH_CHECK(weight.is_cuda(), "CudaBitNetRuntimeRowQuantizeForward: weight must be CUDA");
  TORCH_CHECK(weight.dim() == 2, "CudaBitNetRuntimeRowQuantizeForward: weight must be rank-2");
  TORCH_CHECK(weight.size(1) > 0, "CudaBitNetRuntimeRowQuantizeForward: weight input dimension must be positive");
  TORCH_CHECK(IsSupportedLinearDtype(weight.scalar_type()),
              "CudaBitNetRuntimeRowQuantizeForward: unsupported weight dtype");

  c10::cuda::CUDAGuard device_guard(weight.device());
  auto weight_contig = weight.contiguous();
  const auto rows = weight_contig.size(0);
  const auto cols = weight_contig.size(1);
  auto qweight = torch::empty(
      {rows, cols},
      torch::TensorOptions().device(weight.device()).dtype(torch::kInt8));
  auto row_scale = torch::empty(
      {rows},
      torch::TensorOptions().device(weight.device()).dtype(torch::kFloat32));
  if (rows == 0) {
    return std::make_tuple(qweight, row_scale);
  }

  const dim3 blocks(static_cast<unsigned int>(rows));
  const dim3 threads(kPackThreads);
  const size_t shared_bytes = static_cast<size_t>(kPackThreads) * sizeof(float);
  auto stream = c10::cuda::getCurrentCUDAStream(weight.device().index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      weight_contig.scalar_type(),
      "bitnet_runtime_row_quantize_cuda",
      [&] {
        bitnet_runtime_row_quantize_kernel<scalar_t><<<blocks, threads, shared_bytes, stream.stream()>>>(
            weight_contig.data_ptr<scalar_t>(),
            qweight.data_ptr<int8_t>(),
            row_scale.data_ptr<float>(),
            rows,
            cols,
            static_cast<float>(eps));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(qweight, row_scale);
}

}  // namespace t10::bitnet
