#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include "bitnet_common.cuh"

namespace t10::bitnet {
namespace {

constexpr int kTernaryPackThreads = 256;
constexpr int kTernaryLinearBlockRows = 8;
constexpr int kTernaryLinearBlockCols = 32;

__global__ void bitnet_ternary_pack_masks_kernel(
    const int8_t* __restrict__ qweight,
    uint32_t* __restrict__ pos_masks,
    uint32_t* __restrict__ neg_masks,
    int64_t rows,
    int64_t cols,
    int64_t word_cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * word_cols;
  if (idx >= total) {
    return;
  }
  const int64_t row = idx / word_cols;
  const int64_t word_col = idx - row * word_cols;
  const int64_t col_base = word_col * 32;
  uint32_t pos = 0;
  uint32_t neg = 0;
#pragma unroll
  for (int bit = 0; bit < 32; ++bit) {
    const int64_t col = col_base + bit;
    if (col < cols) {
      const int value = static_cast<int>(qweight[row * cols + col]);
      pos |= static_cast<uint32_t>(value > 0) << bit;
      neg |= static_cast<uint32_t>(value < 0) << bit;
    }
  }
  pos_masks[idx] = pos;
  neg_masks[idx] = neg;
}

template <typename scalar_t>
__global__ void bitnet_ternary_linear_kernel(
    const scalar_t* __restrict__ x,
    const uint32_t* __restrict__ pos_masks,
    const uint32_t* __restrict__ neg_masks,
    const float* __restrict__ row_scale,
    scalar_t* __restrict__ out,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t word_cols) {
  __shared__ scalar_t x_tile[kTernaryLinearBlockRows][32];
  const int row_in_block = static_cast<int>(threadIdx.y);
  const int col_in_block = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kTernaryLinearBlockRows + row_in_block;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * kTernaryLinearBlockCols + col_in_block;
  const bool valid_row = row < m;

  float acc = 0.0f;
  const int64_t x_row = row * k;
  const int64_t mask_row = col * word_cols;
  for (int64_t word_col = 0; word_col < word_cols; ++word_col) {
    const int64_t k_base = word_col * 32;
    const int64_t kk_load = k_base + col_in_block;
    x_tile[row_in_block][col_in_block] =
        (valid_row && kk_load < k) ? x[x_row + kk_load] : static_cast<scalar_t>(0.0f);
    __syncthreads();
    if (!valid_row || col >= n) {
      __syncthreads();
      continue;
    }
    uint32_t pos = pos_masks[mask_row + word_col];
    uint32_t neg = neg_masks[mask_row + word_col];
    while (pos != 0) {
      const int bit = __ffs(pos) - 1;
      const int64_t kk = k_base + bit;
      if (kk < k) {
        acc += static_cast<float>(x_tile[row_in_block][bit]);
      }
      pos &= pos - 1;
    }
    while (neg != 0) {
      const int bit = __ffs(neg) - 1;
      const int64_t kk = k_base + bit;
      if (kk < k) {
        acc -= static_cast<float>(x_tile[row_in_block][bit]);
      }
      neg &= neg - 1;
    }
    __syncthreads();
  }
  if (valid_row && col < n) {
    out[row * n + col] = static_cast<scalar_t>(acc * row_scale[col]);
  }
}

}  // namespace

std::vector<torch::Tensor> CudaBitNetTernaryPackMasksForward(const torch::Tensor& qweight) {
  TORCH_CHECK(qweight.defined(), "bitnet_ternary_pack_masks_forward: qweight must be defined");
  TORCH_CHECK(qweight.is_cuda(), "bitnet_ternary_pack_masks_forward: qweight must be CUDA");
  TORCH_CHECK(qweight.dim() == 2, "bitnet_ternary_pack_masks_forward: qweight must be rank-2");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt8,
              "bitnet_ternary_pack_masks_forward: qweight must use int8 storage");

  c10::cuda::CUDAGuard device_guard(qweight.device());
  auto q_contig = qweight.contiguous();
  const int64_t rows = q_contig.size(0);
  const int64_t cols = q_contig.size(1);
  const int64_t word_cols = (cols + 31) / 32;
  auto options = q_contig.options().dtype(torch::kInt32);
  auto pos_masks = torch::empty({rows, word_cols}, options);
  auto neg_masks = torch::empty({rows, word_cols}, options);
  if (rows == 0 || cols == 0) {
    return {pos_masks, neg_masks};
  }

  const int64_t total = rows * word_cols;
  const dim3 blocks(static_cast<unsigned int>((total + kTernaryPackThreads - 1) / kTernaryPackThreads));
  const dim3 threads(kTernaryPackThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(qweight.device().index());
  bitnet_ternary_pack_masks_kernel<<<blocks, threads, 0, stream.stream()>>>(
      q_contig.data_ptr<int8_t>(),
      reinterpret_cast<uint32_t*>(pos_masks.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(neg_masks.data_ptr<int32_t>()),
      rows,
      cols,
      word_cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {pos_masks, neg_masks};
}

torch::Tensor CudaBitNetTernaryLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& pos_masks,
    const torch::Tensor& neg_masks,
    const torch::Tensor& row_scale) {
  TORCH_CHECK(x.defined() && pos_masks.defined() && neg_masks.defined() && row_scale.defined(),
              "bitnet_ternary_linear_forward: all inputs must be defined");
  TORCH_CHECK(x.is_cuda() && pos_masks.is_cuda() && neg_masks.is_cuda() && row_scale.is_cuda(),
              "bitnet_ternary_linear_forward: all inputs must be CUDA");
  TORCH_CHECK(x.dim() >= 2, "bitnet_ternary_linear_forward: x must have rank >= 2");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "bitnet_ternary_linear_forward: unsupported x dtype");
  TORCH_CHECK(pos_masks.dim() == 2 && neg_masks.dim() == 2,
              "bitnet_ternary_linear_forward: masks must be rank-2");
  TORCH_CHECK(pos_masks.scalar_type() == torch::kInt32 && neg_masks.scalar_type() == torch::kInt32,
              "bitnet_ternary_linear_forward: masks must use int32 storage");
  TORCH_CHECK(pos_masks.sizes() == neg_masks.sizes(),
              "bitnet_ternary_linear_forward: mask shape mismatch");
  TORCH_CHECK(row_scale.dim() == 1 && row_scale.scalar_type() == torch::kFloat32,
              "bitnet_ternary_linear_forward: row_scale must be rank-1 float32");

  c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contig = x.contiguous();
  auto pos_contig = pos_masks.contiguous();
  auto neg_contig = neg_masks.contiguous();
  auto scale_contig = row_scale.contiguous();
  const int64_t k = x_contig.size(-1);
  const int64_t rows = x_contig.numel() / k;
  const int64_t n = pos_contig.size(0);
  const int64_t word_cols = pos_contig.size(1);
  TORCH_CHECK(scale_contig.size(0) == n, "bitnet_ternary_linear_forward: scale N mismatch");
  TORCH_CHECK(word_cols == (k + 31) / 32, "bitnet_ternary_linear_forward: mask K mismatch");

  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = n;
  auto out = torch::empty(out_sizes, x_contig.options());
  if (rows == 0 || n == 0) {
    return out;
  }

  const dim3 threads(kTernaryLinearBlockCols, kTernaryLinearBlockRows);
  const dim3 blocks(
      static_cast<unsigned int>((n + kTernaryLinearBlockCols - 1) / kTernaryLinearBlockCols),
      static_cast<unsigned int>((rows + kTernaryLinearBlockRows - 1) / kTernaryLinearBlockRows));
  auto stream = c10::cuda::getCurrentCUDAStream(x.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_contig.scalar_type(),
      "bitnet_ternary_linear_forward_cuda",
      [&] {
        bitnet_ternary_linear_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_contig.view({rows, k}).data_ptr<scalar_t>(),
            reinterpret_cast<const uint32_t*>(pos_contig.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(neg_contig.data_ptr<int32_t>()),
            scale_contig.data_ptr<float>(),
            out.view({rows, n}).data_ptr<scalar_t>(),
            rows,
            k,
            n,
            word_cols);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace t10::bitnet
