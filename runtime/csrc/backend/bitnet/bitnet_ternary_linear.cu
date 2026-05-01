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
constexpr int kTernaryLinearBlockRows = 1;
constexpr int kTernaryLinearBlockCols = 64;
constexpr int kTernaryLinearGroupLanes = 4;
constexpr int kTernaryLinearChunkWords = 4;
constexpr int kTernaryLinearChunkValues = kTernaryLinearChunkWords * 32;
constexpr int kTernaryLinearThreadsX = kTernaryLinearBlockCols * kTernaryLinearGroupLanes;
constexpr int kStrictTernaryBlockRows = 1;
constexpr int kStrictTernaryGroupLanes = 8;
constexpr int kStrictTernaryBlockCols = 16;
constexpr int kStrictTernaryThreadsX = kStrictTernaryBlockCols * kStrictTernaryGroupLanes;

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
__global__ void bitnet_ternary_quantize_activation_kernel(
    const scalar_t* __restrict__ x,
    uint32_t* __restrict__ pos_masks,
    uint32_t* __restrict__ neg_masks,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols,
    int64_t word_cols,
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
    local_sum += fabsf(static_cast<float>(x[row_offset + col]));
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

  for (int64_t word_col = thread; word_col < word_cols; word_col += blockDim.x) {
    const int64_t col_base = word_col * 32;
    uint32_t pos = 0;
    uint32_t neg = 0;
#pragma unroll
    for (int bit = 0; bit < 32; ++bit) {
      const int64_t col = col_base + bit;
      if (col < cols) {
        const float normalized = static_cast<float>(x[row_offset + col]) / scale;
        const int code = max(-1, min(1, static_cast<int>(nearbyintf(normalized))));
        pos |= static_cast<uint32_t>(code > 0) << bit;
        neg |= static_cast<uint32_t>(code < 0) << bit;
      }
    }
    pos_masks[row * word_cols + word_col] = pos;
    neg_masks[row * word_cols + word_col] = neg;
  }
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
    int64_t word_cols,
    int64_t n_tiles) {
  __shared__ scalar_t x_tile[kTernaryLinearBlockRows][kTernaryLinearChunkValues];
  __shared__ uint32_t pos_tile[kTernaryLinearBlockCols][kTernaryLinearChunkWords];
  __shared__ uint32_t neg_tile[kTernaryLinearBlockCols][kTernaryLinearChunkWords];
  const int row_in_block = static_cast<int>(threadIdx.y);
  const int lane_in_group = static_cast<int>(threadIdx.x) & (kTernaryLinearGroupLanes - 1);
  const int col_in_block = static_cast<int>(threadIdx.x) / kTernaryLinearGroupLanes;
  const int64_t tile = static_cast<int64_t>(blockIdx.x);
  const int64_t row_tile = tile / n_tiles;
  const int64_t col_tile = tile - row_tile * n_tiles;
  const int64_t row = row_tile * kTernaryLinearBlockRows + row_in_block;
  const int64_t col = col_tile * kTernaryLinearBlockCols + col_in_block;
  const bool valid_row = row < m;
  const bool valid_col = col < n;

  float acc = 0.0f;
  const int64_t x_row = row * k;
  const int64_t mask_row = col * word_cols;
  for (int64_t word_base = 0; word_base < word_cols; word_base += kTernaryLinearChunkWords) {
    for (int load = static_cast<int>(threadIdx.x); load < kTernaryLinearChunkValues; load += kTernaryLinearThreadsX) {
      const int64_t kk_load = word_base * 32 + load;
      x_tile[row_in_block][load] =
          (valid_row && kk_load < k) ? x[x_row + kk_load] : static_cast<scalar_t>(0.0f);
    }
    if (row_in_block == 0) {
      const int64_t mask_word = word_base + lane_in_group;
      if (col < n && mask_word < word_cols) {
        pos_tile[col_in_block][lane_in_group] = pos_masks[mask_row + mask_word];
        neg_tile[col_in_block][lane_in_group] = neg_masks[mask_row + mask_word];
      } else {
        pos_tile[col_in_block][lane_in_group] = 0;
        neg_tile[col_in_block][lane_in_group] = 0;
      }
    }
    __syncthreads();

    const int64_t word_col = word_base + lane_in_group;
    if (valid_row && valid_col && word_col < word_cols) {
      uint32_t pos = pos_tile[col_in_block][lane_in_group];
      uint32_t neg = neg_tile[col_in_block][lane_in_group];
      const int tile_base = lane_in_group * 32;
      while (pos != 0) {
        const int bit = __ffs(pos) - 1;
        const int64_t kk = word_col * 32 + bit;
        if (kk < k) {
          acc += static_cast<float>(x_tile[row_in_block][tile_base + bit]);
        }
        pos &= pos - 1;
      }
      while (neg != 0) {
        const int bit = __ffs(neg) - 1;
        const int64_t kk = word_col * 32 + bit;
        if (kk < k) {
          acc -= static_cast<float>(x_tile[row_in_block][tile_base + bit]);
        }
        neg &= neg - 1;
      }
    }
    __syncthreads();
  }

  acc += __shfl_down_sync(0xffffffff, acc, 2, kTernaryLinearGroupLanes);
  acc += __shfl_down_sync(0xffffffff, acc, 1, kTernaryLinearGroupLanes);
  if (valid_row && valid_col && lane_in_group == 0) {
    out[row * n + col] = static_cast<scalar_t>(acc * row_scale[col]);
  }
}

template <typename scalar_t, int StaticWordCols>
__global__ void bitnet_strict_ternary_linear_kernel(
    const uint32_t* __restrict__ x_pos_masks,
    const uint32_t* __restrict__ x_neg_masks,
    const float* __restrict__ x_row_scale,
    const uint32_t* __restrict__ w_pos_masks,
    const uint32_t* __restrict__ w_neg_masks,
    const float* __restrict__ w_row_scale,
    scalar_t* __restrict__ out,
    int64_t m,
    int64_t n,
    int64_t word_cols,
    int64_t n_tiles) {
  const int lane_in_group = static_cast<int>(threadIdx.x) & (kStrictTernaryGroupLanes - 1);
  const int col_in_block = static_cast<int>(threadIdx.x) / kStrictTernaryGroupLanes;
  const int row_in_block = static_cast<int>(threadIdx.y);
  const int64_t tile = static_cast<int64_t>(blockIdx.x);
  const int64_t row_tile = tile / n_tiles;
  const int64_t col_tile = tile - row_tile * n_tiles;
  const int64_t row = row_tile * kStrictTernaryBlockRows + row_in_block;
  const int64_t col = col_tile * kStrictTernaryBlockCols + col_in_block;
  const bool valid = row < m && col < n;

  int acc = 0;
  if constexpr (StaticWordCols > 0) {
#pragma unroll
    for (int word_col = lane_in_group; word_col < StaticWordCols; word_col += kStrictTernaryGroupLanes) {
      if (valid) {
        const uint32_t xp = x_pos_masks[row * word_cols + word_col];
        const uint32_t xn = x_neg_masks[row * word_cols + word_col];
        const uint32_t wp = w_pos_masks[col * word_cols + word_col];
        const uint32_t wn = w_neg_masks[col * word_cols + word_col];
        acc += __popc((xp & wp) | (xn & wn));
        acc -= __popc((xp & wn) | (xn & wp));
      }
    }
  } else {
    for (int64_t word_col = lane_in_group; word_col < word_cols; word_col += kStrictTernaryGroupLanes) {
      if (valid) {
        const uint32_t xp = x_pos_masks[row * word_cols + word_col];
        const uint32_t xn = x_neg_masks[row * word_cols + word_col];
        const uint32_t wp = w_pos_masks[col * word_cols + word_col];
        const uint32_t wn = w_neg_masks[col * word_cols + word_col];
        acc += __popc((xp & wp) | (xn & wn));
        acc -= __popc((xp & wn) | (xn & wp));
      }
    }
  }

  if constexpr (kStrictTernaryGroupLanes >= 32) {
    acc += __shfl_down_sync(0xffffffff, acc, 16, kStrictTernaryGroupLanes);
  }
  if constexpr (kStrictTernaryGroupLanes >= 16) {
    acc += __shfl_down_sync(0xffffffff, acc, 8, kStrictTernaryGroupLanes);
  }
  if constexpr (kStrictTernaryGroupLanes >= 8) {
    acc += __shfl_down_sync(0xffffffff, acc, 4, kStrictTernaryGroupLanes);
  }
  acc += __shfl_down_sync(0xffffffff, acc, 2, kStrictTernaryGroupLanes);
  acc += __shfl_down_sync(0xffffffff, acc, 1, kStrictTernaryGroupLanes);
  if (valid && lane_in_group == 0) {
    const float scale = x_row_scale[row] * w_row_scale[col];
    out[row * n + col] = static_cast<scalar_t>(static_cast<float>(acc) * scale);
  }
}

template <typename scalar_t, int StaticWordCols>
__global__ void bitnet_strict_ternary_linear_aligned_kernel(
    const uint32_t* __restrict__ x_pos_masks,
    const uint32_t* __restrict__ x_neg_masks,
    const float* __restrict__ x_row_scale,
    const uint32_t* __restrict__ w_pos_masks,
    const uint32_t* __restrict__ w_neg_masks,
    const float* __restrict__ w_row_scale,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t word_cols,
    int64_t n_tiles) {
  static_assert(StaticWordCols > 0, "aligned strict ternary kernel requires a static word count");
  const int lane_in_group = static_cast<int>(threadIdx.x) & (kStrictTernaryGroupLanes - 1);
  const int col_in_block = static_cast<int>(threadIdx.x) / kStrictTernaryGroupLanes;
  const int row_in_block = static_cast<int>(threadIdx.y);
  const int64_t tile = static_cast<int64_t>(blockIdx.x);
  const int64_t row_tile = tile / n_tiles;
  const int64_t col_tile = tile - row_tile * n_tiles;
  const int64_t row = row_tile * kStrictTernaryBlockRows + row_in_block;
  const int64_t col = col_tile * kStrictTernaryBlockCols + col_in_block;

  int acc = 0;
#pragma unroll
  for (int word_col = lane_in_group; word_col < StaticWordCols; word_col += kStrictTernaryGroupLanes) {
    const uint32_t xp = x_pos_masks[row * word_cols + word_col];
    const uint32_t xn = x_neg_masks[row * word_cols + word_col];
    const uint32_t wp = w_pos_masks[col * word_cols + word_col];
    const uint32_t wn = w_neg_masks[col * word_cols + word_col];
    acc += __popc((xp & wp) | (xn & wn));
    acc -= __popc((xp & wn) | (xn & wp));
  }

  if constexpr (kStrictTernaryGroupLanes >= 32) {
    acc += __shfl_down_sync(0xffffffff, acc, 16, kStrictTernaryGroupLanes);
  }
  if constexpr (kStrictTernaryGroupLanes >= 16) {
    acc += __shfl_down_sync(0xffffffff, acc, 8, kStrictTernaryGroupLanes);
  }
  if constexpr (kStrictTernaryGroupLanes >= 8) {
    acc += __shfl_down_sync(0xffffffff, acc, 4, kStrictTernaryGroupLanes);
  }
  acc += __shfl_down_sync(0xffffffff, acc, 2, kStrictTernaryGroupLanes);
  acc += __shfl_down_sync(0xffffffff, acc, 1, kStrictTernaryGroupLanes);
  if (lane_in_group == 0) {
    const float scale = x_row_scale[row] * w_row_scale[col];
    out[row * n + col] = static_cast<scalar_t>(static_cast<float>(acc) * scale);
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

std::vector<torch::Tensor> CudaBitNetTernaryQuantizeActivationForward(
    const torch::Tensor& x,
    double eps) {
  TORCH_CHECK(x.defined(), "bitnet_ternary_quantize_activation_forward: x must be defined");
  TORCH_CHECK(x.is_cuda(), "bitnet_ternary_quantize_activation_forward: x must be CUDA");
  TORCH_CHECK(x.dim() >= 2, "bitnet_ternary_quantize_activation_forward: x must have rank >= 2");
  TORCH_CHECK(IsSupportedLinearDtype(x.scalar_type()),
              "bitnet_ternary_quantize_activation_forward: unsupported x dtype");

  c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contig = x.contiguous();
  const int64_t cols = x_contig.size(-1);
  const int64_t rows = x_contig.numel() / cols;
  const int64_t word_cols = (cols + 31) / 32;
  auto mask_options = x_contig.options().dtype(torch::kInt32);
  auto pos_masks = torch::empty({rows, word_cols}, mask_options);
  auto neg_masks = torch::empty({rows, word_cols}, mask_options);
  auto row_scale = torch::empty({rows}, x_contig.options().dtype(torch::kFloat32));
  if (rows == 0 || cols == 0) {
    return {pos_masks, neg_masks, row_scale};
  }

  const dim3 blocks(static_cast<unsigned int>(rows));
  const dim3 threads(kTernaryPackThreads);
  const size_t shared_bytes = static_cast<size_t>(kTernaryPackThreads) * sizeof(float);
  auto stream = c10::cuda::getCurrentCUDAStream(x.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      x_contig.scalar_type(),
      "bitnet_ternary_quantize_activation_cuda",
      [&] {
        bitnet_ternary_quantize_activation_kernel<scalar_t><<<blocks, threads, shared_bytes, stream.stream()>>>(
            x_contig.view({rows, cols}).data_ptr<scalar_t>(),
            reinterpret_cast<uint32_t*>(pos_masks.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(neg_masks.data_ptr<int32_t>()),
            row_scale.data_ptr<float>(),
            rows,
            cols,
            word_cols,
            static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {pos_masks, neg_masks, row_scale};
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

  const dim3 threads(kTernaryLinearThreadsX, kTernaryLinearBlockRows);
  const int64_t n_tiles = (n + kTernaryLinearBlockCols - 1) / kTernaryLinearBlockCols;
  const int64_t row_tiles = (rows + kTernaryLinearBlockRows - 1) / kTernaryLinearBlockRows;
  const dim3 blocks(static_cast<unsigned int>(n_tiles * row_tiles));
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
            word_cols,
            n_tiles);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor CudaBitNetStrictTernaryLinearForward(
    const torch::Tensor& x_pos_masks,
    const torch::Tensor& x_neg_masks,
    const torch::Tensor& x_row_scale,
    const torch::Tensor& w_pos_masks,
    const torch::Tensor& w_neg_masks,
    const torch::Tensor& w_row_scale,
    torch::ScalarType out_dtype) {
  TORCH_CHECK(x_pos_masks.defined() && x_neg_masks.defined() && x_row_scale.defined() &&
                  w_pos_masks.defined() && w_neg_masks.defined() && w_row_scale.defined(),
              "bitnet_strict_ternary_linear_forward: all inputs must be defined");
  TORCH_CHECK(x_pos_masks.is_cuda() && x_neg_masks.is_cuda() && x_row_scale.is_cuda() &&
                  w_pos_masks.is_cuda() && w_neg_masks.is_cuda() && w_row_scale.is_cuda(),
              "bitnet_strict_ternary_linear_forward: all inputs must be CUDA");
  TORCH_CHECK(x_pos_masks.dim() == 2 && x_neg_masks.dim() == 2 &&
                  w_pos_masks.dim() == 2 && w_neg_masks.dim() == 2,
              "bitnet_strict_ternary_linear_forward: masks must be rank-2");
  TORCH_CHECK(x_pos_masks.scalar_type() == torch::kInt32 && x_neg_masks.scalar_type() == torch::kInt32 &&
                  w_pos_masks.scalar_type() == torch::kInt32 && w_neg_masks.scalar_type() == torch::kInt32,
              "bitnet_strict_ternary_linear_forward: masks must use int32 storage");
  TORCH_CHECK(x_pos_masks.sizes() == x_neg_masks.sizes(),
              "bitnet_strict_ternary_linear_forward: activation mask shape mismatch");
  TORCH_CHECK(w_pos_masks.sizes() == w_neg_masks.sizes(),
              "bitnet_strict_ternary_linear_forward: weight mask shape mismatch");
  TORCH_CHECK(x_pos_masks.size(1) == w_pos_masks.size(1),
              "bitnet_strict_ternary_linear_forward: mask K mismatch");
  TORCH_CHECK(x_row_scale.dim() == 1 && w_row_scale.dim() == 1 &&
                  x_row_scale.scalar_type() == torch::kFloat32 && w_row_scale.scalar_type() == torch::kFloat32,
              "bitnet_strict_ternary_linear_forward: scales must be rank-1 float32");
  TORCH_CHECK(out_dtype == torch::kFloat32 || out_dtype == torch::kFloat16 || out_dtype == torch::kBFloat16,
              "bitnet_strict_ternary_linear_forward: unsupported output dtype");

  c10::cuda::CUDAGuard device_guard(x_pos_masks.device());
  auto xp = x_pos_masks.contiguous();
  auto xn = x_neg_masks.contiguous();
  auto xs = x_row_scale.contiguous();
  auto wp = w_pos_masks.contiguous();
  auto wn = w_neg_masks.contiguous();
  auto ws = w_row_scale.contiguous();
  const int64_t rows = xp.size(0);
  const int64_t word_cols = xp.size(1);
  const int64_t n = wp.size(0);
  TORCH_CHECK(xs.size(0) == rows, "bitnet_strict_ternary_linear_forward: activation scale rows mismatch");
  TORCH_CHECK(ws.size(0) == n, "bitnet_strict_ternary_linear_forward: weight scale rows mismatch");
  auto out = torch::empty({rows, n}, xp.options().dtype(out_dtype));
  if (rows == 0 || n == 0) {
    return out;
  }

  const int64_t n_tiles = (n + kStrictTernaryBlockCols - 1) / kStrictTernaryBlockCols;
  const int64_t row_tiles = (rows + kStrictTernaryBlockRows - 1) / kStrictTernaryBlockRows;
  const dim3 blocks(static_cast<unsigned int>(row_tiles * n_tiles));
  const dim3 threads(kStrictTernaryThreadsX, kStrictTernaryBlockRows);
  const bool use_aligned_kernel = (rows % kStrictTernaryBlockRows) == 0 && (n % kStrictTernaryBlockCols) == 0;
  auto stream = c10::cuda::getCurrentCUDAStream(x_pos_masks.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      out_dtype,
      "bitnet_strict_ternary_linear_cuda",
      [&] {
        if (word_cols == 32) {
          if (use_aligned_kernel) {
            bitnet_strict_ternary_linear_aligned_kernel<scalar_t, 32><<<blocks, threads, 0, stream.stream()>>>(
                reinterpret_cast<const uint32_t*>(xp.data_ptr<int32_t>()),
                reinterpret_cast<const uint32_t*>(xn.data_ptr<int32_t>()),
                xs.data_ptr<float>(),
                reinterpret_cast<const uint32_t*>(wp.data_ptr<int32_t>()),
                reinterpret_cast<const uint32_t*>(wn.data_ptr<int32_t>()),
                ws.data_ptr<float>(),
                out.data_ptr<scalar_t>(),
                n,
                word_cols,
                n_tiles);
          } else {
            bitnet_strict_ternary_linear_kernel<scalar_t, 32><<<blocks, threads, 0, stream.stream()>>>(
              reinterpret_cast<const uint32_t*>(xp.data_ptr<int32_t>()),
              reinterpret_cast<const uint32_t*>(xn.data_ptr<int32_t>()),
              xs.data_ptr<float>(),
              reinterpret_cast<const uint32_t*>(wp.data_ptr<int32_t>()),
              reinterpret_cast<const uint32_t*>(wn.data_ptr<int32_t>()),
              ws.data_ptr<float>(),
              out.data_ptr<scalar_t>(),
              rows,
              n,
              word_cols,
              n_tiles);
          }
        } else if (word_cols == 64) {
          if (use_aligned_kernel) {
            bitnet_strict_ternary_linear_aligned_kernel<scalar_t, 64><<<blocks, threads, 0, stream.stream()>>>(
              reinterpret_cast<const uint32_t*>(xp.data_ptr<int32_t>()),
              reinterpret_cast<const uint32_t*>(xn.data_ptr<int32_t>()),
              xs.data_ptr<float>(),
              reinterpret_cast<const uint32_t*>(wp.data_ptr<int32_t>()),
              reinterpret_cast<const uint32_t*>(wn.data_ptr<int32_t>()),
              ws.data_ptr<float>(),
              out.data_ptr<scalar_t>(),
              n,
              word_cols,
              n_tiles);
          } else {
            bitnet_strict_ternary_linear_kernel<scalar_t, 64><<<blocks, threads, 0, stream.stream()>>>(
              reinterpret_cast<const uint32_t*>(xp.data_ptr<int32_t>()),
              reinterpret_cast<const uint32_t*>(xn.data_ptr<int32_t>()),
              xs.data_ptr<float>(),
              reinterpret_cast<const uint32_t*>(wp.data_ptr<int32_t>()),
              reinterpret_cast<const uint32_t*>(wn.data_ptr<int32_t>()),
              ws.data_ptr<float>(),
              out.data_ptr<scalar_t>(),
              rows,
              n,
              word_cols,
              n_tiles);
          }
        } else {
          bitnet_strict_ternary_linear_kernel<scalar_t, 0><<<blocks, threads, 0, stream.stream()>>>(
            reinterpret_cast<const uint32_t*>(xp.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(xn.data_ptr<int32_t>()),
            xs.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(wp.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(wn.data_ptr<int32_t>()),
            ws.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            rows,
            n,
            word_cols,
            n_tiles);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace t10::bitnet
