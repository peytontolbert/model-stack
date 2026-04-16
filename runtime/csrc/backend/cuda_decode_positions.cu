#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void fill_decode_positions_kernel(
    int64_t* __restrict__ pos,
    int64_t total,
    int64_t value) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  pos[idx] = value;
}

}  // namespace

bool HasCudaDecodePositionsKernel() {
  return true;
}

std::vector<torch::Tensor> CudaDecodePositionsForward(
    int64_t batch_size,
    int64_t seq_len,
    const torch::Tensor& reference) {
  TORCH_CHECK(reference.is_cuda(), "CudaDecodePositionsForward: reference must be a CUDA tensor");
  TORCH_CHECK(batch_size > 0 && seq_len > 0,
              "CudaDecodePositionsForward: batch_size and seq_len must be positive");

  c10::cuda::CUDAGuard device_guard{reference.device()};

  auto pos = torch::empty(
      {batch_size, 1},
      torch::TensorOptions().dtype(torch::kLong).device(reference.device()));
  const auto total = pos.numel();
  const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
  const dim3 threads(kThreads);
  auto stream = c10::cuda::getCurrentCUDAStream(reference.get_device());
  fill_decode_positions_kernel<<<blocks, threads, 0, stream.stream()>>>(
      pos.data_ptr<int64_t>(),
      total,
      seq_len - 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {pos, pos.view({-1})};
}
