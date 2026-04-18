#pragma once

#include <torch/extension.h>

#include <cuda_runtime.h>

namespace t10::cuda {

inline bool DeviceComputeCapability(const torch::Tensor& reference, int* major_out, int* minor_out) {
  if (!reference.is_cuda()) {
    return false;
  }
  const int device = reference.get_device();
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return false;
  }
  if (major_out != nullptr) {
    *major_out = static_cast<int>(prop.major);
  }
  if (minor_out != nullptr) {
    *minor_out = static_cast<int>(prop.minor);
  }
  return true;
}

inline bool DeviceIsSm90OrLater(const torch::Tensor& reference) {
  int major = 0;
  return DeviceComputeCapability(reference, &major, nullptr) && major >= 9;
}

}  // namespace t10::cuda
