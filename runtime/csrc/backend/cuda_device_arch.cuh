#pragma once

#include <torch/extension.h>

#include <cuda_runtime.h>

#include <vector>

namespace t10::cuda {

inline const cudaDeviceProp* CachedDeviceProperties(const torch::Tensor& reference) {
  if (!reference.is_cuda()) {
    return nullptr;
  }
  const int device = reference.get_device();
  struct CachedDeviceProp {
    int device = -1;
    cudaDeviceProp prop{};
  };
  thread_local std::vector<CachedDeviceProp> cache;
  for (auto& entry : cache) {
    if (entry.device == device) {
      return &entry.prop;
    }
  }
  CachedDeviceProp entry;
  entry.device = device;
  if (cudaGetDeviceProperties(&entry.prop, device) != cudaSuccess) {
    return nullptr;
  }
  cache.push_back(entry);
  return &cache.back().prop;
}

inline bool DeviceComputeCapability(const torch::Tensor& reference, int* major_out, int* minor_out) {
  const auto* prop = CachedDeviceProperties(reference);
  if (prop == nullptr) {
    return false;
  }
  if (major_out != nullptr) {
    *major_out = static_cast<int>(prop->major);
  }
  if (minor_out != nullptr) {
    *minor_out = static_cast<int>(prop->minor);
  }
  return true;
}

inline bool DeviceIsSm90OrLater(const torch::Tensor& reference) {
  int major = 0;
  return DeviceComputeCapability(reference, &major, nullptr) && major >= 9;
}

}  // namespace t10::cuda
