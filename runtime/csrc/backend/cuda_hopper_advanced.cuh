#pragma once

#include <torch/extension.h>

#include <cstdint>

#include <cuda/barrier>
#include <cuda_runtime.h>

namespace t10::cuda {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
namespace cde = cuda::device::experimental;
#endif

union WgmmaDescriptor {
  __host__ __device__ constexpr WgmmaDescriptor() noexcept : desc_(0) {}

  uint64_t desc_;
  struct {
    uint16_t start_address_ : 14, : 2;
    uint16_t leading_byte_offset_ : 14, : 2;
    uint16_t stride_byte_offset_ : 14, : 2;
    uint8_t : 1, base_offset_ : 3, : 4;
    uint8_t : 6, layout_type_ : 2;
  } bitfield;

  __host__ __device__ constexpr operator uint64_t() const noexcept { return desc_; }
};

inline constexpr bool BuildRequestsSm90aExperimental() {
#if defined(MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL) && MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL
  return true;
#else
  return false;
#endif
}

__device__ inline bool ArchHasSm90aFeatures() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  return true;
#else
  return false;
#endif
}

__device__ inline bool ArchHasSm90BulkAsync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  return true;
#else
  return false;
#endif
}

__device__ inline void WgmmaFenceSyncAligned() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
}

__device__ inline void WgmmaCommitGroupSyncAligned() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#endif
}

template <int N>
__device__ inline void WgmmaWaitGroupSyncAligned() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
#endif
}

template <class PointerType>
__device__ inline WgmmaDescriptor MakeWgmmaSmemDesc(
    PointerType smem_ptr,
    int layout_type,
    int leading_byte_offset,
    int stride_byte_offset) {
  WgmmaDescriptor desc;
  const auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_ = uint_ptr >> 4;
  desc.bitfield.layout_type_ = layout_type;
  desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
  desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
  desc.bitfield.base_offset_ = 0;
  return desc;
}

__device__ inline void AsyncProxyFenceSharedCta() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

__device__ inline void WgmmaFenceOperand(int& reg) {
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+r"(reg) :: "memory");
#else
  (void)reg;
#endif
}

__device__ inline void WgmmaFenceOperand(float& reg) {
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+f"(reg) :: "memory");
#else
  (void)reg;
#endif
}

__device__ inline void WgmmaM64N8K32S32U8S8(
    uint64_t desc_a,
    uint64_t desc_b,
    int (&accum)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %6, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8 "
      "{%0, %1, %2, %3},"
      " %4,"
      " %5,"
      " p;\n"
      "}\n"
      : "+r"(accum[0]), "+r"(accum[1]), "+r"(accum[2]), "+r"(accum[3])
      : "l"(desc_a), "l"(desc_b), "r"(int32_t(1)));
#else
  (void)desc_a;
  (void)desc_b;
  (void)accum;
#endif
}

__device__ inline void Sm90FenceProxyAsyncSharedCta() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cde::fence_proxy_async_shared_cta();
#endif
}

template <typename SharedPtr, typename GlobalPtr, typename Barrier>
__device__ inline void Sm90CpAsyncBulkGlobalToShared(
    SharedPtr shared_dst,
    GlobalPtr global_src,
    uint32_t bytes,
    Barrier& barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cde::cp_async_bulk_global_to_shared(shared_dst, global_src, bytes, barrier);
#else
  (void)shared_dst;
  (void)global_src;
  (void)bytes;
  (void)barrier;
#endif
}

template <typename Barrier>
__device__ inline auto Sm90BarrierArriveTx(
    Barrier& barrier,
    ptrdiff_t arrive_count,
    uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  return cuda::device::barrier_arrive_tx(barrier, arrive_count, bytes);
#else
  (void)arrive_count;
  (void)bytes;
  return barrier.arrive();
#endif
}

}  // namespace t10::cuda
