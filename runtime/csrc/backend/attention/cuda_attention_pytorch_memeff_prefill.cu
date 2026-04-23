#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cuda_attention_pytorch_memeff_prefill.cuh"

#ifdef MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <cutlass/arch/arch.h>
#endif

namespace t10::cuda::attention {

#ifdef MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA

template <typename scalar_t>
struct PyTorchMemEffScalar;

template <>
struct PyTorchMemEffScalar<at::Half> {
  using type = cutlass::half_t;
};

template <>
struct PyTorchMemEffScalar<at::BFloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename scalar_t>
using PyTorchMemEffScalarT = typename PyTorchMemEffScalar<scalar_t>::type;

template <typename scalar_t>
using PyTorchMemEffKernel64x64 = PyTorchMemEffAttention::AttentionKernel<
    PyTorchMemEffScalarT<scalar_t>,
    cutlass::arch::Sm80,
    true,
    64,
    64,
    64,
    true,
    true>;

template <typename scalar_t>
using PyTorchMemEffKernel64x128 = PyTorchMemEffAttention::AttentionKernel<
    PyTorchMemEffScalarT<scalar_t>,
    cutlass::arch::Sm80,
    true,
    64,
    128,
    128,
    true,
    true>;

template <typename scalar_t>
using PyTorchMemEffKernel32x128 = PyTorchMemEffAttention::AttentionKernel<
    PyTorchMemEffScalarT<scalar_t>,
    cutlass::arch::Sm80,
    true,
    32,
    128,
    65536,
    true,
    true>;

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::kNumThreads, Kernel::kMinBlocksPerSm)
PyTorchMemEffAttentionForwardKernel(typename Kernel::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ <= 1210
  if (!params.advance_to_block()) {
    return;
  }
  Kernel::attention_kernel(params);
  return;
#endif
#endif
  printf(
      "FATAL: PyTorch mem-eff attention kernel was built for sm80-sm121, but was built for sm%d\n",
      int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template <typename Kernel, typename scalar_t>
inline bool TryLaunchPyTorchMemEffAttentionKernel(
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  const auto* props = at::cuda::getDeviceProperties(q_contig.get_device());
  if (props == nullptr) {
    return false;
  }

  const size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
  if (smem_bytes > static_cast<size_t>(props->sharedMemPerBlockOptin)) {
    return false;
  }

  typename Kernel::Params params;
  params.query_ptr =
      reinterpret_cast<const typename Kernel::scalar_t*>(q_contig.data_ptr<scalar_t>());
  params.key_ptr =
      reinterpret_cast<const typename Kernel::scalar_t*>(k_contig.data_ptr<scalar_t>());
  params.value_ptr =
      reinterpret_cast<const typename Kernel::scalar_t*>(v_contig.data_ptr<scalar_t>());
  params.output_ptr = reinterpret_cast<typename Kernel::output_t*>(out.data_ptr<scalar_t>());
  params.logsumexp_ptr = nullptr;
  params.scale = scale_value;
  params.head_dim = desc.head_dim;
  params.head_dim_value = desc.head_dim;
  params.num_queries = static_cast<int32_t>(desc.q_len);
  params.num_keys = static_cast<int32_t>(desc.kv_len);
  params.num_keys_absolute = static_cast<int32_t>(desc.kv_len);
  params.custom_mask_type = desc.causal
      ? Kernel::CausalFromTopLeft
      : Kernel::NoCustomMask;

  // Q/K/V and output are BHSD contiguous. Flatten batch and heads together so
  // each logical batch is a single [seq, dim] matrix and the kernel writes
  // directly into our output buffer without a layout-conversion copy.
  params.num_batches = static_cast<int32_t>(desc.batch * desc.q_heads);
  params.num_heads = 1;
  params.q_strideM = desc.head_dim;
  params.k_strideM = desc.head_dim;
  params.v_strideM = desc.head_dim;
  params.q_strideH = 0;
  params.k_strideH = 0;
  params.v_strideH = 0;
  params.q_strideB = static_cast<int64_t>(desc.q_len * desc.head_dim);
  params.k_strideB = static_cast<int64_t>(desc.kv_len * desc.head_dim);
  params.v_strideB = static_cast<int64_t>(desc.kv_len * desc.head_dim);
  params.o_strideM = desc.head_dim;

  torch::Tensor output_accum;
  if constexpr (Kernel::kNeedsOutputAccumulatorBuffer) {
    output_accum = torch::empty(
        {desc.batch * desc.q_heads, desc.q_len, desc.head_dim},
        out.options().dtype(torch::kFloat32));
    params.output_accum_ptr = output_accum.data_ptr<typename Kernel::output_accum_t>();
  } else {
    params.output_accum_ptr = nullptr;
  }

  if (!Kernel::check_supported(params)) {
    return false;
  }

  const dim3 blocks = params.getBlocksGrid();
  if (blocks.x == 0 || blocks.y == 0 || blocks.z == 0) {
    return true;
  }

  constexpr auto kernel_fn = PyTorchMemEffAttentionForwardKernel<Kernel>;
  if (smem_bytes > 0xc000) {
    (void)cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }
  kernel_fn<<<blocks, params.getThreadsGrid(), smem_bytes, stream>>>(params);
  return true;
}

template <typename scalar_t>
inline bool TryLaunchPyTorchMemEffAttentionPrefillImpl(
    PyTorchMemEffPrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  const auto try_64x64 = [&]() {
    return TryLaunchPyTorchMemEffAttentionKernel<PyTorchMemEffKernel64x64<scalar_t>, scalar_t>(
        q_contig, k_contig, v_contig, out, desc, scale_value, stream);
  };
  const auto try_64x128 = [&]() {
    return TryLaunchPyTorchMemEffAttentionKernel<PyTorchMemEffKernel64x128<scalar_t>, scalar_t>(
        q_contig, k_contig, v_contig, out, desc, scale_value, stream);
  };
  const auto try_32x128 = [&]() {
    return TryLaunchPyTorchMemEffAttentionKernel<PyTorchMemEffKernel32x128<scalar_t>, scalar_t>(
        q_contig, k_contig, v_contig, out, desc, scale_value, stream);
  };

  switch (kind) {
    case PyTorchMemEffPrefillKernelKind::k64x64Rf:
      return try_64x64();
    case PyTorchMemEffPrefillKernelKind::k64x128Rf:
      return try_64x128();
    case PyTorchMemEffPrefillKernelKind::k32x128Gmem:
      return try_32x128();
    case PyTorchMemEffPrefillKernelKind::kAuto:
    default:
      break;
  }

  // Mirror the upstream SM80 order first. On the 3090 bf16 prefill path this
  // is also the fastest lane we measured across short and long contexts.
  if (try_64x64()) {
    return true;
  }
  if (try_64x128()) {
    return true;
  }
  return try_32x128();
}

bool TryLaunchPyTorchMemEffAttentionPrefillF16(
    PyTorchMemEffPrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  return TryLaunchPyTorchMemEffAttentionPrefillImpl<at::Half>(
      kind, q_contig, k_contig, v_contig, out, desc, scale_value, stream);
}

bool TryLaunchPyTorchMemEffAttentionPrefillBF16(
    PyTorchMemEffPrefillKernelKind kind,
    const torch::Tensor& q_contig,
    const torch::Tensor& k_contig,
    const torch::Tensor& v_contig,
    const torch::Tensor& out,
    const t10::desc::AttentionDesc& desc,
    float scale_value,
    cudaStream_t stream) {
  return TryLaunchPyTorchMemEffAttentionPrefillImpl<at::BFloat16>(
      kind, q_contig, k_contig, v_contig, out, desc, scale_value, stream);
}

#else

bool TryLaunchPyTorchMemEffAttentionPrefillF16(
    PyTorchMemEffPrefillKernelKind,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const t10::desc::AttentionDesc&,
    float,
    cudaStream_t) {
  return false;
}

bool TryLaunchPyTorchMemEffAttentionPrefillBF16(
    PyTorchMemEffPrefillKernelKind,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const t10::desc::AttentionDesc&,
    float,
    cudaStream_t) {
  return false;
}

#endif

}  // namespace t10::cuda::attention
