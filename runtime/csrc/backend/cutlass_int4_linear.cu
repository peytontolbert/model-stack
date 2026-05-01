#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <limits>
#include <vector>

#if defined(MODEL_STACK_WITH_CUTLASS_GEMM) && MODEL_STACK_WITH_CUTLASS_GEMM
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/util/mixed_dtype_utils.hpp>
#endif

namespace {

bool IsSupportedCutlassInt4ActivationDtype(torch::ScalarType dtype) {
  return dtype == torch::kBFloat16;
}

#if defined(MODEL_STACK_WITH_CUTLASS_GEMM) && MODEL_STACK_WITH_CUTLASS_GEMM && \
    defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <class TileShape, class EngineSrc, class LayoutSrc, class EngineDst, class LayoutDst, class TiledCopy>
__global__ void model_stack_int4_reorder_tensor_kernel(
    cute::Tensor<EngineSrc, LayoutSrc> src,
    cute::Tensor<EngineDst, LayoutDst> dst,
    TiledCopy tiled_copy) {
  using namespace cute;

  Tensor tile_src = local_tile(src, TileShape{}, make_coord(blockIdx.x, _, blockIdx.z));
  Tensor tile_dst = local_tile(dst, TileShape{}, make_coord(blockIdx.x, _, blockIdx.z));

  auto thread_copy = tiled_copy.get_slice(threadIdx.x);
  Tensor thread_src = thread_copy.partition_S(tile_src);
  Tensor thread_dst = thread_copy.partition_D(tile_dst);

  copy(tiled_copy, thread_src, thread_dst);
}

template <class EngineSrc, class LayoutSrc, class EngineDst, class LayoutDst>
void ReorderTensorAsync(
    cute::Tensor<EngineSrc, LayoutSrc> src,
    cute::Tensor<EngineDst, LayoutDst> dst,
    cudaStream_t stream) {
  using namespace cute;

  using T = typename EngineDst::value_type;
  static_assert(cute::is_same_v<cute::remove_const_t<typename EngineSrc::value_type>, T>, "Type mismatch");

  auto has_major_mode = [](auto s) {
    return any_of(flatten(s), [](auto a) { return is_constant<1, decltype(a)>{}; });
  };
  constexpr int NumPackedValues = 8 / cutlass::sizeof_bits_v<T>;
  auto value_layout = cute::conditional_return<has_major_mode(stride<0>(LayoutDst{}))>(
      make_layout(make_shape(Int<NumPackedValues>{}, Int<1>{}), GenColMajor{}),
      make_layout(make_shape(Int<1>{}, Int<NumPackedValues>{}), GenRowMajor{}));

  constexpr int NumThreads = 128;
  auto thread_layout = make_layout(make_shape(Int<1>{}, Int<NumThreads>{}));
  auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, T>{}, thread_layout, value_layout);

  using ReorderTileShape = Shape<_16>;
  auto tiled_dst = group_modes<3, rank_v<LayoutDst>>(tiled_divide(dst, ReorderTileShape{}));
  dim3 blocks{unsigned(size<1>(tiled_dst)), 1u, unsigned(size<3>(tiled_dst))};
  model_stack_int4_reorder_tensor_kernel<ReorderTileShape><<<blocks, NumThreads, 0, stream>>>(
      src,
      dst,
      tiled_copy);
}

template <class T, class LayoutSrc, class LayoutDst>
void ReorderTensorAsync(
    T const* src,
    LayoutSrc const& layout_src,
    T* dst,
    LayoutDst const& layout_dst,
    cudaStream_t stream) {
  using namespace cute;
  ReorderTensorAsync(
      make_tensor(make_gmem_ptr<T>(src), layout_src),
      make_tensor(make_gmem_ptr<T>(dst), layout_dst),
      stream);
}

torch::Tensor RunCutlassInt4Bf16LinearSm90(
    const torch::Tensor& x_2d,
    const torch::Tensor& packed_weight_rowmajor,
    const torch::Tensor& scale_bf16) {
  using namespace cute;

  using ElementA = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 8;

  using ElementB = cutlass::int4b_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 32;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScale = cutlass::bfloat16_t;
  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 8;
  using ElementD = cutlass::bfloat16_t;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutDTranspose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  constexpr int AlignmentD = 8;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<_128, _128, _64>;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using ValueShuffle = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  constexpr int NumShuffleAtoms = 1;
  using MmaAtomShape = Layout<Shape<_1, Int<NumShuffleAtoms>>>;
  using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<
                                   ElementA,
                                   MmaAtomShape,
                                   ValueShuffle>());
  using LayoutBReordered = decltype(cute::tile_to_shape(
      LayoutAtomQuant{},
      Layout<Shape<int, int, int>, StrideB>{}));

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutDTranspose,
      AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      cute::tuple<ElementB, ElementScale>,
      LayoutBReordered,
      AlignmentB,
      ElementA,
      typename cutlass::layout::LayoutTranspose<LayoutA>::type,
      AlignmentA,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using StrideS = typename CollectiveMainloop::StrideScale;

  const int64_t m = x_2d.size(0);
  const int64_t k = x_2d.size(1);
  const int64_t n = scale_bf16.numel();
  if (m <= 0 || n <= 0 || k <= 0) {
    return torch::empty({m, n}, x_2d.options());
  }
  if ((m > static_cast<int64_t>(std::numeric_limits<int>::max())) ||
      (n > static_cast<int64_t>(std::numeric_limits<int>::max())) ||
      (k > static_cast<int64_t>(std::numeric_limits<int>::max()))) {
    return torch::Tensor();
  }
  if ((k % AlignmentA) != 0 || (n % AlignmentD) != 0 || (k % 2) != 0 ||
      packed_weight_rowmajor.size(0) != n || packed_weight_rowmajor.size(1) != ((k + 1) / 2)) {
    return torch::Tensor();
  }

  auto out = torch::empty({m, n}, x_2d.options());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.get_device());

  auto problem_shape = cute::make_shape(
      static_cast<int>(n),
      static_cast<int>(m),
      static_cast<int>(k),
      1);
  auto stride_a = cutlass::make_cute_packed_stride(
      StrideA{},
      cute::make_shape(static_cast<int>(m), static_cast<int>(k), 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      StrideB{},
      cute::make_shape(static_cast<int>(n), static_cast<int>(k), 1));
  auto stride_c = cutlass::make_cute_packed_stride(
      StrideC{},
      cute::make_shape(static_cast<int>(n), static_cast<int>(m), 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      StrideD{},
      cute::make_shape(static_cast<int>(n), static_cast<int>(m), 1));
  auto stride_s = cutlass::make_cute_packed_stride(
      StrideS{},
      cute::make_shape(static_cast<int>(n), 1, 1));
  auto shape_b = cute::make_shape(static_cast<int>(n), static_cast<int>(k), 1);
  auto layout_b = cute::make_layout(shape_b, stride_b);
  LayoutBReordered layout_b_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_b);
  auto reordered_weight = torch::empty_like(packed_weight_rowmajor);
  ReorderTensorAsync(
      reinterpret_cast<const ElementB*>(packed_weight_rowmajor.data_ptr<uint8_t>()),
      layout_b,
      reinterpret_cast<ElementB*>(reordered_weight.data_ptr<uint8_t>()),
      layout_b_reordered,
      stream.stream());

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {
          reinterpret_cast<const ElementB*>(reordered_weight.data_ptr<uint8_t>()),
          layout_b_reordered,
          reinterpret_cast<const ElementA*>(x_2d.data_ptr()),
          stride_a,
          reinterpret_cast<const ElementScale*>(scale_bf16.data_ptr()),
          stride_s,
          static_cast<int>(k),
      },
      {
          {ElementCompute(1.0f), ElementCompute(0.0f)},
          nullptr,
          stride_c,
          reinterpret_cast<ElementD*>(out.data_ptr()),
          stride_d,
      }};

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return torch::Tensor();
  }
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  torch::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        x_2d.options().dtype(torch::kUInt8));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream.stream());
  if (status == cutlass::Status::kSuccess) {
    status = gemm.run(stream.stream());
  }
  if (status != cutlass::Status::kSuccess) {
    return torch::Tensor();
  }
  return out;
}

#endif

}  // namespace

torch::Tensor CutlassInt4Bf16LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight_rowmajor,
    const torch::Tensor& scale,
    const c10::optional<torch::Tensor>& bias) {
#if defined(MODEL_STACK_WITH_CUTLASS_GEMM) && MODEL_STACK_WITH_CUTLASS_GEMM && \
    defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (!x.is_cuda() || !packed_weight_rowmajor.is_cuda() || !scale.is_cuda()) {
    return torch::Tensor();
  }
  if (!IsSupportedCutlassInt4ActivationDtype(x.scalar_type()) ||
      packed_weight_rowmajor.scalar_type() != torch::kUInt8) {
    return torch::Tensor();
  }
  if (bias.has_value() && bias.value().defined()) {
    return torch::Tensor();
  }
  c10::cuda::CUDAGuard device_guard{x.device()};
  auto x_contig = x.contiguous();
  auto packed_contig = packed_weight_rowmajor.contiguous();
  auto scale_bf16 = scale.to(x_contig.device(), torch::kBFloat16).contiguous().reshape({-1});
  const auto in_features = x_contig.size(-1);
  const auto rows = x_contig.numel() / in_features;
  if (packed_contig.dim() != 2 || packed_contig.size(0) != scale_bf16.numel() ||
      packed_contig.size(1) != ((in_features + 1) / 2)) {
    return torch::Tensor();
  }
  auto out_2d = RunCutlassInt4Bf16LinearSm90(
      x_contig.view({rows, in_features}),
      packed_contig,
      scale_bf16);
  if (!out_2d.defined()) {
    return torch::Tensor();
  }
  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = scale_bf16.numel();
  return out_2d.view(out_sizes);
#else
  (void)x;
  (void)packed_weight_colmajor;
  (void)scale;
  (void)bias;
  return torch::Tensor();
#endif
}
