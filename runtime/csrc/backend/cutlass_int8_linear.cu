#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <vector>

#if defined(MODEL_STACK_WITH_CUTLASS_GEMM) && MODEL_STACK_WITH_CUTLASS_GEMM
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/util/packed_stride.hpp>
#endif

namespace {

bool CutlassInt8LinearDebugEnabled() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUTLASS_DEBUG");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool IsSupportedCutlassInt8LinearOutDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

#if defined(MODEL_STACK_WITH_CUTLASS_GEMM) && MODEL_STACK_WITH_CUTLASS_GEMM && \
    defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <typename ElementD, bool HasBias>
torch::Tensor RunCutlassInt8LinearFusedSm90(
    const torch::Tensor& qx_2d,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    torch::ScalarType output_dtype) {
  using namespace cute;

  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 16;

  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 16;

  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementC = ElementD;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 16;
  using LayoutD = cutlass::layout::RowMajor;
  constexpr int AlignmentD = 16;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<_128, _256, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

  using RowScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementCompute,
      ElementCompute,
      Stride<bool, _0, int64_t>,
      1>;
  using ColScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementCompute,
      ElementCompute,
      Stride<_0, bool, int64_t>,
      1>;
  using ColBias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementD,
      ElementCompute,
      Stride<_0, _1, int64_t>,
      AlignmentD>;

  using AccTimesRow = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::multiplies,
          ElementCompute,
          ElementCompute,
          cutlass::FloatRoundStyle::round_to_nearest>,
      RowScale,
      cutlass::epilogue::fusion::Sm90AccFetch>;
  using AccTimesRowCol = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::multiplies,
          ElementCompute,
          ElementCompute,
          cutlass::FloatRoundStyle::round_to_nearest>,
      ColScale,
      AccTimesRow>;
  using FusionCallbacksWithBias = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::plus,
          ElementD,
          ElementCompute,
          cutlass::FloatRoundStyle::round_to_nearest>,
      AccTimesRowCol,
      ColBias>;
  using FusionCallbacksNoBias = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::epilogue::thread::Identity,
          ElementD,
          ElementCompute,
          cutlass::FloatRoundStyle::round_to_nearest>,
      AccTimesRowCol>;
  using FusionCallbacks = std::conditional_t<HasBias, FusionCallbacksWithBias, FusionCallbacksNoBias>;

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
      LayoutD,
      AlignmentD,
      EpilogueSchedule,
      FusionCallbacks>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA,
      AlignmentA,
      ElementB,
      LayoutB,
      AlignmentB,
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
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  const auto m = qx_2d.size(0);
  const auto k = qx_2d.size(1);
  const auto n = qweight.size(0);
  auto out = torch::empty({m, n}, qx_2d.options().dtype(output_dtype));
  if (m == 0 || n == 0) {
    return out;
  }
  if ((k % AlignmentA) != 0 || (k % AlignmentB) != 0 || (n % AlignmentD) != 0) {
    return torch::Tensor();
  }

  auto stream = c10::cuda::getCurrentCUDAStream(qx_2d.get_device());
  auto problem_shape = cute::make_shape(
      static_cast<int>(m),
      static_cast<int>(n),
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
      cute::make_shape(static_cast<int>(m), static_cast<int>(n), 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      StrideD{},
      cute::make_shape(static_cast<int>(m), static_cast<int>(n), 1));

  typename FusionCallbacks::Arguments fusion_args = [&]() {
    if constexpr (HasBias) {
      const ElementD* bias_ptr = reinterpret_cast<const ElementD*>(bias.value().data_ptr());
      return typename FusionCallbacks::Arguments{
          {
              {inv_scale.data_ptr<float>(), ElementCompute(1), Stride<_0, bool, int64_t>{_0{}, bool(1), 0}},
              {
                  {x_scale.data_ptr<float>(), ElementCompute(1), Stride<bool, _0, int64_t>{bool(1), _0{}, 0}},
                  {},
                  {}
              },
              {}
          },
          {bias_ptr, ElementD(0), Stride<_0, _1, int64_t>{_0{}, _1{}, 0}},
          {}
      };
    } else {
      return typename FusionCallbacks::Arguments{
          {
              {inv_scale.data_ptr<float>(), ElementCompute(1), Stride<_0, bool, int64_t>{_0{}, bool(1), 0}},
              {
                  {x_scale.data_ptr<float>(), ElementCompute(1), Stride<bool, _0, int64_t>{bool(1), _0{}, 0}},
                  {},
                  {}
              },
              {},
          },
          {}
      };
    }
  }();

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {
          qx_2d.data_ptr<int8_t>(),
          stride_a,
          qweight.data_ptr<int8_t>(),
          stride_b,
      },
      {
          fusion_args,
          static_cast<ElementC*>(nullptr),
          stride_c,
          reinterpret_cast<ElementD*>(out.data_ptr()),
          stride_d,
      }};

  Gemm gemm;
  const auto can_implement_status = gemm.can_implement(arguments);
  if (can_implement_status != cutlass::Status::kSuccess) {
    if (CutlassInt8LinearDebugEnabled()) {
      TORCH_WARN("CUTLASS int8 fused linear cannot implement shape m=", m, " n=", n, " k=", k);
    }
    return torch::Tensor();
  }
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  torch::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        qx_2d.options().dtype(torch::kUInt8));
    workspace_ptr = workspace.data_ptr();
  }
  auto status = gemm.initialize(arguments, workspace_ptr, stream.stream());
  if (status == cutlass::Status::kSuccess) {
    status = gemm.run(stream.stream());
  }
  if (status != cutlass::Status::kSuccess) {
    if (CutlassInt8LinearDebugEnabled()) {
      TORCH_WARN("CUTLASS int8 fused linear run failed for shape m=", m, " n=", n, " k=", k);
    }
    return torch::Tensor();
  }
  return out;
}

#endif

}  // namespace

torch::Tensor CutlassInt8LinearFusedForward(
    const torch::Tensor& qx,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
#if defined(MODEL_STACK_WITH_CUTLASS_GEMM) && MODEL_STACK_WITH_CUTLASS_GEMM && \
    defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (!qx.is_cuda() || !x_scale.is_cuda() || !qweight.is_cuda() || !inv_scale.is_cuda()) {
    return torch::Tensor();
  }
  if (qx.scalar_type() != torch::kInt8 || qweight.scalar_type() != torch::kInt8) {
    return torch::Tensor();
  }
  if (x_scale.scalar_type() != torch::kFloat32 || inv_scale.scalar_type() != torch::kFloat32) {
    return torch::Tensor();
  }
  const auto output_dtype = out_dtype.has_value() ? out_dtype.value() : torch::kBFloat16;
  if (!IsSupportedCutlassInt8LinearOutDtype(output_dtype)) {
    return torch::Tensor();
  }
  if (bias.has_value() && bias.value().defined() && !bias.value().is_cuda()) {
    return torch::Tensor();
  }

  c10::cuda::CUDAGuard device_guard{qx.device()};
  auto qx_contig = qx.contiguous();
  auto x_scale_contig = x_scale.reshape({-1}).contiguous();
  auto qweight_contig = qweight.contiguous();
  auto inv_scale_contig = inv_scale.contiguous();
  const auto in_features = qx_contig.size(-1);
  const auto rows = qx_contig.numel() / in_features;
  const auto out_features = qweight_contig.size(0);
  if (x_scale_contig.numel() != rows || qweight_contig.size(1) != in_features ||
      inv_scale_contig.size(0) != out_features) {
    return torch::Tensor();
  }
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    if (bias.value().numel() != out_features) {
      return torch::Tensor();
    }
    bias_cast = bias.value().to(qx_contig.device(), output_dtype).contiguous();
  }
  const bool has_bias = bias_cast.has_value() && bias_cast.value().defined();
  auto qx_2d = qx_contig.view({rows, in_features});
  if (output_dtype == torch::kBFloat16) {
    auto out = has_bias
        ? RunCutlassInt8LinearFusedSm90<cutlass::bfloat16_t, true>(
              qx_2d,
              x_scale_contig,
              qweight_contig,
              inv_scale_contig,
              bias_cast,
              output_dtype)
        : RunCutlassInt8LinearFusedSm90<cutlass::bfloat16_t, false>(
              qx_2d,
              x_scale_contig,
              qweight_contig,
              inv_scale_contig,
              bias_cast,
              output_dtype);
    if (!out.defined()) {
      return torch::Tensor();
    }
    std::vector<int64_t> out_sizes(qx_contig.sizes().begin(), qx_contig.sizes().end());
    out_sizes.back() = out_features;
    return out.view(out_sizes);
  }
  if (output_dtype == torch::kFloat16) {
    auto out = has_bias
        ? RunCutlassInt8LinearFusedSm90<cutlass::half_t, true>(
              qx_2d,
              x_scale_contig,
              qweight_contig,
              inv_scale_contig,
              bias_cast,
              output_dtype)
        : RunCutlassInt8LinearFusedSm90<cutlass::half_t, false>(
              qx_2d,
              x_scale_contig,
              qweight_contig,
              inv_scale_contig,
              bias_cast,
              output_dtype);
    if (!out.defined()) {
      return torch::Tensor();
    }
    std::vector<int64_t> out_sizes(qx_contig.sizes().begin(), qx_contig.sizes().end());
    out_sizes.back() = out_features;
    return out.view(out_sizes);
  }
#else
  (void)qx;
  (void)x_scale;
  (void)qweight;
  (void)inv_scale;
  (void)bias;
  (void)out_dtype;
#endif
  return torch::Tensor();
}
