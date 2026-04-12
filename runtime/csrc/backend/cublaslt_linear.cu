#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/linear.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cublasLt.h>

#include <algorithm>
#include <vector>

namespace {

const char* CublasStatusString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

inline void CheckCublas(cublasStatus_t status, const char* msg) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg, ": ", CublasStatusString(status));
}

bool IsSupportedLinearDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

cudaDataType_t ToCudaDataType(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:
      return CUDA_R_32F;
    case torch::kFloat16:
      return CUDA_R_16F;
    case torch::kBFloat16:
      return CUDA_R_16BF;
    default:
      TORCH_CHECK(false, "Unsupported dtype for cublasLt linear backend");
  }
}

torch::Tensor RunCublasLtLinear(
    const torch::Tensor& x_2d,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  const auto m = x_2d.size(0);
  const auto k = x_2d.size(1);
  const auto n = weight.size(0);

  auto out = torch::empty({m, n}, x_2d.options());
  auto stream = c10::cuda::getCurrentCUDAStream(x_2d.get_device());
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto workspace = at::cuda::getCUDABlasLtWorkspace();
  auto workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  const auto scale_type = CUDA_R_32F;
  const auto data_type = ToCudaDataType(x_2d.scalar_type());
  const auto compute_type = CUBLAS_COMPUTE_32F;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;

  auto cleanup = [&]() {
    if (preference != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference);
    }
    if (c_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(c_desc);
    }
    if (b_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(b_desc);
    }
    if (a_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(a_desc);
    }
    if (op_desc != nullptr) {
      cublasLtMatmulDescDestroy(op_desc);
    }
  };

  CheckCublas(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type), "cublasLtMatmulDescCreate");
  CheckCublas(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
      "cublasLtMatmulDescSetAttribute(transa)");
  CheckCublas(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
      "cublasLtMatmulDescSetAttribute(transb)");

  CheckCublas(cublasLtMatrixLayoutCreate(&a_desc, data_type, k, n, k), "cublasLtMatrixLayoutCreate(A)");
  CheckCublas(cublasLtMatrixLayoutCreate(&b_desc, data_type, k, m, k), "cublasLtMatrixLayoutCreate(B)");
  CheckCublas(cublasLtMatrixLayoutCreate(&c_desc, data_type, n, m, n), "cublasLtMatrixLayoutCreate(C)");

  CheckCublas(cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate");
  CheckCublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_size,
          sizeof(workspace_size)),
      "cublasLtMatmulPreferenceSetAttribute(workspace)");

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      op_desc,
      a_desc,
      b_desc,
      c_desc,
      c_desc,
      preference,
      1,
      &heuristic,
      &returned_results);
  if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
    cleanup();
    return at::linear(x_2d, weight, bias);
  }

  CheckCublas(
      cublasLtMatmul(
          handle,
          op_desc,
          &alpha,
          weight.data_ptr(),
          a_desc,
          x_2d.data_ptr(),
          b_desc,
          &beta,
          out.data_ptr(),
          c_desc,
          out.data_ptr(),
          c_desc,
          &heuristic.algo,
          workspace,
          workspace_size,
          stream.stream()),
      "cublasLtMatmul");
  cleanup();

  if (bias.has_value() && bias.value().defined()) {
    out.add_(bias.value().view({1, n}).to(out.device(), out.scalar_type()));
  }
  return out;
}

}  // namespace

bool HasCublasLtLinearBackend() {
  return true;
}

torch::Tensor CublasLtLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  if (!x.is_cuda() || !weight.is_cuda()) {
    return at::linear(x, weight, bias);
  }
  if (!IsSupportedLinearDtype(x.scalar_type()) || x.scalar_type() != weight.scalar_type()) {
    return at::linear(x, weight, bias);
  }
  if (bias.has_value() && bias.value().defined()) {
    if (!bias.value().is_cuda() || bias.value().scalar_type() != x.scalar_type()) {
      return at::linear(x, weight, bias);
    }
  }
  TORCH_CHECK(x.dim() >= 2, "CublasLtLinearForward: x must have rank >= 2");
  TORCH_CHECK(weight.dim() == 2, "CublasLtLinearForward: weight must be rank-2");
  TORCH_CHECK(x.size(-1) == weight.size(1), "CublasLtLinearForward: input feature size mismatch");

  c10::cuda::CUDAGuard device_guard{x.device()};

  auto x_contig = x.contiguous();
  auto weight_contig = weight.contiguous();
  const auto k = x_contig.size(-1);
  const auto leading = x_contig.numel() / k;

  auto x_2d = x_contig.view({leading, k});
  auto out_2d = RunCublasLtLinear(x_2d, weight_contig, bias);

  std::vector<int64_t> out_sizes(x_contig.sizes().begin(), x_contig.sizes().end());
  out_sizes.back() = weight_contig.size(0);
  return out_2d.view(out_sizes);
}
