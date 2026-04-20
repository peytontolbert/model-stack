#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/linear.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cublasLt.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
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

bool IsSupportedInt8LinearOutDtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

bool Int8CublasLtDisabled() {
  const char* env = std::getenv("MODEL_STACK_DISABLE_INT8_LINEAR_CUBLASLT");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

struct CachedInt8AccumMatmulPlan {
  int device_index = -1;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  size_t workspace_size = 0;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};

  ~CachedInt8AccumMatmulPlan() {
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
  }

  bool Matches(int device, int64_t m_value, int64_t n_value, int64_t k_value, size_t workspace_value) const {
    return device_index == device && m == m_value && n == n_value && k == k_value && workspace_size == workspace_value;
  }
};

struct CachedInt8AccumScratch {
  int device_index = -1;
  std::uintptr_t stream_key = 0;
  int64_t m = 0;
  int64_t n = 0;
  torch::Tensor accum;

  bool Matches(int device, std::uintptr_t stream, int64_t m_value, int64_t n_value) const {
    return device_index == device && stream_key == stream && m == m_value && n == n_value && accum.defined();
  }
};

CachedInt8AccumMatmulPlan* GetOrCreateCachedInt8AccumMatmulPlan(
    cublasLtHandle_t handle,
    int device_index,
    int64_t m,
    int64_t n,
    int64_t k,
    size_t workspace_size) {
  thread_local std::vector<std::unique_ptr<CachedInt8AccumMatmulPlan>> plans;
  for (auto& plan_ptr : plans) {
    if (plan_ptr != nullptr && plan_ptr->Matches(device_index, m, n, k, workspace_size)) {
      return plan_ptr.get();
    }
  }

  auto plan = std::make_unique<CachedInt8AccumMatmulPlan>();
  plan->device_index = device_index;
  plan->m = m;
  plan->n = n;
  plan->k = k;
  plan->workspace_size = workspace_size;

  const auto scale_type = CUDA_R_32I;
  const auto data_type = CUDA_R_8I;
  const auto accum_type = CUDA_R_32I;
  const auto compute_type = CUBLAS_COMPUTE_32I;
  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;

  auto reset_plan = [&]() {
    plan.reset();
    return static_cast<CachedInt8AccumMatmulPlan*>(nullptr);
  };

  if (cublasLtMatmulDescCreate(&plan->op_desc, compute_type, scale_type) != CUBLAS_STATUS_SUCCESS) {
    return reset_plan();
  }
  if (cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
          CUBLAS_STATUS_SUCCESS ||
      cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
          CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&plan->a_desc, data_type, k, n, k) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&plan->b_desc, data_type, k, m, k) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&plan->c_desc, accum_type, n, m, n) != CUBLAS_STATUS_SUCCESS) {
    return reset_plan();
  }

  cublasLtMatmulPreference_t preference = nullptr;
  const bool preference_ok =
      cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_size,
          sizeof(workspace_size)) == CUBLAS_STATUS_SUCCESS;
  if (!preference_ok) {
    if (preference != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference);
    }
    return reset_plan();
  }

  int returned_results = 0;
  const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      plan->op_desc,
      plan->a_desc,
      plan->b_desc,
      plan->c_desc,
      plan->c_desc,
      preference,
      1,
      &plan->heuristic,
      &returned_results);
  cublasLtMatmulPreferenceDestroy(preference);
  if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
    return reset_plan();
  }

  auto* out = plan.get();
  plans.push_back(std::move(plan));
  return out;
}

torch::Tensor GetOrCreateCachedInt8AccumScratch(
    const torch::Tensor& reference,
    int64_t m,
    int64_t n,
    cudaStream_t stream) {
  thread_local std::vector<std::unique_ptr<CachedInt8AccumScratch>> scratch_cache;
  const auto device_index = reference.get_device();
  const auto stream_key = reinterpret_cast<std::uintptr_t>(stream);
  for (auto& entry : scratch_cache) {
    if (entry != nullptr && entry->Matches(device_index, stream_key, m, n)) {
      return entry->accum;
    }
  }

  auto entry = std::make_unique<CachedInt8AccumScratch>();
  entry->device_index = device_index;
  entry->stream_key = stream_key;
  entry->m = m;
  entry->n = n;
  entry->accum = torch::empty({m, n}, reference.options().dtype(torch::kInt32));
  auto out = entry->accum;
  scratch_cache.push_back(std::move(entry));
  return out;
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

template <typename scalar_t>
__global__ void int8_linear_rescale_epilogue_kernel(
    const int32_t* __restrict__ accum,
    const float* __restrict__ x_scale,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * out_features;
  if (idx >= total) {
    return;
  }
  const int64_t row = idx / out_features;
  const int64_t col = idx % out_features;
  float value = static_cast<float>(accum[idx]) * x_scale[row] * inv_scale[col];
  if (bias != nullptr) {
    value += static_cast<float>(bias[col]);
  }
  out[idx] = static_cast<scalar_t>(value);
}

template <typename scalar_t, int ColsPerThread>
__global__ void int8_linear_row1_rescale_epilogue_kernel(
    const int32_t* __restrict__ accum,
    const float* __restrict__ x_scale,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t out_features) {
  const float row_scale = x_scale[0];
  const int64_t col0 =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * ColsPerThread;
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t col = col0 + idx;
    if (col >= out_features) {
      return;
    }
    float value = static_cast<float>(accum[col]) * row_scale * inv_scale[col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[col]);
    }
    out[col] = static_cast<scalar_t>(value);
  }
}

torch::Tensor RunCublasLtInt8LinearAccum(
    const torch::Tensor& qx_2d,
    const torch::Tensor& qweight) {
  const auto m = qx_2d.size(0);
  const auto k = qx_2d.size(1);
  const auto n = qweight.size(0);

  auto stream = c10::cuda::getCurrentCUDAStream(qx_2d.get_device());
  auto out = m == 1
      ? GetOrCreateCachedInt8AccumScratch(qx_2d, m, n, stream.stream())
      : torch::empty({m, n}, qx_2d.options().dtype(torch::kInt32));
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto workspace = at::cuda::getCUDABlasLtWorkspace();
  auto workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();
  const auto compute_type = CUBLAS_COMPUTE_32I;
  const int32_t alpha = 1;
  const int32_t beta = 0;
  auto* plan = GetOrCreateCachedInt8AccumMatmulPlan(handle, qx_2d.get_device(), m, n, k, workspace_size);
  if (plan == nullptr) {
    return torch::Tensor();
  }

  const auto matmul_status = cublasLtMatmul(
      handle,
      plan->op_desc,
      &alpha,
      qweight.data_ptr(),
      plan->a_desc,
      qx_2d.data_ptr(),
      plan->b_desc,
      &beta,
      out.data_ptr(),
      plan->c_desc,
      out.data_ptr(),
      plan->c_desc,
      &plan->heuristic.algo,
      workspace,
      workspace_size,
      stream.stream());
  if (matmul_status != CUBLAS_STATUS_SUCCESS) {
    return torch::Tensor();
  }
  return out;
}

}  // namespace

bool HasCublasLtLinearBackend() {
  return true;
}

torch::Tensor CublasLtInt8LinearForward(
    const torch::Tensor& qx,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  if (Int8CublasLtDisabled()) {
    return torch::Tensor();
  }
  if (!qx.is_cuda() || !x_scale.is_cuda() || !qweight.is_cuda() || !inv_scale.is_cuda()) {
    return torch::Tensor();
  }
  if (qx.scalar_type() != torch::kInt8 || qweight.scalar_type() != torch::kInt8) {
    return torch::Tensor();
  }
  if (x_scale.scalar_type() != torch::kFloat32 || inv_scale.scalar_type() != torch::kFloat32) {
    return torch::Tensor();
  }
  const auto output_dtype = out_dtype.has_value() ? out_dtype.value() : torch::kFloat32;
  if (!IsSupportedInt8LinearOutDtype(output_dtype)) {
    return torch::Tensor();
  }
  if (bias.has_value() && bias.value().defined()) {
    if (!bias.value().is_cuda()) {
      return torch::Tensor();
    }
  }

  c10::cuda::CUDAGuard device_guard{qx.device()};

  auto qx_contig = qx.contiguous();
  auto x_scale_contig = x_scale.reshape({-1}).contiguous();
  auto qweight_contig = qweight.contiguous();
  auto inv_scale_contig = inv_scale.contiguous();
  const auto in_features = qx_contig.size(-1);
  const auto rows = qx_contig.numel() / in_features;
  const auto out_features = qweight_contig.size(0);
  if (rows == 0 || out_features == 0) {
    return torch::empty(
        {rows, out_features},
        qx_contig.options().dtype(output_dtype));
  }
  if (x_scale_contig.numel() != rows || qweight_contig.size(1) != in_features || inv_scale_contig.size(0) != out_features) {
    return torch::Tensor();
  }

  auto qx_2d = qx_contig.view({rows, in_features});
  auto accum = RunCublasLtInt8LinearAccum(qx_2d, qweight_contig);
  if (!accum.defined()) {
    return torch::Tensor();
  }

  auto out = torch::empty({rows, out_features}, qx_contig.options().dtype(output_dtype));
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    bias_cast = bias.value().to(qx_contig.device(), output_dtype).contiguous();
  }
  auto stream = c10::cuda::getCurrentCUDAStream(qx.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output_dtype,
      "model_stack_cublaslt_int8_linear_epilogue",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias_cast.has_value() && bias_cast.value().defined()) {
          bias_ptr = bias_cast.value().data_ptr<scalar_t>();
        }
        if (rows == 1) {
          constexpr int kRow1Threads = 128;
          constexpr int kRow1ColsPerThread = 4;
          const dim3 threads(kRow1Threads);
          const dim3 blocks(static_cast<unsigned int>(
              (out_features + (kRow1Threads * kRow1ColsPerThread) - 1) /
              (kRow1Threads * kRow1ColsPerThread)));
          int8_linear_row1_rescale_epilogue_kernel<scalar_t, kRow1ColsPerThread><<<
              blocks,
              threads,
              0,
              stream.stream()>>>(
              accum.data_ptr<int32_t>(),
              x_scale_contig.data_ptr<float>(),
              inv_scale_contig.data_ptr<float>(),
              bias_ptr,
              out.data_ptr<scalar_t>(),
              out_features);
        } else {
          constexpr int kThreads = 256;
          const dim3 threads(kThreads);
          const dim3 blocks(static_cast<unsigned int>((rows * out_features + kThreads - 1) / kThreads));
          int8_linear_rescale_epilogue_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
              accum.data_ptr<int32_t>(),
              x_scale_contig.data_ptr<float>(),
              inv_scale_contig.data_ptr<float>(),
              bias_ptr,
              out.data_ptr<scalar_t>(),
              rows,
              out_features);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_sizes(qx_contig.sizes().begin(), qx_contig.sizes().end());
  out_sizes.back() = out_features;
  return out.view(out_sizes);
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
