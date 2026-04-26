#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/linear.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
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

bool Int8CublasLtAutotuneEnabled() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_AUTOTUNE");
  return env == nullptr || env[0] == '\0' || env[0] != '0';
}

bool Int8CublasLtRowMajorDirectEnabled() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_ROW_MAJOR");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

int Int8CublasLtRequestedAlgoCount() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_ALGO_COUNT");
  if (env == nullptr || env[0] == '\0') {
    return 16;
  }
  return std::max(1, std::min(64, static_cast<int>(std::strtol(env, nullptr, 10))));
}

int Int8CublasLtAutotuneRepeats() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_AUTOTUNE_REPEATS");
  if (env == nullptr || env[0] == '\0') {
    return 3;
  }
  return std::max(1, std::min(10, static_cast<int>(std::strtol(env, nullptr, 10))));
}

bool Int8CublasLtCustomFindEnabled() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_CUSTOM_FIND");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

int Int8CublasLtCustomFindMaxCandidates() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_CUSTOM_FIND_MAX");
  if (env == nullptr || env[0] == '\0') {
    return 256;
  }
  return std::max(1, std::min(4096, static_cast<int>(std::strtol(env, nullptr, 10))));
}

int Int8CublasLtCustomFindAlgoIds() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_CUSTOM_FIND_ALGO_IDS");
  if (env == nullptr || env[0] == '\0') {
    return 64;
  }
  return std::max(1, std::min(256, static_cast<int>(std::strtol(env, nullptr, 10))));
}

int Int8CublasLtEpilogueThreads() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_EPILOGUE_THREADS");
  if (env == nullptr || env[0] == '\0') {
    return 256;
  }
  return std::max(32, std::min(1024, static_cast<int>(std::strtol(env, nullptr, 10))));
}

int Int8CublasLtEpilogueColsPerThread() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_EPILOGUE_COLS_PER_THREAD");
  if (env == nullptr || env[0] == '\0') {
    return 2;
  }
  return std::max(1, std::min(16, static_cast<int>(std::strtol(env, nullptr, 10))));
}

bool Int8CublasLtSplitScaleEnabled() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_SPLIT_SCALE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool Int8CublasLtDebugEnabled() {
  const char* env = std::getenv("MODEL_STACK_INT8_LINEAR_CUBLASLT_DEBUG");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

struct CachedInt8AccumMatmulPlan {
  int device_index = -1;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  size_t workspace_size = 0;
  bool row_major_direct = false;
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

  bool Matches(
      int device,
      int64_t m_value,
      int64_t n_value,
      int64_t k_value,
      size_t workspace_value,
      bool row_major_value) const {
    return device_index == device && m == m_value && n == n_value && k == k_value &&
        workspace_size == workspace_value && row_major_direct == row_major_value;
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

struct CachedInt8ScaledMatmulPlan {
  int device_index = -1;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  size_t workspace_size = 0;
  torch::ScalarType output_dtype = torch::kFloat32;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};

  ~CachedInt8ScaledMatmulPlan() {
    if (d_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(d_desc);
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

  bool Matches(
      int device,
      int64_t m_value,
      int64_t n_value,
      int64_t k_value,
      size_t workspace_value,
      torch::ScalarType dtype_value) const {
    return device_index == device && m == m_value && n == n_value && k == k_value &&
        workspace_size == workspace_value && output_dtype == dtype_value;
  }
};

CachedInt8AccumMatmulPlan* GetOrCreateCachedInt8AccumMatmulPlan(
    cublasLtHandle_t handle,
    int device_index,
    int64_t m,
    int64_t n,
    int64_t k,
    size_t workspace_size,
    const torch::Tensor& qx_2d,
    const torch::Tensor& qweight,
    const torch::Tensor& out,
    void* workspace,
    cudaStream_t stream) {
  thread_local std::vector<std::unique_ptr<CachedInt8AccumMatmulPlan>> plans;
  const bool row_major_direct = Int8CublasLtRowMajorDirectEnabled();
  for (auto& plan_ptr : plans) {
    if (plan_ptr != nullptr && plan_ptr->Matches(device_index, m, n, k, workspace_size, row_major_direct)) {
      return plan_ptr.get();
    }
  }

  auto plan = std::make_unique<CachedInt8AccumMatmulPlan>();
  plan->device_index = device_index;
  plan->m = m;
  plan->n = n;
  plan->k = k;
  plan->workspace_size = workspace_size;
  plan->row_major_direct = row_major_direct;

  const auto scale_type = CUDA_R_32I;
  const auto data_type = CUDA_R_8I;
  const auto accum_type = CUDA_R_32I;
  const auto compute_type = CUBLAS_COMPUTE_32I;
  cublasOperation_t trans_a = row_major_direct ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t trans_b = row_major_direct ? CUBLAS_OP_T : CUBLAS_OP_N;

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
          CUBLAS_STATUS_SUCCESS) {
    return reset_plan();
  }
  if (row_major_direct) {
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    if (cublasLtMatrixLayoutCreate(&plan->a_desc, data_type, m, k, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan->b_desc, data_type, n, k, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan->c_desc, accum_type, m, n, n) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutSetAttribute(
            plan->a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutSetAttribute(
            plan->b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutSetAttribute(
            plan->c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS) {
      return reset_plan();
    }
  } else {
    if (cublasLtMatrixLayoutCreate(&plan->a_desc, data_type, k, n, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan->b_desc, data_type, k, m, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan->c_desc, accum_type, n, m, n) != CUBLAS_STATUS_SUCCESS) {
      return reset_plan();
    }
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

  const int requested_algo_count = Int8CublasLtRequestedAlgoCount();
  std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
      static_cast<size_t>(requested_algo_count));
  int returned_results = 0;
  const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      plan->op_desc,
      plan->a_desc,
      plan->b_desc,
      plan->c_desc,
      plan->c_desc,
      preference,
      requested_algo_count,
      heuristic_results.data(),
      &returned_results);
  cublasLtMatmulPreferenceDestroy(preference);
  if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
    return reset_plan();
  }

  const int32_t alpha = 1;
  const int32_t beta = 0;
  const void* matmul_a = row_major_direct ? qx_2d.data_ptr() : qweight.data_ptr();
  const void* matmul_b = row_major_direct ? qweight.data_ptr() : qx_2d.data_ptr();

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  auto ensure_events = [&]() {
    if (start_event == nullptr) {
      C10_CUDA_CHECK(cudaEventCreate(&start_event));
    }
    if (stop_event == nullptr) {
      C10_CUDA_CHECK(cudaEventCreate(&stop_event));
    }
  };
  auto destroy_events = [&]() {
    if (stop_event != nullptr) {
      C10_CUDA_CHECK(cudaEventDestroy(stop_event));
      stop_event = nullptr;
    }
    if (start_event != nullptr) {
      C10_CUDA_CHECK(cudaEventDestroy(start_event));
      start_event = nullptr;
    }
  };
  auto time_algo = [&](const cublasLtMatmulAlgo_t& algo, int repeats, float* avg_ms) {
    ensure_events();
    float total_ms = 0.0f;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      C10_CUDA_CHECK(cudaEventRecord(start_event, stream));
      const auto status = cublasLtMatmul(
          handle,
          plan->op_desc,
          &alpha,
          matmul_a,
          plan->a_desc,
          matmul_b,
          plan->b_desc,
          &beta,
          out.data_ptr(),
          plan->c_desc,
          out.data_ptr(),
          plan->c_desc,
          &algo,
          workspace,
          workspace_size,
          stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      C10_CUDA_CHECK(cudaEventRecord(stop_event, stream));
      C10_CUDA_CHECK(cudaEventSynchronize(stop_event));
      float elapsed_ms = 0.0f;
      C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
      total_ms += elapsed_ms;
    }
    *avg_ms = total_ms / static_cast<float>(repeats);
    return true;
  };

  cublasLtMatmulHeuristicResult_t best_result = heuristic_results[0];
  float best_ms = std::numeric_limits<float>::infinity();
  if (Int8CublasLtAutotuneEnabled() && returned_results > 1) {
    const int repeats = Int8CublasLtAutotuneRepeats();
    for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
      float candidate_ms = 0.0f;
      const bool ok = time_algo(heuristic_results[algo_idx].algo, repeats, &candidate_ms);
      if (ok) {
        if (candidate_ms < best_ms) {
          best_result = heuristic_results[algo_idx];
          best_ms = candidate_ms;
        }
      }
    }
  }
  if (Int8CublasLtCustomFindEnabled()) {
    if (!std::isfinite(best_ms)) {
      float candidate_ms = 0.0f;
      if (time_algo(best_result.algo, Int8CublasLtAutotuneRepeats(), &candidate_ms)) {
        best_ms = candidate_ms;
      }
    }

    const int requested_algo_ids = Int8CublasLtCustomFindAlgoIds();
    std::vector<int> algo_ids(static_cast<size_t>(requested_algo_ids));
    int returned_algo_ids = 0;
    const auto get_ids_status = cublasLtMatmulAlgoGetIds(
        handle,
        compute_type,
        scale_type,
        data_type,
        data_type,
        accum_type,
        accum_type,
        requested_algo_ids,
        algo_ids.data(),
        &returned_algo_ids);
    const int splitk_values[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
    int checked_candidates = 0;
    if (get_ids_status == CUBLAS_STATUS_SUCCESS && returned_algo_ids > 0) {
      int device = 0;
      C10_CUDA_CHECK(cudaGetDevice(&device));
      int cluster_launch_supported = 0;
      C10_CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch_supported, cudaDevAttrClusterLaunch, device));
      const uint16_t cluster_shape_end =
          cluster_launch_supported ? CUBLASLT_CLUSTER_SHAPE_END : (CUBLASLT_CLUSTER_SHAPE_AUTO + 1);
      const int max_candidates = Int8CublasLtCustomFindMaxCandidates();
      const int repeats = Int8CublasLtAutotuneRepeats();
      for (int algo_id_idx = 0; algo_id_idx < returned_algo_ids && checked_candidates < max_candidates; ++algo_id_idx) {
        cublasLtMatmulAlgo_t algo{};
        if (cublasLtMatmulAlgoInit(
                handle,
                compute_type,
                scale_type,
                data_type,
                data_type,
                accum_type,
                accum_type,
                algo_ids[algo_id_idx],
                &algo) != CUBLAS_STATUS_SUCCESS) {
          continue;
        }

        size_t size_written = 0;
        std::vector<int> tile_ids;
        if (cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &size_written) == CUBLAS_STATUS_SUCCESS &&
            size_written > 0) {
          tile_ids.resize(size_written / sizeof(int));
          cublasLtMatmulAlgoCapGetAttribute(
              &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tile_ids.data(), size_written, &size_written);
        }
        if (tile_ids.empty()) {
          tile_ids.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
        }

        size_written = 0;
        std::vector<int> stage_ids;
        if (cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &size_written) == CUBLAS_STATUS_SUCCESS &&
            size_written > 0) {
          stage_ids.resize(size_written / sizeof(int));
          cublasLtMatmulAlgoCapGetAttribute(
              &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stage_ids.data(), size_written, &size_written);
        }
        if (stage_ids.empty()) {
          stage_ids.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
        }

        int splitk_support = 0;
        int reduction_mask = 0;
        int swizzling_max = 0;
        int custom_option_max = 0;
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_support, sizeof(splitk_support), &size_written);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &reduction_mask, sizeof(reduction_mask), &size_written);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzling_max, sizeof(swizzling_max), &size_written);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_option_max, sizeof(custom_option_max), &size_written);

        for (int tile_id : tile_ids) {
          if (cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(tile_id)) != CUBLAS_STATUS_SUCCESS) {
            continue;
          }
          for (int stage_id : stage_ids) {
            if (cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stage_id, sizeof(stage_id)) != CUBLAS_STATUS_SUCCESS) {
              continue;
            }
            for (uint16_t cluster_shape = CUBLASLT_CLUSTER_SHAPE_AUTO;
                 cluster_shape < cluster_shape_end && checked_candidates < max_candidates;
                 ++cluster_shape) {
              if (cublasLtMatmulAlgoConfigSetAttribute(
                      &algo,
                      CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
                      &cluster_shape,
                      sizeof(cluster_shape)) != CUBLAS_STATUS_SUCCESS) {
                continue;
              }
              for (int custom_option = 0;
                   custom_option <= custom_option_max && checked_candidates < max_candidates;
                   ++custom_option) {
                if (cublasLtMatmulAlgoConfigSetAttribute(
                        &algo,
                        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                        &custom_option,
                        sizeof(custom_option)) != CUBLAS_STATUS_SUCCESS) {
                  continue;
                }
                for (int swizzle = 0; swizzle <= swizzling_max && checked_candidates < max_candidates; ++swizzle) {
                  if (cublasLtMatmulAlgoConfigSetAttribute(
                          &algo,
                          CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                          &swizzle,
                          sizeof(swizzle)) != CUBLAS_STATUS_SUCCESS) {
                    continue;
                  }

                  std::vector<std::pair<int, int>> splitk_reduction_pairs;
                  splitk_reduction_pairs.push_back({0, CUBLASLT_REDUCTION_SCHEME_NONE});
                  if (splitk_support) {
                    for (int splitk : splitk_values) {
                      for (int reduction = 1; reduction < static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK);
                           reduction <<= 1) {
                        if ((reduction & reduction_mask) != 0) {
                          splitk_reduction_pairs.push_back({splitk, reduction});
                        }
                      }
                    }
                  }

                  for (const auto& splitk_reduction : splitk_reduction_pairs) {
                    if (checked_candidates >= max_candidates) {
                      break;
                    }
                    const int splitk = splitk_reduction.first;
                    const int reduction = splitk_reduction.second;
                    if (cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk)) !=
                            CUBLAS_STATUS_SUCCESS ||
                        cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction)) !=
                            CUBLAS_STATUS_SUCCESS) {
                      continue;
                    }
                    ++checked_candidates;
                    cublasLtMatmulHeuristicResult_t checked_result{};
                    const auto check_status = cublasLtMatmulAlgoCheck(
                        handle,
                        plan->op_desc,
                        plan->a_desc,
                        plan->b_desc,
                        plan->c_desc,
                        plan->c_desc,
                        &algo,
                        &checked_result);
                    if (check_status != CUBLAS_STATUS_SUCCESS || checked_result.workspaceSize > workspace_size) {
                      continue;
                    }
                    float candidate_ms = 0.0f;
                    if (time_algo(algo, repeats, &candidate_ms) && candidate_ms < best_ms) {
                      checked_result.algo = algo;
                      best_result = checked_result;
                      best_ms = candidate_ms;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  destroy_events();
  plan->heuristic = best_result;

  auto* plan_out = plan.get();
  plans.push_back(std::move(plan));
  return plan_out;
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

CachedInt8ScaledMatmulPlan* GetOrCreateCachedInt8ScaledMatmulPlan(
    cublasLtHandle_t handle,
    int device_index,
    int64_t m,
    int64_t n,
    int64_t k,
    size_t workspace_size,
    torch::ScalarType output_dtype) {
  thread_local std::vector<std::unique_ptr<CachedInt8ScaledMatmulPlan>> plans;
  for (auto& plan_ptr : plans) {
    if (plan_ptr != nullptr && plan_ptr->Matches(device_index, m, n, k, workspace_size, output_dtype)) {
      return plan_ptr.get();
    }
  }

  auto plan = std::make_unique<CachedInt8ScaledMatmulPlan>();
  plan->device_index = device_index;
  plan->m = m;
  plan->n = n;
  plan->k = k;
  plan->workspace_size = workspace_size;
  plan->output_dtype = output_dtype;

  auto reset_plan = [&]() {
    plan.reset();
    return static_cast<CachedInt8ScaledMatmulPlan*>(nullptr);
  };

  const auto scale_type = CUDA_R_32F;
  const auto data_type = CUDA_R_8I;
  const auto output_data_type = ToCudaDataType(output_dtype);
  const auto compute_type = CUBLAS_COMPUTE_32I;
  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;

  auto status = cublasLtMatmulDescCreate(&plan->op_desc, compute_type, scale_type);
  if (status != CUBLAS_STATUS_SUCCESS) {
    if (Int8CublasLtDebugEnabled()) {
      TORCH_WARN(
          "cublasLt int8 split-scale desc create failed: ",
          CublasStatusString(status),
          " m=",
          m,
          " n=",
          n,
          " k=",
          k);
    }
    return reset_plan();
  }
  status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(&plan->a_desc, data_type, k, n, k);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(&plan->b_desc, data_type, k, m, k);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(&plan->d_desc, output_data_type, n, m, n);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    if (Int8CublasLtDebugEnabled()) {
      TORCH_WARN(
          "cublasLt int8 split-scale descriptor setup failed: ",
          CublasStatusString(status),
          " m=",
          m,
          " n=",
          n,
          " k=",
          k,
          " output_dtype=",
          static_cast<int>(output_dtype));
    }
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
    if (Int8CublasLtDebugEnabled()) {
      TORCH_WARN(
          "cublasLt int8 split-scale preference setup failed: m=",
          m,
          " n=",
          n,
          " k=",
          k,
          " workspace_size=",
          workspace_size);
    }
    return reset_plan();
  }

  const int requested_algo_count = Int8CublasLtRequestedAlgoCount();
  std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
      static_cast<size_t>(requested_algo_count));
  int returned_results = 0;
  const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      plan->op_desc,
      plan->a_desc,
      plan->b_desc,
      plan->d_desc,
      plan->d_desc,
      preference,
      requested_algo_count,
      heuristic_results.data(),
      &returned_results);
  cublasLtMatmulPreferenceDestroy(preference);
  if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
    if (Int8CublasLtDebugEnabled()) {
      TORCH_WARN(
          "cublasLt int8 split-scale heuristic failed: ",
          CublasStatusString(heuristic_status),
          " returned_results=",
          returned_results,
          " requested=",
          requested_algo_count,
          " m=",
          m,
          " n=",
          n,
          " k=",
          k,
          " output_dtype=",
          static_cast<int>(output_dtype));
    }
    return reset_plan();
  }

  plan->heuristic = heuristic_results[0];
  auto* plan_out = plan.get();
  plans.push_back(std::move(plan));
  return plan_out;
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
__global__ void int8_linear_rescale_epilogue_2d_kernel(
    const int32_t* __restrict__ accum,
    const float* __restrict__ x_scale,
    const float* __restrict__ inv_scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t out_features) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const float row_scale = x_scale[row];
  const int64_t col0 =
      (static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x) * ColsPerThread;
  const int64_t base = row * out_features;
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t col = col0 + idx;
    if (col >= out_features) {
      return;
    }
    float value = static_cast<float>(accum[base + col]) * row_scale * inv_scale[col];
    if (bias != nullptr) {
      value += static_cast<float>(bias[col]);
    }
    out[base + col] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t, int ColsPerThread>
__global__ void int8_linear_apply_row_scale_2d_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ x_scale,
    const scalar_t* __restrict__ bias,
    int64_t rows,
    int64_t out_features) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const float row_scale = x_scale[row];
  const int64_t col0 =
      (static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x) * ColsPerThread;
  const int64_t base = row * out_features;
  #pragma unroll
  for (int idx = 0; idx < ColsPerThread; ++idx) {
    const int64_t col = col0 + idx;
    if (col >= out_features) {
      return;
    }
    float value = static_cast<float>(out[base + col]) * row_scale;
    if (bias != nullptr) {
      value += static_cast<float>(bias[col]);
    }
    out[base + col] = static_cast<scalar_t>(value);
  }
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
  const int32_t alpha = 1;
  const int32_t beta = 0;
  auto* plan = GetOrCreateCachedInt8AccumMatmulPlan(
      handle,
      qx_2d.get_device(),
      m,
      n,
      k,
      workspace_size,
      qx_2d,
      qweight,
      out,
      workspace,
      stream.stream());
  if (plan == nullptr) {
    return torch::Tensor();
  }

  const void* matmul_a = plan->row_major_direct ? qx_2d.data_ptr() : qweight.data_ptr();
  const void* matmul_b = plan->row_major_direct ? qweight.data_ptr() : qx_2d.data_ptr();
  const auto matmul_status = cublasLtMatmul(
      handle,
      plan->op_desc,
      &alpha,
      matmul_a,
      plan->a_desc,
      matmul_b,
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

torch::Tensor RunCublasLtInt8LinearColumnScaled(
    const torch::Tensor& qx_2d,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    torch::ScalarType output_dtype) {
  const auto m = qx_2d.size(0);
  const auto k = qx_2d.size(1);
  const auto n = qweight.size(0);

  auto out = torch::empty({m, n}, qx_2d.options().dtype(output_dtype));
  auto stream = c10::cuda::getCurrentCUDAStream(qx_2d.get_device());
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto workspace = at::cuda::getCUDABlasLtWorkspace();
  auto workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();
  const float beta = 0.0f;
  auto* plan = GetOrCreateCachedInt8ScaledMatmulPlan(
      handle,
      qx_2d.get_device(),
      m,
      n,
      k,
      workspace_size,
      output_dtype);
  if (plan == nullptr) {
    return torch::Tensor();
  }

  const auto matmul_status = cublasLtMatmul(
      handle,
      plan->op_desc,
      inv_scale.data_ptr<float>(),
      qweight.data_ptr(),
      plan->a_desc,
      qx_2d.data_ptr(),
      plan->b_desc,
      &beta,
      out.data_ptr(),
      plan->d_desc,
      out.data_ptr(),
      plan->d_desc,
      &plan->heuristic.algo,
      workspace,
      workspace_size,
      stream.stream());
  if (matmul_status != CUBLAS_STATUS_SUCCESS) {
    if (Int8CublasLtDebugEnabled()) {
      TORCH_WARN(
          "cublasLt int8 split-scale matmul failed: ",
          CublasStatusString(matmul_status),
          " m=",
          m,
          " n=",
          n,
          " k=",
          k,
          " output_dtype=",
          static_cast<int>(output_dtype));
    }
    return torch::Tensor();
  }
  if (Int8CublasLtDebugEnabled()) {
    TORCH_WARN(
        "cublasLt int8 split-scale matmul used: m=",
        m,
        " n=",
        n,
        " k=",
        k,
        " output_dtype=",
        static_cast<int>(output_dtype));
  }
  return out;
}

}  // namespace

bool HasCublasLtLinearBackend() {
  return true;
}

torch::Tensor CublasLtInt8LinearAccumForward(
    const torch::Tensor& qx,
    const torch::Tensor& qweight) {
  if (Int8CublasLtDisabled()) {
    return torch::Tensor();
  }
  if (!qx.is_cuda() || !qweight.is_cuda()) {
    return torch::Tensor();
  }
  if (qx.scalar_type() != torch::kInt8 || qweight.scalar_type() != torch::kInt8) {
    return torch::Tensor();
  }
  TORCH_CHECK(qx.dim() >= 2, "CublasLtInt8LinearAccumForward: qx must have rank >= 2");
  TORCH_CHECK(qweight.dim() == 2, "CublasLtInt8LinearAccumForward: qweight must be rank-2");

  c10::cuda::CUDAGuard device_guard{qx.device()};

  auto qx_contig = qx.contiguous();
  auto qweight_contig = qweight.contiguous();
  const auto in_features = qx_contig.size(-1);
  const auto rows = qx_contig.numel() / in_features;
  const auto out_features = qweight_contig.size(0);
  TORCH_CHECK(
      qweight_contig.size(1) == in_features,
      "CublasLtInt8LinearAccumForward: qweight column count mismatch");
  if (rows == 0 || out_features == 0) {
    std::vector<int64_t> out_sizes(qx_contig.sizes().begin(), qx_contig.sizes().end());
    out_sizes.back() = out_features;
    return torch::empty(out_sizes, qx_contig.options().dtype(torch::kInt32));
  }

  auto qx_2d = qx_contig.view({rows, in_features});
  auto accum = RunCublasLtInt8LinearAccum(qx_2d, qweight_contig);
  if (!accum.defined()) {
    return torch::Tensor();
  }
  std::vector<int64_t> out_sizes(qx_contig.sizes().begin(), qx_contig.sizes().end());
  out_sizes.back() = out_features;
  return accum.view(out_sizes);
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
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    bias_cast = bias.value().to(qx_contig.device(), output_dtype).contiguous();
  }
  auto stream = c10::cuda::getCurrentCUDAStream(qx.get_device());
  if (Int8CublasLtSplitScaleEnabled() && output_dtype != torch::kFloat32) {
    auto out = RunCublasLtInt8LinearColumnScaled(qx_2d, qweight_contig, inv_scale_contig, output_dtype);
    if (out.defined()) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          output_dtype,
          "model_stack_cublaslt_int8_linear_row_scale_epilogue",
          [&] {
            const scalar_t* bias_ptr = nullptr;
            if (bias_cast.has_value() && bias_cast.value().defined()) {
              bias_ptr = bias_cast.value().data_ptr<scalar_t>();
            }
            constexpr int kThreads = 256;
            constexpr int kColsPerThread = 2;
            const dim3 threads(kThreads);
            const dim3 blocks(
                static_cast<unsigned int>(rows),
                static_cast<unsigned int>(
                    (out_features + (kThreads * kColsPerThread) - 1) /
                    (kThreads * kColsPerThread)));
            int8_linear_apply_row_scale_2d_kernel<scalar_t, kColsPerThread><<<
                blocks,
                threads,
                0,
                stream.stream()>>>(
                out.data_ptr<scalar_t>(),
                x_scale_contig.data_ptr<float>(),
                bias_ptr,
                rows,
                out_features);
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      std::vector<int64_t> out_sizes(qx_contig.sizes().begin(), qx_contig.sizes().end());
      out_sizes.back() = out_features;
      return out.view(out_sizes);
    }
  }

  auto accum = RunCublasLtInt8LinearAccum(qx_2d, qweight_contig);
  if (!accum.defined()) {
    return torch::Tensor();
  }

  auto out = torch::empty({rows, out_features}, qx_contig.options().dtype(output_dtype));
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
          const int threads_value = Int8CublasLtEpilogueThreads();
          const int cols_per_thread_value = Int8CublasLtEpilogueColsPerThread();
          const dim3 threads(static_cast<unsigned int>(threads_value));
          auto launch_epilogue = [&](auto cols_per_thread_constant) {
            constexpr int kColsPerThread = decltype(cols_per_thread_constant)::value;
            const dim3 blocks(
                static_cast<unsigned int>(rows),
                static_cast<unsigned int>(
                    (out_features + (threads_value * kColsPerThread) - 1) /
                    (threads_value * kColsPerThread)));
            int8_linear_rescale_epilogue_2d_kernel<scalar_t, kColsPerThread><<<
                blocks,
                threads,
                0,
                stream.stream()>>>(
                accum.data_ptr<int32_t>(),
                x_scale_contig.data_ptr<float>(),
                inv_scale_contig.data_ptr<float>(),
                bias_ptr,
                out.data_ptr<scalar_t>(),
                rows,
                out_features);
          };
          if (cols_per_thread_value >= 16) {
            launch_epilogue(std::integral_constant<int, 16>{});
          } else if (cols_per_thread_value >= 8) {
            launch_epilogue(std::integral_constant<int, 8>{});
          } else if (cols_per_thread_value >= 4) {
            launch_epilogue(std::integral_constant<int, 4>{});
          } else if (cols_per_thread_value >= 2) {
            launch_epilogue(std::integral_constant<int, 2>{});
          } else {
            launch_epilogue(std::integral_constant<int, 1>{});
          }
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
