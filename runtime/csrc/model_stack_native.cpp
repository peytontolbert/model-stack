#include <torch/extension.h>
#include <ATen/TensorIndexing.h>
#include <ATen/ops/_unique2.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/silu.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#include "descriptors/attention_desc.h"
#include "policy/attention_policy.h"
#include "reference/aten_reference.h"

namespace py = pybind11;
namespace torch_ext = torch;

#if MODEL_STACK_WITH_CUDA
torch::Tensor CudaRmsNormForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    double eps);
bool HasCudaRmsNormKernel();
std::vector<torch::Tensor> CudaAddRmsNormForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& weight,
    double residual_scale,
    double eps);
bool HasCudaAddRmsNormKernel();
torch::Tensor CudaResidualAddForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    double residual_scale);
bool HasCudaResidualAddKernel();
torch::Tensor CudaLayerNormForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps);
bool HasCudaLayerNormKernel();
std::vector<torch::Tensor> CudaAddLayerNormForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double residual_scale,
    double eps);
bool HasCudaAddLayerNormKernel();
torch::Tensor CudaEmbeddingForward(
    const torch::Tensor& weight,
    const torch::Tensor& indices,
    int64_t padding_idx);
bool HasCudaEmbeddingKernel();
torch::Tensor CudaTemperatureForward(
    const torch::Tensor& logits,
    double tau);
torch::Tensor CudaTopkMaskForward(
    const torch::Tensor& logits,
    int64_t k);
torch::Tensor CudaToppMaskForward(
    const torch::Tensor& logits,
    double p);
torch::Tensor CudaApplySamplingMaskForward(
    const torch::Tensor& logits,
    const c10::optional<torch::Tensor>& topk_mask,
    const c10::optional<torch::Tensor>& topp_mask,
    const c10::optional<torch::Tensor>& no_repeat_mask);
torch::Tensor CudaPresenceFrequencyPenaltyForward(
    const torch::Tensor& logits,
    const torch::Tensor& counts,
    double alpha_presence,
    double alpha_frequency);
torch::Tensor CudaTokenCountsForward(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    torch::ScalarType counts_dtype);
torch::Tensor CudaNoRepeatNgramMaskForward(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    int64_t n);
torch::Tensor CudaRepetitionPenaltyForward(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    double penalty);
torch::Tensor CudaGreedyNextTokenForward(const torch::Tensor& logits);
torch::Tensor CudaMultinomialSampleForward(const torch::Tensor& logits);
torch::Tensor CudaSampleWithPoliciesForward(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    bool do_sample,
    double temperature,
    const c10::optional<int64_t>& top_k,
    const c10::optional<double>& top_p,
    int64_t no_repeat_ngram,
    double repetition_penalty,
    double presence_penalty,
    double frequency_penalty);
bool HasCudaSamplingKernel();
std::vector<torch::Tensor> CudaAppendTokensForward(
    const torch::Tensor& seq,
    const torch::Tensor& next_id,
    const c10::optional<torch::Tensor>& attention_mask);
bool HasCudaAppendTokensKernel();
std::vector<torch::Tensor> CudaDecodePositionsForward(
    int64_t batch_size,
    int64_t seq_len,
    const torch::Tensor& reference);
bool HasCudaDecodePositionsKernel();
torch::Tensor CublasLtLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias);
bool HasCublasLtLinearBackend();
torch::Tensor CudaActivationForward(
    const torch::Tensor& x,
    const std::string& activation);
bool HasCudaActivationKernel();
torch::Tensor CudaGatedActivationForward(
    const torch::Tensor& x,
    const std::string& activation);
bool HasCudaGatedActivationKernel();
std::vector<torch::Tensor> CudaApplyRotaryForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin);
bool HasCudaRopeKernel();
torch::Tensor CudaKvCacheWriteForward(
    const torch::Tensor& cache,
    const torch::Tensor& chunk,
    int64_t start);
torch::Tensor CudaKvCacheGatherForward(
    const torch::Tensor& cache,
    const torch::Tensor& positions);
torch::Tensor CudaPagedKvGatherForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions);
std::vector<torch::Tensor> CudaPagedKvReadLastForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t keep);
torch::Tensor CudaPagedKvReadRangeForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t start,
    int64_t end);
torch::Tensor CudaPagedKvWriteForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions,
    const torch::Tensor& values);
bool HasCudaKvCacheKernel();
torch::Tensor CudaAttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale);
bool HasCudaAttentionKernel();
#else
bool HasCudaRmsNormKernel() {
  return false;
}
bool HasCudaAddRmsNormKernel() {
  return false;
}
bool HasCudaResidualAddKernel() {
  return false;
}
bool HasCudaLayerNormKernel() {
  return false;
}
bool HasCudaAddLayerNormKernel() {
  return false;
}
bool HasCudaEmbeddingKernel() {
  return false;
}
bool HasCudaSamplingKernel() {
  return false;
}
bool HasCudaAppendTokensKernel() {
  return false;
}
bool HasCudaDecodePositionsKernel() {
  return false;
}
bool HasCublasLtLinearBackend() {
  return false;
}
bool HasCudaActivationKernel() {
  return false;
}
bool HasCudaGatedActivationKernel() {
  return false;
}
bool HasCudaRopeKernel() {
  return false;
}
bool HasCudaKvCacheKernel() {
  return false;
}
bool HasCudaAttentionKernel() {
  return false;
}
#endif

namespace {

constexpr int kAbiVersion = MODEL_STACK_ABI_VERSION;

std::string AttentionPhaseName(t10::desc::AttentionPhase phase) {
  switch (phase) {
    case t10::desc::AttentionPhase::kDecode:
      return "decode";
    default:
      return "prefill";
  }
}

std::string AttentionHeadModeName(t10::desc::AttentionHeadMode head_mode) {
  switch (head_mode) {
    case t10::desc::AttentionHeadMode::kMQA:
      return "mqa";
    case t10::desc::AttentionHeadMode::kGQA:
      return "gqa";
    default:
      return "mha";
  }
}

std::string AttentionMaskKindName(t10::desc::AttentionMaskKind mask_kind) {
  switch (mask_kind) {
    case t10::desc::AttentionMaskKind::kBool:
      return "bool";
    case t10::desc::AttentionMaskKind::kAdditiveSameDtype:
      return "additive_same_dtype";
    case t10::desc::AttentionMaskKind::kAdditiveFloat32:
      return "additive_float32";
    default:
      return "none";
  }
}

std::string AttentionKernelName(t10::policy::AttentionKernelKind kernel) {
  switch (kernel) {
    case t10::policy::AttentionKernelKind::kGenericPrefillNoMask:
      return "generic_prefill_nomask";
    case t10::policy::AttentionKernelKind::kPrefillMHANoMask:
      return "prefill_mha_nomask";
    case t10::policy::AttentionKernelKind::kPrefillMHA:
      return "prefill_mha";
    case t10::policy::AttentionKernelKind::kDecodeQ1MHANoMask:
      return "decode_q1_mha_nomask";
    case t10::policy::AttentionKernelKind::kDecodeQ1NoMask:
      return "decode_q1_nomask";
    case t10::policy::AttentionKernelKind::kDecodeQ1MHA:
      return "decode_q1_mha";
    case t10::policy::AttentionKernelKind::kPrefillHdim32:
      return "prefill_hdim32";
    case t10::policy::AttentionKernelKind::kPrefillHdim64:
      return "prefill_hdim64";
    case t10::policy::AttentionKernelKind::kPrefillHdim96:
      return "prefill_hdim96";
    case t10::policy::AttentionKernelKind::kPrefillHdim128:
      return "prefill_hdim128";
    case t10::policy::AttentionKernelKind::kDecodeQ1Hdim32:
      return "decode_q1_hdim32";
    case t10::policy::AttentionKernelKind::kDecodeQ1Hdim64:
      return "decode_q1_hdim64";
    case t10::policy::AttentionKernelKind::kDecodeQ1Hdim96:
      return "decode_q1_hdim96";
    case t10::policy::AttentionKernelKind::kDecodeQ1Hdim128:
      return "decode_q1_hdim128";
    case t10::policy::AttentionKernelKind::kDecodeQ1:
      return "decode_q1";
    case t10::policy::AttentionKernelKind::kGenericDecode:
      return "generic_decode";
    default:
      return "generic_prefill";
  }
}

bool CanUseCudaAttentionPath(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask) {
#if MODEL_STACK_WITH_CUDA
  if (!(q.is_cuda() && k.is_cuda() && v.is_cuda() && HasCudaAttentionKernel())) {
    return false;
  }
  if (!t10::desc::IsSupportedAttentionDtype(q.scalar_type())) {
    return false;
  }
  if (q.scalar_type() != k.scalar_type() || q.scalar_type() != v.scalar_type()) {
    return false;
  }
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    const auto& mask = attn_mask.value();
    if (!mask.is_cuda() || mask.dim() != 4) {
      return false;
    }
    if (!(mask.scalar_type() == torch::kBool ||
          mask.scalar_type() == q.scalar_type() ||
          mask.scalar_type() == torch::kFloat32)) {
      return false;
    }
    if (mask.size(0) != q.size(0) || mask.size(1) != q.size(1) ||
        mask.size(2) != q.size(2) || mask.size(3) != k.size(2)) {
      return false;
    }
  }
  return true;
#else
  (void)q;
  (void)k;
  (void)v;
  (void)attn_mask;
  return false;
#endif
}

std::map<std::string, bool> NativeOpMap() {
  return {
      {"activation", true},
      {"gated_activation", true},
      {"embedding", true},
      {"linear", true},
      {"pack_linear_weight", true},
      {"mlp", true},
      {"qkv_projection", true},
      {"pack_qkv_weights", true},
      {"qkv_packed_heads_projection", true},
      {"qkv_heads_projection", true},
      {"split_heads", true},
      {"merge_heads", true},
      {"head_output_projection", true},
      {"prepare_attention_mask", true},
      {"resolve_position_ids", true},
      {"create_causal_mask", true},
      {"resolve_rotary_embedding", true},
      {"token_counts", true},
      {"append_tokens", true},
      {"decode_positions", true},
      {"rms_norm", true},
      {"add_rms_norm", true},
      {"residual_add", true},
      {"layer_norm", true},
      {"add_layer_norm", true},
      {"rope", true},
      {"kv_cache_append", true},
      {"kv_cache_write", true},
      {"kv_cache_gather", true},
      {"paged_kv_assign_blocks", true},
      {"paged_kv_reserve_pages", true},
      {"paged_kv_read_range", true},
      {"paged_kv_read_last", true},
      {"paged_kv_append", true},
      {"paged_kv_compact", true},
      {"paged_kv_gather", true},
      {"paged_kv_write", true},
      {"attention_decode", true},
      {"attention_prefill", true},
      {"sampling", true},
      {"beam_search_step", true},
  };
}

std::vector<std::string> SupportedLinearBackends() {
#if MODEL_STACK_WITH_CUDA
  if (HasCublasLtLinearBackend()) {
    return {"aten", "cublaslt"};
  }
#endif
  return {"aten"};
}

std::vector<std::string> PlannedLinearBackends() {
  return HasCublasLtLinearBackend() ? std::vector<std::string>{} : std::vector<std::string>{"cublaslt"};
}

std::string NormalizeBackendName(const std::string& name) {
  std::string out = name;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return out;
}

std::string ResolveLinearBackend(const std::string& requested) {
  std::string candidate = NormalizeBackendName(requested);
  if (candidate.empty() || candidate == "auto") {
    const char* env_value = std::getenv("MODEL_STACK_LINEAR_BACKEND");
    candidate = env_value != nullptr ? NormalizeBackendName(env_value) : "";
    if (candidate.empty() || candidate == "auto") {
      candidate = HasCublasLtLinearBackend() ? "cublaslt" : "aten";
    }
  }
  TORCH_CHECK(
      candidate == "aten" || (candidate == "cublaslt" && HasCublasLtLinearBackend()),
      "Unsupported linear backend request: ",
      candidate,
      " (supported backends: aten; planned backends: cublaslt)");
  return candidate;
}

bool ForceReferenceGatedMlpFp16() {
  const char* env_value = std::getenv("MODEL_STACK_MLP_FP16_REFERENCE");
  if (env_value == nullptr) {
    return false;
  }
  const auto normalized = NormalizeBackendName(env_value);
  return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

torch::Tensor ApplyActivation(const torch::Tensor& x, const std::string& activation) {
  const auto act = NormalizeBackendName(activation);
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && HasCudaActivationKernel()) {
    return CudaActivationForward(x, act);
  }
#endif
  if (act == "gelu" || act == "geglu") {
    return at::gelu(x, "none");
  }
  if (act == "silu" || act == "swish" || act == "swiglu" || act == "gated-silu") {
    return at::silu(x);
  }
  if (act == "relu" || act == "reglu") {
    return at::relu(x);
  }
  return at::gelu(x, "none");
}

torch::Tensor ApplyGatedActivation(const torch::Tensor& x, const std::string& activation) {
  TORCH_CHECK(x.size(-1) % 2 == 0, "apply_gated_activation: hidden width must be even");
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && HasCudaGatedActivationKernel()) {
    return CudaGatedActivationForward(x, activation);
  }
#endif
  const auto split = x.size(-1) / 2;
  auto a = x.slice(-1, 0, split);
  auto b = x.slice(-1, split, x.size(-1));
  return ApplyActivation(a, activation) * b;
}

py::dict RuntimeInfo() {
  py::dict info;
  info["abi_version"] = kAbiVersion;
#if MODEL_STACK_WITH_CUDA
  info["compiled_with_cuda"] = true;
#else
  info["compiled_with_cuda"] = false;
#endif
  info["native_ops"] = std::vector<std::string>{
      "activation", "gated_activation", "embedding", "linear", "pack_linear_weight", "mlp", "qkv_projection", "pack_qkv_weights", "qkv_packed_heads_projection", "qkv_heads_projection", "split_heads", "merge_heads", "head_output_projection", "prepare_attention_mask", "resolve_position_ids", "create_causal_mask", "resolve_rotary_embedding", "token_counts", "append_tokens", "decode_positions", "rms_norm", "add_rms_norm", "residual_add", "layer_norm", "add_layer_norm", "rope", "kv_cache_append", "kv_cache_write", "kv_cache_gather", "paged_kv_assign_blocks", "paged_kv_reserve_pages", "paged_kv_read_range", "paged_kv_read_last", "paged_kv_append", "paged_kv_compact", "paged_kv_gather", "paged_kv_write", "attention_decode", "attention_prefill", "sampling", "beam_search_step"};
  info["planned_ops"] = std::vector<std::string>{
      "activation", "gated_activation", "embedding", "linear", "pack_linear_weight", "mlp", "qkv_projection", "pack_qkv_weights", "qkv_packed_heads_projection", "qkv_heads_projection", "split_heads", "merge_heads", "head_output_projection", "prepare_attention_mask", "resolve_position_ids", "create_causal_mask", "resolve_rotary_embedding", "token_counts", "append_tokens", "decode_positions", "rms_norm", "add_rms_norm", "residual_add", "layer_norm", "add_layer_norm", "rope", "kv_cache_append", "kv_cache_write", "kv_cache_gather", "paged_kv_assign_blocks", "paged_kv_reserve_pages", "paged_kv_read_range", "paged_kv_read_last", "paged_kv_append", "paged_kv_compact", "paged_kv_gather", "paged_kv_write", "attention_decode",
      "attention_prefill", "sampling", "beam_search_step"};
  info["linear_backend_default"] = HasCublasLtLinearBackend() ? "cublaslt" : "aten";
  info["linear_backends_supported"] = SupportedLinearBackends();
  info["linear_backends_planned"] = PlannedLinearBackends();
  info["cublaslt_linear_dtypes"] = HasCublasLtLinearBackend()
      ? std::vector<std::string>{"float32", "float16", "bfloat16"}
      : std::vector<std::string>{};
  std::vector<std::string> cuda_backend_ops;
  if (HasCudaRmsNormKernel()) {
    cuda_backend_ops.push_back("rms_norm");
  }
  if (HasCudaAddRmsNormKernel()) {
    cuda_backend_ops.push_back("add_rms_norm");
  }
  if (HasCudaResidualAddKernel()) {
    cuda_backend_ops.push_back("residual_add");
  }
  if (HasCudaLayerNormKernel()) {
    cuda_backend_ops.push_back("layer_norm");
  }
  if (HasCudaAddLayerNormKernel()) {
    cuda_backend_ops.push_back("add_layer_norm");
  }
  if (HasCudaEmbeddingKernel()) {
    cuda_backend_ops.push_back("embedding");
  }
  if (HasCudaSamplingKernel()) {
    cuda_backend_ops.push_back("sampling");
  }
  if (HasCudaAppendTokensKernel()) {
    cuda_backend_ops.push_back("append_tokens");
  }
  if (HasCudaDecodePositionsKernel()) {
    cuda_backend_ops.push_back("decode_positions");
  }
  if (HasCudaRopeKernel()) {
    cuda_backend_ops.push_back("rope");
  }
  if (HasCudaKvCacheKernel()) {
    cuda_backend_ops.push_back("kv_cache");
  }
  if (HasCudaAttentionKernel()) {
    cuda_backend_ops.push_back("attention");
  }
  if (HasCublasLtLinearBackend()) {
    cuda_backend_ops.push_back("linear");
    cuda_backend_ops.push_back("qkv_projection");
  }
  if (HasCudaActivationKernel()) {
    cuda_backend_ops.push_back("activation");
  }
  if (HasCudaGatedActivationKernel()) {
    cuda_backend_ops.push_back("gated_activation");
  }
  info["cuda_backend_ops"] = cuda_backend_ops;
  return info;
}

bool HasOp(const std::string& name) {
  const auto ops = NativeOpMap();
  const auto it = ops.find(name);
  return it != ops.end() && it->second;
}

torch::Tensor SplitHeadsForward(
    const torch::Tensor& x,
    int64_t num_heads);

torch::Tensor MergeHeadsForward(const torch::Tensor& x);
torch::Tensor TemperatureForward(const torch::Tensor& logits, double tau);
torch::Tensor TopkMaskForward(const torch::Tensor& logits, int64_t k);
torch::Tensor ToppMaskForward(const torch::Tensor& logits, double p);
torch::Tensor ApplySamplingMaskForward(
    const torch::Tensor& logits,
    const c10::optional<torch::Tensor>& topk_mask,
    const c10::optional<torch::Tensor>& topp_mask,
    const c10::optional<torch::Tensor>& no_repeat_mask);
torch::Tensor PresenceFrequencyPenaltyForward(
    const torch::Tensor& logits,
    const torch::Tensor& counts,
    double alpha_presence,
    double alpha_frequency);
torch::Tensor NoRepeatNgramMaskForward(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    int64_t n);
torch::Tensor SampleNextTokenForward(const torch::Tensor& logits, bool do_sample);
torch::Tensor RepetitionPenaltyForward(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    double penalty);
torch::Tensor TokenCountsForward(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    torch::ScalarType counts_dtype);
py::tuple BeamSearchStepForward(
    const torch::Tensor& beams,
    const torch::Tensor& logits,
    const torch::Tensor& raw_scores,
    const torch::Tensor& finished,
    const torch::Tensor& lengths,
    int64_t beam_size,
    int64_t eos_id,
    int64_t pad_id);

torch::Tensor RmsNormForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    double eps) {
  TORCH_CHECK(x.defined(), "rms_norm_forward: x must be defined");
  TORCH_CHECK(x.dim() >= 1, "rms_norm_forward: x must have at least one dimension");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "rms_norm_forward: eps must be positive and finite");

#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && HasCudaRmsNormKernel()) {
    return CudaRmsNormForward(x, weight, eps);
  }
#endif

  const auto input_dtype = x.scalar_type();
  auto xf = x.to(torch::kFloat32);
  auto variance = xf.pow(2).mean(-1, true);
  auto normalized = xf * torch::rsqrt(variance + eps);
  auto out = normalized.to(input_dtype);

  if (weight.has_value()) {
    auto w = weight.value();
    TORCH_CHECK(
        w.defined(),
        "rms_norm_forward: weight optional was provided but undefined");
    TORCH_CHECK(
        w.dim() == 1,
        "rms_norm_forward: weight must be 1D for the current native path");
    TORCH_CHECK(
        w.size(0) == x.size(-1),
        "rms_norm_forward: weight size must match the last dimension of x");
    out = out * w.to(x.device(), input_dtype);
  }

  return out;
}

torch::Tensor LayerNormReference(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps) {
  TORCH_CHECK(x.defined(), "layer_norm_forward: x must be defined");
  TORCH_CHECK(x.dim() >= 1, "layer_norm_forward: x must have at least one dimension");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "layer_norm_forward: eps must be positive and finite");

  auto xf = x.to(torch::kFloat32);
  auto mean = xf.mean(-1, true);
  auto centered = xf - mean;
  auto var = centered.pow(2).mean(-1, true);
  auto normalized = centered * torch::rsqrt(var + eps);
  auto out = normalized.to(x.scalar_type());

  if (weight.has_value() && weight.value().defined()) {
    auto w = weight.value();
    TORCH_CHECK(w.dim() == 1, "layer_norm_forward: weight must be rank-1");
    TORCH_CHECK(w.size(0) == x.size(-1), "layer_norm_forward: weight size mismatch");
    out = out * w.to(x.device(), x.scalar_type());
  }
  if (bias.has_value() && bias.value().defined()) {
    auto b = bias.value();
    TORCH_CHECK(b.dim() == 1, "layer_norm_forward: bias must be rank-1");
    TORCH_CHECK(b.size(0) == x.size(-1), "layer_norm_forward: bias size mismatch");
    out = out + b.to(x.device(), x.scalar_type());
  }
  return out;
}

torch::Tensor LayerNormForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps) {
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && HasCudaLayerNormKernel()) {
    return CudaLayerNormForward(x, weight, bias, eps);
  }
#endif
  return LayerNormReference(x, weight, bias, eps);
}

std::vector<torch::Tensor> AddRmsNormForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& weight,
    double residual_scale,
    double eps) {
  TORCH_CHECK(x.defined() && update.defined(), "add_rms_norm_forward: x and update must be defined");
  TORCH_CHECK(x.dim() >= 1 && update.dim() >= 1, "add_rms_norm_forward: x and update must have rank >= 1");
  TORCH_CHECK(x.sizes() == update.sizes(), "add_rms_norm_forward: x and update shape mismatch");
  TORCH_CHECK(x.scalar_type() == update.scalar_type(), "add_rms_norm_forward: x and update dtype mismatch");
  TORCH_CHECK(x.device() == update.device(), "add_rms_norm_forward: x and update device mismatch");
  TORCH_CHECK(std::isfinite(residual_scale), "add_rms_norm_forward: residual_scale must be finite");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "add_rms_norm_forward: eps must be positive and finite");

#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && update.is_cuda() && HasCudaAddRmsNormKernel()) {
    return CudaAddRmsNormForward(x, update, weight, residual_scale, eps);
  }
#endif

  auto combined = x + (update * residual_scale);
  auto normalized = RmsNormForward(combined, weight, eps);
  return {combined, normalized};
}

torch::Tensor ResidualAddForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    double residual_scale) {
  TORCH_CHECK(x.defined() && update.defined(), "residual_add_forward: x and update must be defined");
  TORCH_CHECK(x.dim() >= 1 && update.dim() >= 1, "residual_add_forward: x and update must have rank >= 1");
  TORCH_CHECK(x.sizes() == update.sizes(), "residual_add_forward: x and update shape mismatch");
  TORCH_CHECK(x.scalar_type() == update.scalar_type(), "residual_add_forward: x and update dtype mismatch");
  TORCH_CHECK(x.device() == update.device(), "residual_add_forward: x and update device mismatch");
  TORCH_CHECK(std::isfinite(residual_scale), "residual_add_forward: residual_scale must be finite");

#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && update.is_cuda() && HasCudaResidualAddKernel()) {
    return CudaResidualAddForward(x, update, residual_scale);
  }
#endif

  return x + (update * residual_scale);
}

std::vector<torch::Tensor> AddLayerNormForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double residual_scale,
    double eps) {
  TORCH_CHECK(x.defined() && update.defined(), "add_layer_norm_forward: x and update must be defined");
  TORCH_CHECK(x.dim() >= 1 && update.dim() >= 1, "add_layer_norm_forward: x and update must have rank >= 1");
  TORCH_CHECK(x.sizes() == update.sizes(), "add_layer_norm_forward: x and update shape mismatch");
  TORCH_CHECK(x.scalar_type() == update.scalar_type(), "add_layer_norm_forward: x and update dtype mismatch");
  TORCH_CHECK(x.device() == update.device(), "add_layer_norm_forward: x and update device mismatch");
  TORCH_CHECK(std::isfinite(residual_scale), "add_layer_norm_forward: residual_scale must be finite");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "add_layer_norm_forward: eps must be positive and finite");

#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && update.is_cuda() && HasCudaAddLayerNormKernel()) {
    return CudaAddLayerNormForward(x, update, weight, bias, residual_scale, eps);
  }
#endif

  auto combined = x + (update * residual_scale);
  auto normalized = LayerNormReference(combined, weight, bias, eps);
  return {combined, normalized};
}

std::vector<torch::Tensor> ApplyRotaryForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  TORCH_CHECK(q.defined() && k.defined(), "apply_rotary_forward: q and k must be defined");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "apply_rotary_forward: q and k must be rank-4");
  TORCH_CHECK(q.size(0) == k.size(0), "apply_rotary_forward: q and k batch size mismatch");
  TORCH_CHECK(q.size(2) == k.size(2), "apply_rotary_forward: q and k sequence length mismatch");
  TORCH_CHECK(q.size(3) == k.size(3), "apply_rotary_forward: q and k head_dim mismatch");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "apply_rotary_forward: cos and sin must be rank-2");
  TORCH_CHECK(
      cos.sizes() == sin.sizes(),
      "apply_rotary_forward: cos and sin must have the same shape");
  TORCH_CHECK(
      cos.size(0) == q.size(2) && cos.size(1) == q.size(3),
      "apply_rotary_forward: cos/sin shape must match (T, Dh)");
  TORCH_CHECK(
      q.size(3) % 2 == 0,
      "apply_rotary_forward: head_dim must be even");

#if MODEL_STACK_WITH_CUDA
  if (q.is_cuda() && k.is_cuda() && cos.is_cuda() && sin.is_cuda() &&
      HasCudaRopeKernel()) {
    return CudaApplyRotaryForward(q, k, cos, sin);
  }
#endif

  auto cos_b = cos.view({1, 1, cos.size(0), cos.size(1)});
  auto sin_b = sin.view({1, 1, sin.size(0), sin.size(1)});
  const auto half = q.size(3) / 2;

  auto rotate_half = [half](const torch::Tensor& x) {
    auto x1 = x.slice(-1, 0, half);
    auto x2 = x.slice(-1, half, x.size(-1));
    return torch::cat({-x2, x1}, -1);
  };

  auto q_out = (q * cos_b) + (rotate_half(q) * sin_b);
  auto k_out = (k * cos_b) + (rotate_half(k) * sin_b);
  return {q_out, k_out};
}

std::vector<torch::Tensor> KvCacheAppendForward(
    const c10::optional<torch::Tensor>& k_cache,
    const c10::optional<torch::Tensor>& v_cache,
    const torch::Tensor& k_new,
    const torch::Tensor& v_new) {
  TORCH_CHECK(k_new.defined() && v_new.defined(), "kv_cache_append_forward: k_new and v_new must be defined");
  TORCH_CHECK(k_new.dim() == 3 && v_new.dim() == 3, "kv_cache_append_forward: k_new and v_new must be rank-3");
  TORCH_CHECK(k_new.sizes() == v_new.sizes(), "kv_cache_append_forward: k_new and v_new shapes must match");

  auto k_chunk = k_new.contiguous();
  auto v_chunk = v_new.contiguous();

  if (!k_cache.has_value() || !k_cache.value().defined()) {
    TORCH_CHECK(
        !v_cache.has_value() || !v_cache.value().defined(),
        "kv_cache_append_forward: v_cache cannot be defined when k_cache is undefined");
    return {k_chunk, v_chunk};
  }

  TORCH_CHECK(v_cache.has_value() && v_cache.value().defined(), "kv_cache_append_forward: v_cache must be defined");
  const auto& k_prev = k_cache.value();
  const auto& v_prev = v_cache.value();
  TORCH_CHECK(k_prev.dim() == 3 && v_prev.dim() == 3, "kv_cache_append_forward: cached tensors must be rank-3");
  TORCH_CHECK(k_prev.sizes() == v_prev.sizes(), "kv_cache_append_forward: cached K/V shapes must match");
  TORCH_CHECK(k_prev.size(0) == k_chunk.size(0), "kv_cache_append_forward: head count mismatch");
  TORCH_CHECK(k_prev.size(2) == k_chunk.size(2), "kv_cache_append_forward: head_dim mismatch");
  TORCH_CHECK(k_prev.device() == k_chunk.device(), "kv_cache_append_forward: device mismatch");
  TORCH_CHECK(k_prev.scalar_type() == k_chunk.scalar_type(), "kv_cache_append_forward: dtype mismatch");

  return {
      torch::cat({k_prev.contiguous(), k_chunk}, 1),
      torch::cat({v_prev.contiguous(), v_chunk}, 1),
  };
}

torch::Tensor KvCacheWriteForward(
    const torch::Tensor& cache,
    const torch::Tensor& chunk,
    int64_t start) {
  TORCH_CHECK(cache.defined() && chunk.defined(), "kv_cache_write_forward: cache and chunk must be defined");
  TORCH_CHECK(cache.dim() == 3 && chunk.dim() == 3, "kv_cache_write_forward: cache and chunk must be rank-3");
  TORCH_CHECK(cache.size(0) == chunk.size(0), "kv_cache_write_forward: head count mismatch");
  TORCH_CHECK(cache.size(2) == chunk.size(2), "kv_cache_write_forward: head_dim mismatch");
  TORCH_CHECK(cache.device() == chunk.device(), "kv_cache_write_forward: device mismatch");
  TORCH_CHECK(cache.scalar_type() == chunk.scalar_type(), "kv_cache_write_forward: dtype mismatch");
  TORCH_CHECK(start >= 0, "kv_cache_write_forward: start must be non-negative");
  TORCH_CHECK(start + chunk.size(1) <= cache.size(1), "kv_cache_write_forward: chunk exceeds cache capacity");

#if MODEL_STACK_WITH_CUDA
  if (cache.is_cuda() && chunk.is_cuda() && HasCudaKvCacheKernel()) {
    return CudaKvCacheWriteForward(cache, chunk, start);
  }
#endif

  cache.slice(1, start, start + chunk.size(1)).copy_(chunk);
  return cache;
}

torch::Tensor KvCacheGatherForward(
    const torch::Tensor& cache,
    const torch::Tensor& positions) {
  TORCH_CHECK(cache.defined() && positions.defined(), "kv_cache_gather_forward: cache and positions must be defined");
  TORCH_CHECK(cache.dim() == 3 || cache.dim() == 4, "kv_cache_gather_forward: cache must be rank-3 or rank-4");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2,
              "kv_cache_gather_forward: positions must be rank-1 or rank-2");
  TORCH_CHECK(cache.device() == positions.device(), "kv_cache_gather_forward: device mismatch");

  if (cache.dim() == 3) {
    TORCH_CHECK(positions.dim() == 1, "kv_cache_gather_forward: rank-3 cache requires rank-1 positions");
  } else if (positions.dim() == 2) {
    TORCH_CHECK(positions.size(0) == cache.size(0),
                "kv_cache_gather_forward: batch dimension mismatch between cache and positions");
  }

  auto positions_long = positions.to(torch::kLong).contiguous();
  if (positions_long.numel() > 0) {
    const auto min_pos = positions_long.min().item<int64_t>();
    const auto max_pos = positions_long.max().item<int64_t>();
    TORCH_CHECK(min_pos >= 0, "kv_cache_gather_forward: positions must be non-negative");
    const auto cache_seq = cache.dim() == 3 ? cache.size(1) : cache.size(2);
    TORCH_CHECK(max_pos < cache_seq, "kv_cache_gather_forward: positions exceed cache length");
  }

#if MODEL_STACK_WITH_CUDA
  if (cache.is_cuda() && positions_long.is_cuda() && HasCudaKvCacheKernel()) {
    return CudaKvCacheGatherForward(cache, positions_long);
  }
#endif

  if (cache.dim() == 3) {
    return cache.index_select(1, positions_long);
  }
  if (positions_long.dim() == 1) {
    return cache.index_select(2, positions_long);
  }

  std::vector<torch::Tensor> gathered;
  gathered.reserve(static_cast<size_t>(cache.size(0)));
  for (int64_t b = 0; b < cache.size(0); ++b) {
    gathered.push_back(cache.select(0, b).index_select(1, positions_long.select(0, b)));
  }
  return torch::stack(gathered, 0);
}

std::tuple<torch::Tensor, torch::Tensor, int64_t> PagedKvAssignBlocksForward(
    const torch::Tensor& block_table,
    const torch::Tensor& block_ids,
    const torch::Tensor& starts,
    int64_t total,
    int64_t page_size,
    int64_t next_page_id) {
  TORCH_CHECK(block_table.defined() && block_ids.defined() && starts.defined(),
              "paged_kv_assign_blocks_forward: block_table, block_ids, and starts must be defined");
  TORCH_CHECK(block_table.dim() == 2,
              "paged_kv_assign_blocks_forward: block_table must be rank-2 (B, max_blocks)");
  TORCH_CHECK(block_ids.dim() == 1, "paged_kv_assign_blocks_forward: block_ids must be rank-1");
  TORCH_CHECK(starts.dim() == 1, "paged_kv_assign_blocks_forward: starts must be rank-1");
  TORCH_CHECK(block_table.device() == block_ids.device() && block_table.device() == starts.device(),
              "paged_kv_assign_blocks_forward: device mismatch");
  TORCH_CHECK(block_ids.numel() == starts.numel(),
              "paged_kv_assign_blocks_forward: block_ids and starts must have the same length");
  TORCH_CHECK(total >= 0, "paged_kv_assign_blocks_forward: total must be non-negative");
  TORCH_CHECK(page_size > 0, "paged_kv_assign_blocks_forward: page_size must be positive");
  TORCH_CHECK(next_page_id >= 0, "paged_kv_assign_blocks_forward: next_page_id must be non-negative");

  auto table_long = block_table.to(torch::kLong).contiguous();
  auto block_ids_long = block_ids.to(torch::kLong).contiguous();
  auto starts_long = starts.to(torch::kLong).contiguous();

  if (block_ids_long.numel() > 0) {
    const auto min_block = block_ids_long.min().item<int64_t>();
    const auto max_block = block_ids_long.max().item<int64_t>();
    TORCH_CHECK(min_block >= 0, "paged_kv_assign_blocks_forward: block_ids must be non-negative");
    TORCH_CHECK(max_block < table_long.size(0),
                "paged_kv_assign_blocks_forward: block_ids exceed block_table batch size");
    const auto unique_result = at::_unique2(block_ids_long, true, false, false);
    TORCH_CHECK(std::get<0>(unique_result).numel() == block_ids_long.numel(),
                "paged_kv_assign_blocks_forward: block_ids must be unique");
  }
  if (starts_long.numel() > 0) {
    const auto min_start = starts_long.min().item<int64_t>();
    TORCH_CHECK(min_start >= 0, "paged_kv_assign_blocks_forward: starts must be non-negative");
  }

  if (block_ids_long.numel() == 0 || total == 0) {
    auto empty = torch::empty({block_ids_long.size(0), 0}, table_long.options());
    return std::make_tuple(table_long, empty, next_page_id);
  }

  const auto rows = block_ids_long.size(0);
  auto end_positions = starts_long + (total - 1);
  auto start_blocks = at::floor_divide(starts_long, page_size);
  auto end_blocks = at::floor_divide(end_positions, page_size);
  const auto needed_blocks = end_blocks.max().item<int64_t>() + 1;

  int64_t next_blocks = table_long.size(1);
  if (next_blocks < needed_blocks) {
    next_blocks = std::max<int64_t>(1, next_blocks > 0 ? next_blocks : 1);
    while (next_blocks < needed_blocks) {
      next_blocks *= 2;
    }
  }

  auto next_table = table_long;
  if (next_blocks != table_long.size(1)) {
    next_table = torch::full({table_long.size(0), next_blocks}, -1, table_long.options());
    if (table_long.numel() > 0) {
      next_table.slice(1, 0, table_long.size(1)).copy_(table_long);
    }
  } else {
    next_table = table_long.clone();
  }

  auto selected = next_table.index_select(0, block_ids_long).clone();
  auto selected_prefix = selected.slice(1, 0, needed_blocks).clone();
  auto active_slots = torch::arange(needed_blocks, table_long.options()).view({1, needed_blocks});
  auto active_mask =
      active_slots.ge(start_blocks.view({rows, 1})) & active_slots.le(end_blocks.view({rows, 1}));
  auto missing_mask = active_mask & selected_prefix.lt(0);
  const auto missing_count = missing_mask.sum().item<int64_t>();
  if (missing_count > 0) {
    auto new_ids = torch::arange(next_page_id, next_page_id + missing_count, table_long.options());
    selected_prefix.masked_scatter_(missing_mask, new_ids);
  }

  selected.slice(1, 0, needed_blocks).copy_(selected_prefix);
  next_table.index_copy_(0, block_ids_long, selected);
  auto selected_active = selected_prefix.clamp_min(0).contiguous();
  return std::make_tuple(next_table, selected_active, next_page_id + missing_count);
}

torch::Tensor PagedKvReservePagesForward(
    const torch::Tensor& pages,
    int64_t used_pages,
    int64_t needed_pages) {
  TORCH_CHECK(pages.defined(), "paged_kv_reserve_pages_forward: pages must be defined");
  TORCH_CHECK(pages.dim() == 4,
              "paged_kv_reserve_pages_forward: pages must be rank-4 (P, H, page_size, D)");
  TORCH_CHECK(used_pages >= 0, "paged_kv_reserve_pages_forward: used_pages must be non-negative");
  TORCH_CHECK(needed_pages >= 0, "paged_kv_reserve_pages_forward: needed_pages must be non-negative");
  TORCH_CHECK(used_pages <= pages.size(0),
              "paged_kv_reserve_pages_forward: used_pages exceed current capacity");
  if (needed_pages <= pages.size(0)) {
    return pages;
  }

  int64_t new_cap = std::max<int64_t>(1, pages.size(0) > 0 ? pages.size(0) : 1);
  while (new_cap < needed_pages) {
    new_cap *= 2;
  }

  auto next_pages = torch::empty({new_cap, pages.size(1), pages.size(2), pages.size(3)}, pages.options());
  if (used_pages > 0) {
    next_pages.slice(0, 0, used_pages).copy_(pages.slice(0, 0, used_pages));
  }
  return next_pages;
}

std::vector<torch::Tensor> PagedKvReadLastForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t keep) {
  TORCH_CHECK(pages.defined() && block_table.defined() && lengths.defined(),
              "paged_kv_read_last_forward: pages, block_table, and lengths must be defined");
  TORCH_CHECK(pages.dim() == 4, "paged_kv_read_last_forward: pages must be rank-4 (P, H, page_size, D)");
  TORCH_CHECK(block_table.dim() == 2,
              "paged_kv_read_last_forward: block_table must be rank-2 (B, max_blocks)");
  TORCH_CHECK(lengths.dim() == 1, "paged_kv_read_last_forward: lengths must be rank-1");
  TORCH_CHECK(pages.device() == block_table.device() && pages.device() == lengths.device(),
              "paged_kv_read_last_forward: device mismatch");
  TORCH_CHECK(block_table.size(0) == lengths.size(0),
              "paged_kv_read_last_forward: batch dimension mismatch between block_table and lengths");
  TORCH_CHECK(keep >= 0, "paged_kv_read_last_forward: keep must be non-negative");

  auto block_table_long = block_table.to(torch::kLong).contiguous();
  auto lengths_long = lengths.to(torch::kLong).contiguous();
  if (block_table_long.numel() > 0) {
    const auto min_page = block_table_long.min().item<int64_t>();
    const auto max_page = block_table_long.max().item<int64_t>();
    TORCH_CHECK(min_page >= 0, "paged_kv_read_last_forward: block_table entries must be non-negative");
    TORCH_CHECK(max_page < pages.size(0), "paged_kv_read_last_forward: block_table entries exceed page pool size");
  }
  if (lengths_long.numel() > 0) {
    const auto min_len = lengths_long.min().item<int64_t>();
    const auto max_len = lengths_long.max().item<int64_t>();
    TORCH_CHECK(min_len >= 0, "paged_kv_read_last_forward: lengths must be non-negative");
    TORCH_CHECK(max_len <= block_table_long.size(1) * pages.size(2),
                "paged_kv_read_last_forward: lengths exceed logical block-table capacity");
  }

  auto kept_lengths = torch::clamp(lengths_long, 0, keep);
  const auto max_keep = kept_lengths.numel() > 0 ? kept_lengths.max().item<int64_t>() : 0;
  auto out = torch::zeros({block_table_long.size(0), pages.size(1), max_keep, pages.size(3)}, pages.options());
  if (block_table_long.size(0) == 0 || max_keep == 0) {
    return {out, kept_lengths};
  }

#if MODEL_STACK_WITH_CUDA
  if (pages.is_cuda() && block_table_long.is_cuda() && lengths_long.is_cuda() && HasCudaKvCacheKernel()) {
    return CudaPagedKvReadLastForward(pages, block_table_long, lengths_long, keep);
  }
#endif

  for (int64_t b = 0; b < block_table_long.size(0); ++b) {
    const auto live_len = lengths_long[b].item<int64_t>();
    const auto row_keep = std::min<int64_t>(live_len, keep);
    const auto start = std::max<int64_t>(live_len - row_keep, 0);
    for (int64_t t = 0; t < row_keep; ++t) {
      const auto pos = start + t;
      const auto block_idx = pos / pages.size(2);
      const auto page_offset = pos % pages.size(2);
      const auto page_id = block_table_long.index({b, block_idx}).item<int64_t>();
      out.index_put_({b, at::indexing::Slice(), t, at::indexing::Slice()},
                     pages.index({page_id, at::indexing::Slice(), page_offset, at::indexing::Slice()}));
    }
  }
  return {out, kept_lengths};
}

torch::Tensor PagedKvReadRangeForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t start,
    int64_t end) {
  TORCH_CHECK(pages.defined() && block_table.defined() && lengths.defined(),
              "paged_kv_read_range_forward: pages, block_table, and lengths must be defined");
  TORCH_CHECK(pages.dim() == 4, "paged_kv_read_range_forward: pages must be rank-4 (P, H, page_size, D)");
  TORCH_CHECK(block_table.dim() == 2,
              "paged_kv_read_range_forward: block_table must be rank-2 (B, max_blocks)");
  TORCH_CHECK(lengths.dim() == 1, "paged_kv_read_range_forward: lengths must be rank-1");
  TORCH_CHECK(pages.device() == block_table.device() && pages.device() == lengths.device(),
              "paged_kv_read_range_forward: device mismatch");
  TORCH_CHECK(block_table.size(0) == lengths.size(0),
              "paged_kv_read_range_forward: batch dimension mismatch between block_table and lengths");
  TORCH_CHECK(start >= 0, "paged_kv_read_range_forward: start must be non-negative");
  TORCH_CHECK(end >= start, "paged_kv_read_range_forward: end must be >= start");

  auto block_table_long = block_table.to(torch::kLong).contiguous();
  auto block_table_safe = block_table_long.clone();
  block_table_safe.clamp_min_(0);
  auto lengths_long = lengths.to(torch::kLong).contiguous();
  if (block_table_long.numel() > 0) {
    const auto max_page = block_table_safe.max().item<int64_t>();
    TORCH_CHECK(max_page < pages.size(0), "paged_kv_read_range_forward: block_table entries exceed page pool size");
  }
  if (lengths_long.numel() > 0) {
    const auto min_len = lengths_long.min().item<int64_t>();
    const auto max_len = lengths_long.max().item<int64_t>();
    TORCH_CHECK(min_len >= 0, "paged_kv_read_range_forward: lengths must be non-negative");
    TORCH_CHECK(max_len <= block_table_long.size(1) * pages.size(2),
                "paged_kv_read_range_forward: lengths exceed logical block-table capacity");
  }

  const auto gather_seq = end - start;
  auto out = torch::zeros({block_table_long.size(0), pages.size(1), gather_seq, pages.size(3)}, pages.options());
  if (block_table_long.size(0) == 0 || gather_seq == 0) {
    return out;
  }

#if MODEL_STACK_WITH_CUDA
  if (pages.is_cuda() && block_table_safe.is_cuda() && lengths_long.is_cuda() && HasCudaKvCacheKernel()) {
    return CudaPagedKvReadRangeForward(pages, block_table_safe, lengths_long, start, end);
  }
#endif

  for (int64_t b = 0; b < block_table_long.size(0); ++b) {
    const auto live_len = lengths_long[b].item<int64_t>();
    for (int64_t t = 0; t < gather_seq; ++t) {
      const auto pos = start + t;
      if (pos >= live_len) {
        continue;
      }
      const auto block_idx = pos / pages.size(2);
      const auto page_offset = pos % pages.size(2);
      const auto page_id = block_table_safe.index({b, block_idx}).item<int64_t>();
      out.index_put_({b, at::indexing::Slice(), t, at::indexing::Slice()},
                     pages.index({page_id, at::indexing::Slice(), page_offset, at::indexing::Slice()}));
    }
  }
  return out;
}

torch::Tensor PagedKvWriteForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions,
    const torch::Tensor& values);
std::vector<torch::Tensor> PagedKvAppendForward(
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t page_count,
    const torch::Tensor& k_chunk,
    const torch::Tensor& v_chunk,
    const torch::Tensor& block_ids);
std::vector<torch::Tensor> PagedKvCompactForward(
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t keep);

class PagedKvLayerState {
 public:
  PagedKvLayerState(
      int64_t batch,
      int64_t heads,
      int64_t head_dim,
      int64_t page_size,
      const torch::Tensor& example)
      : batch_(batch),
        heads_(heads),
        head_dim_(head_dim),
        page_size_(std::max<int64_t>(page_size, 1)) {
    TORCH_CHECK(example.defined(), "PagedKvLayerState: example tensor must be defined");
    auto page_options = example.options();
    auto long_options = page_options.dtype(torch::kLong);
    k_pages_ = torch::empty({0, heads_, page_size_, head_dim_}, page_options);
    v_pages_ = torch::empty({0, heads_, page_size_, head_dim_}, page_options);
    block_table_ = torch::empty({batch_, 0}, long_options);
    lengths_ = torch::zeros({batch_}, long_options);
    page_count_ = 0;
  }

  void Reset() {
    auto page_options = k_pages_.options();
    auto long_options = block_table_.options();
    k_pages_ = torch::empty({0, heads_, page_size_, head_dim_}, page_options);
    v_pages_ = torch::empty({0, heads_, page_size_, head_dim_}, page_options);
    block_table_ = torch::empty({batch_, 0}, long_options);
    lengths_ = torch::zeros({batch_}, long_options);
    page_count_ = 0;
  }

  void Append(
      const torch::Tensor& k_chunk,
      const torch::Tensor& v_chunk,
      const c10::optional<torch::Tensor>& block_ids) {
    auto ids = NormalizeBlockIds(block_ids, k_chunk.size(0));
    auto result = PagedKvAppendForward(
        k_pages_,
        v_pages_,
        block_table_,
        lengths_,
        page_count_,
        k_chunk.to(k_pages_.options()).contiguous(),
        v_chunk.to(v_pages_.options()).contiguous(),
        ids);
    k_pages_ = result[0];
    v_pages_ = result[1];
    block_table_ = result[2];
    lengths_ = result[3];
    page_count_ = result[4].item<int64_t>();
  }

  std::vector<torch::Tensor> ReadRange(
      int64_t start,
      int64_t end,
      const c10::optional<torch::Tensor>& block_ids) const {
    auto ids = NormalizeBlockIds(block_ids);
    const auto gather_seq = std::max<int64_t>(end - start, 0);
    if (ids.numel() == 0) {
      auto empty = torch::empty({0, heads_, gather_seq, head_dim_}, k_pages_.options());
      return {empty, empty.clone()};
    }
    auto live_lengths = lengths_.index_select(0, ids);
    auto block_table = block_table_.index_select(0, ids).contiguous();
    auto k = PagedKvReadRangeForward(k_pages_.slice(0, 0, page_count_), block_table, live_lengths, start, end);
    auto v = PagedKvReadRangeForward(v_pages_.slice(0, 0, page_count_), block_table, live_lengths, start, end);
    return {k, v};
  }

  std::vector<torch::Tensor> ReadLast(
      int64_t keep,
      const c10::optional<torch::Tensor>& block_ids) const {
    auto ids = NormalizeBlockIds(block_ids);
    if (ids.numel() == 0) {
      auto empty = torch::empty({0, heads_, 0, head_dim_}, k_pages_.options());
      auto empty_lengths = torch::empty({0}, lengths_.options());
      return {empty, empty.clone(), ids, empty_lengths};
    }
    auto live_lengths = lengths_.index_select(0, ids);
    auto block_table = block_table_.index_select(0, ids).contiguous().clone();
    block_table.clamp_min_(0);
    auto keep_k = PagedKvReadLastForward(k_pages_.slice(0, 0, page_count_), block_table, live_lengths, keep);
    auto keep_v = PagedKvReadLastForward(v_pages_.slice(0, 0, page_count_), block_table, live_lengths, keep);
    TORCH_CHECK(torch::equal(keep_k[1], keep_v[1]), "PagedKvLayerState: kept lengths mismatch");
    return {keep_k[0], keep_v[0], ids, keep_k[1]};
  }

  void Compact(int64_t keep) {
    auto result = PagedKvCompactForward(
        k_pages_.slice(0, 0, page_count_),
        v_pages_.slice(0, 0, page_count_),
        block_table_,
        lengths_,
        keep);
    k_pages_ = result[0];
    v_pages_ = result[1];
    block_table_ = result[2];
    lengths_ = result[3];
    page_count_ = k_pages_.size(0);
  }

  torch::Tensor KPages() const { return k_pages_; }
  torch::Tensor VPages() const { return v_pages_; }
  torch::Tensor BlockTable() const { return block_table_; }
  torch::Tensor Lengths() const { return lengths_; }
  int64_t PageCount() const { return page_count_; }

 private:
  torch::Tensor NormalizeBlockIds(
      const c10::optional<torch::Tensor>& block_ids,
      c10::optional<int64_t> expected_rows = c10::nullopt) const {
    if (!block_ids.has_value() || !block_ids.value().defined()) {
      if (expected_rows.has_value()) {
        TORCH_CHECK(
            expected_rows.value() == batch_,
            "PagedKvLayerState: chunk batch size must match cache batch size when block_ids are omitted");
      }
      return torch::arange(batch_, block_table_.options());
    }
    auto ids = block_ids.value().to(torch::kLong).contiguous().view(-1);
    if (expected_rows.has_value()) {
      TORCH_CHECK(
          ids.numel() == expected_rows.value(),
          "PagedKvLayerState: block_ids must match chunk batch size");
    }
    if (ids.numel() == 0) {
      return ids;
    }
    const auto min_id = ids.min().item<int64_t>();
    const auto max_id = ids.max().item<int64_t>();
    TORCH_CHECK(min_id >= 0, "PagedKvLayerState: block_ids must be non-negative");
    TORCH_CHECK(max_id < batch_, "PagedKvLayerState: block_ids exceed cache batch size");
    const auto unique_result = at::_unique2(ids, true, false, false);
    TORCH_CHECK(std::get<0>(unique_result).numel() == ids.numel(), "PagedKvLayerState: block_ids must be unique");
    return ids;
  }

  int64_t batch_;
  int64_t heads_;
  int64_t head_dim_;
  int64_t page_size_;
  torch::Tensor k_pages_;
  torch::Tensor v_pages_;
  torch::Tensor block_table_;
  torch::Tensor lengths_;
  int64_t page_count_;
};

class PagedKvCacheState {
 public:
  PagedKvCacheState(
      int64_t batch,
      int64_t layers,
      int64_t heads,
      int64_t head_dim,
      int64_t page_size,
      const torch::Tensor& example)
      : batch_(batch),
        layers_count_(layers),
        heads_(heads),
        head_dim_(head_dim),
        page_size_(std::max<int64_t>(page_size, 1)) {
    TORCH_CHECK(layers_count_ >= 0, "PagedKvCacheState: layers must be non-negative");
    layers_.reserve(static_cast<size_t>(layers_count_));
    for (int64_t layer_idx = 0; layer_idx < layers_count_; ++layer_idx) {
      layers_.push_back(std::make_shared<PagedKvLayerState>(batch_, heads_, head_dim_, page_size_, example));
    }
  }

  void Reset() {
    for (const auto& layer : layers_) {
      layer->Reset();
    }
  }

  void ResetLayer(int64_t layer_idx) { Layer(layer_idx)->Reset(); }

  void Append(
      int64_t layer_idx,
      const torch::Tensor& k_chunk,
      const torch::Tensor& v_chunk,
      const c10::optional<torch::Tensor>& block_ids) {
    Layer(layer_idx)->Append(k_chunk, v_chunk, block_ids);
  }

  std::vector<torch::Tensor> ReadRange(
      int64_t layer_idx,
      int64_t start,
      int64_t end,
      const c10::optional<torch::Tensor>& block_ids) const {
    return Layer(layer_idx)->ReadRange(start, end, block_ids);
  }

  std::vector<torch::Tensor> ReadLast(
      int64_t layer_idx,
      int64_t keep,
      const c10::optional<torch::Tensor>& block_ids) const {
    return Layer(layer_idx)->ReadLast(keep, block_ids);
  }

  void Compact(int64_t keep) {
    for (const auto& layer : layers_) {
      layer->Compact(keep);
    }
  }

  void CompactLayer(int64_t layer_idx, int64_t keep) { Layer(layer_idx)->Compact(keep); }

  torch::Tensor KPages(int64_t layer_idx) const { return Layer(layer_idx)->KPages(); }
  torch::Tensor VPages(int64_t layer_idx) const { return Layer(layer_idx)->VPages(); }
  torch::Tensor BlockTable(int64_t layer_idx) const { return Layer(layer_idx)->BlockTable(); }
  torch::Tensor Lengths(int64_t layer_idx) const { return Layer(layer_idx)->Lengths(); }
  int64_t PageCount(int64_t layer_idx) const { return Layer(layer_idx)->PageCount(); }

  int64_t MaxLength(int64_t layer_idx) const {
    auto lengths = Layer(layer_idx)->Lengths();
    return lengths.numel() > 0 ? lengths.max().item<int64_t>() : 0;
  }

  int64_t NumLayers() const { return layers_count_; }

 private:
  std::shared_ptr<PagedKvLayerState> Layer(int64_t layer_idx) const {
    TORCH_CHECK(layer_idx >= 0 && layer_idx < layers_count_, "PagedKvCacheState: layer_idx out of range");
    return layers_[static_cast<size_t>(layer_idx)];
  }

  int64_t batch_;
  int64_t layers_count_;
  int64_t heads_;
  int64_t head_dim_;
  int64_t page_size_;
  std::vector<std::shared_ptr<PagedKvLayerState>> layers_;
};

std::vector<torch::Tensor> PagedKvAppendForward(
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t page_count,
    const torch::Tensor& k_chunk,
    const torch::Tensor& v_chunk,
    const torch::Tensor& block_ids) {
  TORCH_CHECK(k_pages.defined() && v_pages.defined() && block_table.defined() && lengths.defined(),
              "paged_kv_append_forward: k_pages, v_pages, block_table, and lengths must be defined");
  TORCH_CHECK(k_chunk.defined() && v_chunk.defined() && block_ids.defined(),
              "paged_kv_append_forward: k_chunk, v_chunk, and block_ids must be defined");
  TORCH_CHECK(k_pages.dim() == 4 && v_pages.dim() == 4,
              "paged_kv_append_forward: k_pages and v_pages must be rank-4");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(),
              "paged_kv_append_forward: k_pages and v_pages shapes must match");
  TORCH_CHECK(block_table.dim() == 2, "paged_kv_append_forward: block_table must be rank-2");
  TORCH_CHECK(lengths.dim() == 1, "paged_kv_append_forward: lengths must be rank-1");
  TORCH_CHECK(k_chunk.dim() == 4 && v_chunk.dim() == 4,
              "paged_kv_append_forward: k_chunk and v_chunk must be rank-4 (B,H,T,D)");
  TORCH_CHECK(k_chunk.sizes() == v_chunk.sizes(),
              "paged_kv_append_forward: k_chunk and v_chunk shapes must match");
  TORCH_CHECK(block_ids.dim() == 1, "paged_kv_append_forward: block_ids must be rank-1");
  TORCH_CHECK(k_pages.device() == v_pages.device() && k_pages.device() == block_table.device() &&
                  k_pages.device() == lengths.device() && k_pages.device() == k_chunk.device() &&
                  k_pages.device() == v_chunk.device() && k_pages.device() == block_ids.device(),
              "paged_kv_append_forward: device mismatch");
  TORCH_CHECK(k_pages.scalar_type() == v_pages.scalar_type() &&
                  k_pages.scalar_type() == k_chunk.scalar_type() &&
                  k_pages.scalar_type() == v_chunk.scalar_type(),
              "paged_kv_append_forward: dtype mismatch");
  TORCH_CHECK(block_table.size(0) == lengths.size(0),
              "paged_kv_append_forward: block_table and lengths batch sizes must match");
  TORCH_CHECK(k_pages.size(1) == k_chunk.size(1), "paged_kv_append_forward: head count mismatch");
  TORCH_CHECK(k_pages.size(3) == k_chunk.size(3), "paged_kv_append_forward: head_dim mismatch");
  TORCH_CHECK(block_ids.size(0) == k_chunk.size(0),
              "paged_kv_append_forward: block_ids must match chunk batch size");
  TORCH_CHECK(page_count >= 0, "paged_kv_append_forward: page_count must be non-negative");
  TORCH_CHECK(page_count <= k_pages.size(0),
              "paged_kv_append_forward: page_count exceeds current page capacity");

  auto block_table_long = block_table.to(torch::kLong).contiguous();
  auto lengths_long = lengths.to(torch::kLong).contiguous();
  auto block_ids_long = block_ids.to(torch::kLong).contiguous();
  auto k_chunk_contig = k_chunk.contiguous();
  auto v_chunk_contig = v_chunk.contiguous();
  if (block_ids_long.numel() > 0) {
    const auto min_block = block_ids_long.min().item<int64_t>();
    const auto max_block = block_ids_long.max().item<int64_t>();
    TORCH_CHECK(min_block >= 0, "paged_kv_append_forward: block_ids must be non-negative");
    TORCH_CHECK(max_block < block_table_long.size(0),
                "paged_kv_append_forward: block_ids exceed block_table batch size");
  }

  const auto total = k_chunk_contig.size(2);
  if (total == 0 || block_ids_long.numel() == 0) {
    return {k_pages, v_pages, block_table_long, lengths_long, torch::tensor(page_count, block_table_long.options())};
  }

  auto starts_long = lengths_long.index_select(0, block_ids_long);
  auto base = torch::arange(total, block_table_long.options()).view({1, total});
  auto positions = starts_long.view({block_ids_long.size(0), 1}) + base;
  auto assign_result = PagedKvAssignBlocksForward(
      block_table_long,
      block_ids_long,
      starts_long,
      total,
      k_pages.size(2),
      page_count);
  auto next_block_table = std::get<0>(assign_result);
  auto selected_block_table = std::get<1>(assign_result);
  auto next_page_count = std::get<2>(assign_result);
  auto next_k_pages = PagedKvReservePagesForward(k_pages, page_count, next_page_count);
  auto next_v_pages = PagedKvReservePagesForward(v_pages, page_count, next_page_count);
  next_k_pages = PagedKvWriteForward(next_k_pages, selected_block_table, positions, k_chunk_contig);
  next_v_pages = PagedKvWriteForward(next_v_pages, selected_block_table, positions, v_chunk_contig);
  auto next_lengths = lengths_long.clone();
  next_lengths.index_copy_(0, block_ids_long, starts_long + total);
  return {
      next_k_pages,
      next_v_pages,
      next_block_table,
      next_lengths,
      torch::tensor(next_page_count, block_table_long.options()),
  };
}

std::vector<torch::Tensor> PagedKvCompactForward(
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t keep) {
  TORCH_CHECK(k_pages.defined() && v_pages.defined() && block_table.defined() && lengths.defined(),
              "paged_kv_compact_forward: k_pages, v_pages, block_table, and lengths must be defined");
  TORCH_CHECK(k_pages.dim() == 4 && v_pages.dim() == 4,
              "paged_kv_compact_forward: k_pages and v_pages must be rank-4");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(),
              "paged_kv_compact_forward: k_pages and v_pages shapes must match");
  TORCH_CHECK(block_table.dim() == 2, "paged_kv_compact_forward: block_table must be rank-2");
  TORCH_CHECK(lengths.dim() == 1, "paged_kv_compact_forward: lengths must be rank-1");
  TORCH_CHECK(k_pages.device() == v_pages.device() && k_pages.device() == block_table.device() &&
                  k_pages.device() == lengths.device(),
              "paged_kv_compact_forward: device mismatch");
  TORCH_CHECK(k_pages.scalar_type() == v_pages.scalar_type(),
              "paged_kv_compact_forward: dtype mismatch between k_pages and v_pages");
  TORCH_CHECK(block_table.size(0) == lengths.size(0),
              "paged_kv_compact_forward: batch dimension mismatch between block_table and lengths");
  TORCH_CHECK(keep >= 0, "paged_kv_compact_forward: keep must be non-negative");

  auto block_table_long = block_table.to(torch::kLong).contiguous();
  auto block_table_safe = block_table_long.clone();
  block_table_safe.clamp_min_(0);
  auto lengths_long = lengths.to(torch::kLong).contiguous();
  auto keep_kv = PagedKvReadLastForward(k_pages, block_table_safe, lengths_long, keep);
  auto keep_vv = PagedKvReadLastForward(v_pages, block_table_safe, lengths_long, keep);
  auto kept_k = keep_kv[0];
  auto kept_v = keep_vv[0];
  auto kept_lengths = keep_kv[1];
  TORCH_CHECK(torch::equal(kept_lengths, keep_vv[1]),
              "paged_kv_compact_forward: kept lengths mismatch between K and V");

  auto next_block_table = torch::empty({block_table_long.size(0), 0}, block_table_long.options());
  auto next_k_pages = torch::empty({0, k_pages.size(1), k_pages.size(2), k_pages.size(3)}, k_pages.options());
  auto next_v_pages = torch::empty({0, v_pages.size(1), v_pages.size(2), v_pages.size(3)}, v_pages.options());
  int64_t next_page_id = 0;

  if (kept_lengths.numel() == 0) {
    return {next_k_pages, next_v_pages, next_block_table, kept_lengths};
  }

  auto kept_cpu = kept_lengths.to(torch::kCPU);
  std::vector<int64_t> unique_lengths;
  unique_lengths.reserve(static_cast<size_t>(kept_cpu.numel()));
  for (int64_t i = 0; i < kept_cpu.numel(); ++i) {
    const auto length = kept_cpu[i].item<int64_t>();
    if (length > 0) {
      unique_lengths.push_back(length);
    }
  }
  std::sort(unique_lengths.begin(), unique_lengths.end());
  unique_lengths.erase(std::unique(unique_lengths.begin(), unique_lengths.end()), unique_lengths.end());

  for (const auto length : unique_lengths) {
    auto group_ids = torch::nonzero(kept_lengths == length).view(-1);
    if (group_ids.numel() == 0) {
      continue;
    }
    auto starts = torch::zeros({group_ids.size(0)}, block_table_long.options());
    auto assign_result = PagedKvAssignBlocksForward(
        next_block_table,
        group_ids,
        starts,
        length,
        k_pages.size(2),
        next_page_id);
    auto previous_page_id = next_page_id;
    next_block_table = std::get<0>(assign_result);
    auto selected_block_table = std::get<1>(assign_result);
    next_page_id = std::get<2>(assign_result);

    next_k_pages = PagedKvReservePagesForward(next_k_pages, previous_page_id, next_page_id);
    next_v_pages = PagedKvReservePagesForward(next_v_pages, previous_page_id, next_page_id);
    auto positions = torch::arange(length, block_table_long.options());
    auto group_k = kept_k.index_select(0, group_ids).slice(2, 0, length).contiguous();
    auto group_v = kept_v.index_select(0, group_ids).slice(2, 0, length).contiguous();
    next_k_pages = PagedKvWriteForward(next_k_pages, selected_block_table, positions, group_k);
    next_v_pages = PagedKvWriteForward(next_v_pages, selected_block_table, positions, group_v);
  }

  return {
      next_k_pages.slice(0, 0, next_page_id).contiguous(),
      next_v_pages.slice(0, 0, next_page_id).contiguous(),
      next_block_table,
      kept_lengths,
  };
}

torch::Tensor PagedKvGatherForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions) {
  TORCH_CHECK(pages.defined() && block_table.defined() && positions.defined(),
              "paged_kv_gather_forward: pages, block_table, and positions must be defined");
  TORCH_CHECK(pages.dim() == 4, "paged_kv_gather_forward: pages must be rank-4 (P, H, page_size, D)");
  TORCH_CHECK(block_table.dim() == 2, "paged_kv_gather_forward: block_table must be rank-2 (B, max_blocks)");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2,
              "paged_kv_gather_forward: positions must be rank-1 or rank-2");
  TORCH_CHECK(pages.device() == block_table.device() && pages.device() == positions.device(),
              "paged_kv_gather_forward: device mismatch");
  if (positions.dim() == 2) {
    TORCH_CHECK(positions.size(0) == block_table.size(0),
                "paged_kv_gather_forward: batch dimension mismatch between block_table and positions");
  }

  auto block_table_long = block_table.to(torch::kLong).contiguous();
  auto positions_long = positions.to(torch::kLong).contiguous();
  if (block_table_long.numel() > 0) {
    const auto min_page = block_table_long.min().item<int64_t>();
    const auto max_page = block_table_long.max().item<int64_t>();
    TORCH_CHECK(min_page >= 0, "paged_kv_gather_forward: block_table entries must be non-negative");
    TORCH_CHECK(max_page < pages.size(0), "paged_kv_gather_forward: block_table entries exceed page pool size");
  }
  if (positions_long.numel() > 0) {
    const auto min_pos = positions_long.min().item<int64_t>();
    const auto max_pos = positions_long.max().item<int64_t>();
    TORCH_CHECK(min_pos >= 0, "paged_kv_gather_forward: positions must be non-negative");
    TORCH_CHECK(max_pos < block_table_long.size(1) * pages.size(2),
                "paged_kv_gather_forward: positions exceed logical block-table capacity");
  }

#if MODEL_STACK_WITH_CUDA
  if (pages.is_cuda() && block_table_long.is_cuda() && positions_long.is_cuda() && HasCudaKvCacheKernel()) {
    return CudaPagedKvGatherForward(pages, block_table_long, positions_long);
  }
#endif

  const auto gather_seq = positions_long.dim() == 1 ? positions_long.size(0) : positions_long.size(1);
  auto out = torch::empty(
      {block_table_long.size(0), pages.size(1), gather_seq, pages.size(3)},
      pages.options());
  for (int64_t b = 0; b < block_table_long.size(0); ++b) {
    for (int64_t t = 0; t < gather_seq; ++t) {
      const auto pos = positions_long.dim() == 1 ? positions_long[t].item<int64_t>()
                                                 : positions_long.index({b, t}).item<int64_t>();
      const auto block_idx = pos / pages.size(2);
      const auto page_offset = pos % pages.size(2);
      const auto page_id = block_table_long.index({b, block_idx}).item<int64_t>();
      out.index_put_({b, at::indexing::Slice(), t, at::indexing::Slice()},
                     pages.index({page_id, at::indexing::Slice(), page_offset, at::indexing::Slice()}));
    }
  }
  return out;
}

torch::Tensor PagedKvWriteForward(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions,
    const torch::Tensor& values) {
  TORCH_CHECK(pages.defined() && block_table.defined() && positions.defined() && values.defined(),
              "paged_kv_write_forward: pages, block_table, positions, and values must be defined");
  TORCH_CHECK(pages.dim() == 4, "paged_kv_write_forward: pages must be rank-4 (P, H, page_size, D)");
  TORCH_CHECK(block_table.dim() == 2, "paged_kv_write_forward: block_table must be rank-2 (B, max_blocks)");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2,
              "paged_kv_write_forward: positions must be rank-1 or rank-2");
  TORCH_CHECK(values.dim() == 4, "paged_kv_write_forward: values must be rank-4 (B, H, T, D)");
  TORCH_CHECK(pages.device() == block_table.device() && pages.device() == positions.device() &&
                  pages.device() == values.device(),
              "paged_kv_write_forward: device mismatch");
  TORCH_CHECK(pages.scalar_type() == values.scalar_type(), "paged_kv_write_forward: dtype mismatch");
  TORCH_CHECK(block_table.size(0) == values.size(0), "paged_kv_write_forward: batch dimension mismatch");
  TORCH_CHECK(pages.size(1) == values.size(1), "paged_kv_write_forward: head count mismatch");
  TORCH_CHECK(pages.size(3) == values.size(3), "paged_kv_write_forward: head_dim mismatch");
  if (positions.dim() == 2) {
    TORCH_CHECK(positions.size(0) == values.size(0),
                "paged_kv_write_forward: batch dimension mismatch between positions and values");
    TORCH_CHECK(positions.size(1) == values.size(2),
                "paged_kv_write_forward: sequence-length mismatch between positions and values");
  } else {
    TORCH_CHECK(positions.size(0) == values.size(2),
                "paged_kv_write_forward: sequence-length mismatch between positions and values");
  }

  auto block_table_long = block_table.to(torch::kLong).contiguous();
  auto positions_long = positions.to(torch::kLong).contiguous();
  if (block_table_long.numel() > 0) {
    const auto min_page = block_table_long.min().item<int64_t>();
    const auto max_page = block_table_long.max().item<int64_t>();
    TORCH_CHECK(min_page >= 0, "paged_kv_write_forward: block_table entries must be non-negative");
    TORCH_CHECK(max_page < pages.size(0), "paged_kv_write_forward: block_table entries exceed page pool size");
  }
  if (positions_long.numel() > 0) {
    const auto min_pos = positions_long.min().item<int64_t>();
    const auto max_pos = positions_long.max().item<int64_t>();
    TORCH_CHECK(min_pos >= 0, "paged_kv_write_forward: positions must be non-negative");
    TORCH_CHECK(max_pos < block_table_long.size(1) * pages.size(2),
                "paged_kv_write_forward: positions exceed logical block-table capacity");
  }

#if MODEL_STACK_WITH_CUDA
  if (pages.is_cuda() && block_table_long.is_cuda() && positions_long.is_cuda() && values.is_cuda() &&
      HasCudaKvCacheKernel()) {
    return CudaPagedKvWriteForward(pages, block_table_long, positions_long, values);
  }
#endif

  auto pages_out = pages.contiguous();
  const auto write_seq = values.size(2);
  for (int64_t b = 0; b < values.size(0); ++b) {
    for (int64_t t = 0; t < write_seq; ++t) {
      const auto pos = positions_long.dim() == 1 ? positions_long[t].item<int64_t>()
                                                 : positions_long.index({b, t}).item<int64_t>();
      const auto block_idx = pos / pages.size(2);
      const auto page_offset = pos % pages.size(2);
      const auto page_id = block_table_long.index({b, block_idx}).item<int64_t>();
      pages_out.index_put_({page_id, at::indexing::Slice(), page_offset, at::indexing::Slice()},
                           values.index({b, at::indexing::Slice(), t, at::indexing::Slice()}));
    }
  }
  return pages_out;
}

torch::Tensor PrepareAttentionMaskForward(
    const torch::Tensor& mask,
    int64_t batch_size,
    int64_t num_heads,
    int64_t tgt_len,
    int64_t src_len,
    const c10::optional<torch::Tensor>& position_ids) {
  TORCH_CHECK(mask.defined(), "prepare_attention_mask_forward: mask must be defined");
  TORCH_CHECK(batch_size > 0 && num_heads > 0 && tgt_len >= 0 && src_len >= 0,
              "prepare_attention_mask_forward: invalid target dimensions");
  TORCH_CHECK(mask.dim() >= 2 && mask.dim() <= 4, "prepare_attention_mask_forward: mask rank must be between 2 and 4");

  torch::Tensor prepared;
  if (mask.scalar_type() == torch::kBool) {
    auto masked = torch::full(mask.sizes(), -std::numeric_limits<float>::infinity(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(mask.device()));
    auto zeros = torch::zeros(mask.sizes(), torch::TensorOptions().dtype(torch::kFloat32).device(mask.device()));
    prepared = torch::where(mask, masked, zeros);
  } else {
    prepared = mask.to(torch::kFloat32);
  }

  if (prepared.size(-1) != src_len) {
    TORCH_CHECK(prepared.size(-1) >= src_len, "prepare_attention_mask_forward: source length exceeds mask width");
    prepared = prepared.slice(-1, 0, src_len);
  }

  if (prepared.dim() == 4 && prepared.size(1) == 1) {
    prepared = prepared.expand({batch_size, num_heads, tgt_len, src_len});
  }

  if (position_ids.has_value() && position_ids.value().defined()) {
    auto pos = position_ids.value();
    if (pos.dim() == 2) {
      auto pos_idx = pos.clamp_min(0).clamp_max(src_len > 0 ? src_len - 1 : 0);
      auto zeros = torch::zeros({batch_size, num_heads, tgt_len, 1},
                                torch::TensorOptions().dtype(prepared.scalar_type()).device(prepared.device()));
      auto idx = pos_idx.view({batch_size, 1, tgt_len, 1}).expand({batch_size, num_heads, tgt_len, 1});
      prepared = prepared.scatter(3, idx, zeros);
    }
  }

  return prepared;
}

torch::Tensor ResolvePositionIdsForward(
    int64_t batch_size,
    int64_t seq_len,
    const torch::Tensor& reference,
    const c10::optional<torch::Tensor>& attention_mask,
    const c10::optional<torch::Tensor>& cache_position,
    int64_t past_length) {
  TORCH_CHECK(reference.defined(), "resolve_position_ids_forward: reference must be defined");
  TORCH_CHECK(batch_size > 0 && seq_len >= 0, "resolve_position_ids_forward: invalid batch_size or seq_len");
  auto options = torch::TensorOptions().dtype(torch::kLong).device(reference.device());
  if (cache_position.has_value() && cache_position.value().defined()) {
    return cache_position.value().to(options).view({1, -1}).expand({batch_size, -1});
  }
  if (past_length > 0) {
    auto base = torch::arange(past_length, past_length + seq_len, options);
    return base.view({1, -1}).expand({batch_size, -1});
  }
  if (attention_mask.has_value() && attention_mask.value().defined()) {
    auto lengths = attention_mask.value().to(torch::kLong).sum(-1).view({batch_size, 1});
    auto starts = (lengths - seq_len).clamp_min(0);
    auto base = torch::arange(seq_len, options).view({1, seq_len});
    return base + starts;
  }
  auto base = torch::arange(seq_len, options);
  return base.view({1, -1}).expand({batch_size, -1});
}

torch::Tensor CreateCausalMaskForward(
    const torch::Tensor& reference,
    const c10::optional<torch::Tensor>& attention_mask,
    const c10::optional<torch::Tensor>& cache_position,
    const c10::optional<torch::Tensor>& position_ids) {
  TORCH_CHECK(reference.defined(), "create_causal_mask_forward: reference must be defined");
  TORCH_CHECK(reference.dim() == 3, "create_causal_mask_forward: reference must have shape (B, T, D)");
  const auto batch_size = reference.size(0);
  const auto tgt_len = reference.size(1);
  auto device = reference.device();

  int64_t src_len = tgt_len;
  if (position_ids.has_value() && position_ids.value().defined()) {
    try {
      src_len = position_ids.value().to(torch::kLong).max().item<int64_t>() + 1;
    } catch (...) {
      src_len = std::max<int64_t>(tgt_len, 1);
    }
  } else if (cache_position.has_value() && cache_position.value().defined()) {
    try {
      src_len = cache_position.value().to(torch::kLong).max().item<int64_t>() + 1;
    } catch (...) {
      src_len = std::max<int64_t>(tgt_len, 1);
    }
  }
  if (attention_mask.has_value() && attention_mask.value().defined() && attention_mask.value().size(-1) > src_len) {
    src_len = attention_mask.value().size(-1);
  }

  torch::Tensor causal_bool;
  if (position_ids.has_value() && position_ids.value().defined()) {
    auto pos = position_ids.value().to(torch::kLong);
    if (pos.dim() == 2) {
      pos = pos[0];
    }
    auto qpos = pos.view({tgt_len, 1});
    auto kpos = torch::arange(src_len, torch::TensorOptions().dtype(qpos.scalar_type()).device(device)).view({1, src_len});
    causal_bool = kpos.gt(qpos).view({1, 1, tgt_len, src_len});
  } else {
    auto causal = torch::ones({src_len, src_len}, torch::TensorOptions().dtype(torch::kBool).device(device)).triu(1);
    causal_bool = causal.slice(0, std::max<int64_t>(src_len - tgt_len, 0), src_len).view({1, 1, tgt_len, src_len});
  }

  torch::Tensor combined_bool = causal_bool;
  if (attention_mask.has_value() && attention_mask.value().defined()) {
    auto pad = attention_mask.value();
    if (pad.size(-1) < src_len) {
      auto pad_fill = torch::ones({pad.size(0), src_len - pad.size(-1)},
                                  torch::TensorOptions().dtype(pad.scalar_type()).device(pad.device()));
      pad = torch::cat({pad, pad_fill}, -1);
    } else if (pad.size(-1) > src_len) {
      pad = pad.slice(-1, 0, src_len);
    }
    auto pad_bool = pad.eq(0).view({batch_size, 1, 1, src_len});
    combined_bool = combined_bool.logical_or(pad_bool);
  }

  auto masked = torch::full(combined_bool.sizes(), -std::numeric_limits<float>::infinity(),
                            torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto zeros = torch::zeros(combined_bool.sizes(), torch::TensorOptions().dtype(torch::kFloat32).device(device));
  return torch::where(combined_bool, masked, zeros);
}

std::vector<torch::Tensor> ResolveRotaryEmbeddingForward(
    const torch::Tensor& reference,
    int64_t head_dim,
    double base_theta,
    double attention_scaling,
    const c10::optional<torch::Tensor>& position_ids) {
  TORCH_CHECK(reference.defined(), "resolve_rotary_embedding_forward: reference must be defined");
  TORCH_CHECK(reference.dim() == 3, "resolve_rotary_embedding_forward: reference must have shape (B, T, D)");
  TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0, "resolve_rotary_embedding_forward: head_dim must be positive and even");
  TORCH_CHECK(std::isfinite(base_theta) && base_theta > 0.0,
              "resolve_rotary_embedding_forward: base_theta must be positive and finite");
  TORCH_CHECK(std::isfinite(attention_scaling),
              "resolve_rotary_embedding_forward: attention_scaling must be finite");

  const auto seq_len = reference.size(1);
  auto device = reference.device();
  int64_t needed = seq_len;
  torch::Tensor gather_pos;
  if (position_ids.has_value() && position_ids.value().defined()) {
    gather_pos = position_ids.value().to(torch::TensorOptions().dtype(torch::kLong).device(device));
    if (gather_pos.dim() == 2) {
      gather_pos = gather_pos[0];
    }
    if (gather_pos.numel() > 0) {
      needed = gather_pos.max().item<int64_t>() + 1;
    }
  }

  auto t = torch::arange(needed, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto idx = torch::arange(0, head_dim, 2, torch::TensorOptions().dtype(torch::kInt64).device(device)).to(torch::kFloat32);
  auto inv_freq = torch::pow(torch::full({idx.size(0)}, base_theta, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
                             -(idx / static_cast<double>(head_dim)));
  auto freqs = torch::einsum("t,d->td", {t, inv_freq});
  auto emb = torch::cat({freqs, freqs}, -1);
  auto cos = torch::cos(emb);
  auto sin = torch::sin(emb);
  if (attention_scaling != 1.0) {
    cos = cos * attention_scaling;
    sin = sin * attention_scaling;
  }
  cos = cos.to(reference.scalar_type());
  sin = sin.to(reference.scalar_type());
  if (gather_pos.defined()) {
    cos = at::index_select(cos, 0, gather_pos.reshape({-1})).view({seq_len, head_dim});
    sin = at::index_select(sin, 0, gather_pos.reshape({-1})).view({seq_len, head_dim});
  } else {
    cos = cos.slice(0, 0, seq_len);
    sin = sin.slice(0, 0, seq_len);
  }
  return {cos, sin};
}

torch::Tensor NativeAttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale) {
  TORCH_CHECK(q.defined() && k.defined() && v.defined(), "attention_forward: q, k, v must be defined");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "attention_forward: q, k, v must be rank-4");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "attention_forward: batch mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "attention_forward: key/value head mismatch");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "attention_forward: query heads must be a multiple of kv heads");
  TORCH_CHECK(k.size(2) == v.size(2), "attention_forward: source-length mismatch");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "attention_forward: head_dim mismatch");

#if MODEL_STACK_WITH_CUDA
  if (q.is_cuda() && k.is_cuda() && v.is_cuda() && HasCudaAttentionKernel()) {
    return CudaAttentionForward(q, k, v, attn_mask, is_causal, scale);
  }
#endif
  return ReferenceAttentionForward(q, k, v, attn_mask, is_causal, scale);
}

py::dict NativeAttentionPlanInfo(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal) {
  TORCH_CHECK(q.defined() && k.defined() && v.defined(), "attention_plan_info: q, k, v must be defined");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "attention_plan_info: q, k, v must be rank-4");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "attention_plan_info: batch mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "attention_plan_info: key/value head mismatch");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "attention_plan_info: query heads must be a multiple of kv heads");
  TORCH_CHECK(k.size(2) == v.size(2), "attention_plan_info: source-length mismatch");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "attention_plan_info: head_dim mismatch");

  const t10::desc::AttentionDesc desc{
      q.scalar_type(),
      q.size(0),
      q.size(1),
      k.size(1),
      q.size(2),
      k.size(2),
      q.size(3),
      t10::desc::ResolveAttentionMaskKind(attn_mask, q.scalar_type()),
      t10::desc::ResolveAttentionPhase(q.size(2)),
      t10::desc::ResolveAttentionHeadMode(q.size(1), k.size(1)),
      t10::desc::AttentionLayoutKind::kBHSD,
      t10::desc::AttentionLayoutKind::kBHSD,
      is_causal};
  const auto plan = t10::policy::ResolveAttentionPlan(desc);

  py::dict info;
  info["backend"] = CanUseCudaAttentionPath(q, k, v, attn_mask) ? "cuda" : "aten";
  info["kernel"] = AttentionKernelName(plan.kernel);
  info["phase"] = AttentionPhaseName(desc.phase);
  info["head_mode"] = AttentionHeadModeName(desc.head_mode);
  info["mask_kind"] = AttentionMaskKindName(desc.mask_kind);
  info["row_reduce_threads"] = plan.row_reduce_threads;
  info["head_dim_bucket"] = plan.head_dim_bucket;
  info["batch"] = desc.batch;
  info["q_heads"] = desc.q_heads;
  info["kv_heads"] = desc.kv_heads;
  info["q_len"] = desc.q_len;
  info["kv_len"] = desc.kv_len;
  info["head_dim"] = desc.head_dim;
  info["causal"] = desc.causal;
  return info;
}

torch::Tensor TemperatureForward(const torch::Tensor& logits, double tau) {
  TORCH_CHECK(logits.defined(), "temperature_forward: logits must be defined");
  TORCH_CHECK(std::isfinite(tau), "temperature_forward: tau must be finite");
#if MODEL_STACK_WITH_CUDA
  if (logits.is_cuda() && HasCudaSamplingKernel()) {
    return CudaTemperatureForward(logits, tau);
  }
#endif
  const auto denom = std::max(tau, 1e-8);
  return logits / denom;
}

torch::Tensor TopkMaskForward(const torch::Tensor& logits, int64_t k) {
  TORCH_CHECK(logits.defined(), "topk_mask_forward: logits must be defined");
  TORCH_CHECK(logits.dim() >= 1, "topk_mask_forward: logits must have rank >= 1");
  const auto vocab = logits.size(-1);
  TORCH_CHECK(vocab > 0, "topk_mask_forward: logits last dimension must be non-empty");
  TORCH_CHECK(k > 0, "topk_mask_forward: k must be positive");
#if MODEL_STACK_WITH_CUDA
  if (logits.is_cuda() && logits.dim() == 2 && HasCudaSamplingKernel()) {
    return CudaTopkMaskForward(logits, k);
  }
#endif
  const auto bounded_k = std::min<int64_t>(k, vocab);
  auto topk = std::get<0>(torch::topk(logits, bounded_k, -1));
  auto kth = topk.select(-1, bounded_k - 1).unsqueeze(-1);
  auto mask = logits < kth;
  auto equals = logits == kth;
  return mask.logical_and(equals.logical_not());
}

torch::Tensor ToppMaskForward(const torch::Tensor& logits, double p) {
  TORCH_CHECK(logits.defined(), "topp_mask_forward: logits must be defined");
  TORCH_CHECK(logits.dim() >= 1, "topp_mask_forward: logits must have rank >= 1");
#if MODEL_STACK_WITH_CUDA
  if (logits.is_cuda() && logits.dim() == 2 && HasCudaSamplingKernel()) {
    return CudaToppMaskForward(logits, p);
  }
#endif
  auto probs = torch::softmax(logits.to(torch::kFloat32), -1);
  auto sorted = torch::sort(probs, -1, true);
  auto sorted_probs = std::get<0>(sorted);
  auto sorted_idx = std::get<1>(sorted);
  auto cum = torch::cumsum(sorted_probs, -1);
  auto cutoff = cum > p;
  cutoff.select(-1, 0).fill_(false);
  auto mask = torch::zeros_like(cutoff, cutoff.options().dtype(torch::kBool));
  return mask.scatter(-1, sorted_idx, cutoff);
}

torch::Tensor PresenceFrequencyPenaltyForward(
    const torch::Tensor& logits,
    const torch::Tensor& counts,
    double alpha_presence,
    double alpha_frequency) {
  TORCH_CHECK(logits.defined() && counts.defined(), "presence_frequency_penalty_forward: tensors must be defined");
  TORCH_CHECK(logits.sizes() == counts.sizes(), "presence_frequency_penalty_forward: logits/counts shape mismatch");
#if MODEL_STACK_WITH_CUDA
  if (logits.is_cuda() && counts.is_cuda() && HasCudaSamplingKernel()) {
    return CudaPresenceFrequencyPenaltyForward(
        logits, counts, alpha_presence, alpha_frequency);
  }
#endif
  auto penalty =
      alpha_presence * counts.gt(0).to(logits.scalar_type()) +
      alpha_frequency * counts.to(logits.scalar_type());
  return logits - penalty;
}

torch::Tensor ApplySamplingMaskForward(
    const torch::Tensor& logits,
    const c10::optional<torch::Tensor>& topk_mask,
    const c10::optional<torch::Tensor>& topp_mask,
    const c10::optional<torch::Tensor>& no_repeat_mask) {
  TORCH_CHECK(logits.defined(), "apply_sampling_mask_forward: logits must be defined");
  const auto validate_mask = [&](const c10::optional<torch::Tensor>& mask, const char* name) {
    if (!mask.has_value() || !mask.value().defined()) {
      return;
    }
    TORCH_CHECK(mask.value().sizes() == logits.sizes(), "apply_sampling_mask_forward: ", name, " shape mismatch");
    TORCH_CHECK(mask.value().scalar_type() == torch::kBool, "apply_sampling_mask_forward: ", name, " must be bool");
  };
  validate_mask(topk_mask, "topk_mask");
  validate_mask(topp_mask, "topp_mask");
  validate_mask(no_repeat_mask, "no_repeat_mask");
#if MODEL_STACK_WITH_CUDA
  if (logits.is_cuda() && HasCudaSamplingKernel()) {
    return CudaApplySamplingMaskForward(logits, topk_mask, topp_mask, no_repeat_mask);
  }
#endif
  auto combined = torch::zeros_like(logits, logits.options().dtype(torch::kBool));
  if (topk_mask.has_value() && topk_mask.value().defined()) {
    combined = combined.logical_or(topk_mask.value());
  }
  if (topp_mask.has_value() && topp_mask.value().defined()) {
    combined = combined.logical_or(topp_mask.value());
  }
  if (no_repeat_mask.has_value() && no_repeat_mask.value().defined()) {
    combined = combined.logical_or(no_repeat_mask.value());
  }
  if (!combined.any().item<bool>()) {
    return logits;
  }
  auto out = logits.clone();
  const double fill_value = logits.is_floating_point() ? std::numeric_limits<float>::lowest() : -1e9;
  return out.masked_fill(combined, fill_value);
}

torch::Tensor SampleWithPoliciesForward(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    bool do_sample,
    double temperature,
    const c10::optional<int64_t>& top_k,
    const c10::optional<double>& top_p,
    int64_t no_repeat_ngram,
    double repetition_penalty,
    double presence_penalty,
    double frequency_penalty) {
  TORCH_CHECK(logits.defined(), "sample_with_policies_forward: logits must be defined");
  TORCH_CHECK(logits.dim() == 2, "sample_with_policies_forward: logits must be rank-2 (B, V)");
  TORCH_CHECK(token_ids.defined(), "sample_with_policies_forward: token_ids must be defined");
  TORCH_CHECK(token_ids.dim() == 2, "sample_with_policies_forward: token_ids must be rank-2 (B, T)");
  TORCH_CHECK(logits.size(0) == token_ids.size(0), "sample_with_policies_forward: batch mismatch");
#if MODEL_STACK_WITH_CUDA
  if (logits.is_cuda() && token_ids.is_cuda() && HasCudaSamplingKernel()) {
    return CudaSampleWithPoliciesForward(
        logits,
        token_ids,
        do_sample,
        temperature,
        top_k,
        top_p,
        no_repeat_ngram,
        repetition_penalty,
        presence_penalty,
        frequency_penalty);
  }
#endif

  auto x = logits;
  if (repetition_penalty != 1.0) {
    x = RepetitionPenaltyForward(x, token_ids, repetition_penalty);
  }
  if (do_sample && temperature != 1.0) {
    x = TemperatureForward(x, temperature);
  }

  c10::optional<torch::Tensor> topk_mask = c10::nullopt;
  c10::optional<torch::Tensor> topp_mask = c10::nullopt;
  c10::optional<torch::Tensor> no_repeat_mask = c10::nullopt;
  if (do_sample) {
    if (top_k.has_value() && top_k.value() > 0) {
      topk_mask = TopkMaskForward(x, top_k.value());
    }
    if (top_p.has_value() && top_p.value() > 0.0 && top_p.value() < 1.0) {
      topp_mask = ToppMaskForward(x, top_p.value());
    }
  }
  if (no_repeat_ngram > 0) {
    no_repeat_mask = NoRepeatNgramMaskForward(token_ids, x.size(-1), no_repeat_ngram);
  }
  if (topk_mask.has_value() || topp_mask.has_value() || no_repeat_mask.has_value()) {
    x = ApplySamplingMaskForward(x, topk_mask, topp_mask, no_repeat_mask);
  }
  if (presence_penalty != 0.0 || frequency_penalty != 0.0) {
    auto counts = TokenCountsForward(token_ids, x.size(-1), x.scalar_type());
    x = PresenceFrequencyPenaltyForward(x, counts, presence_penalty, frequency_penalty);
  }
  return SampleNextTokenForward(x, do_sample);
}

torch::Tensor NoRepeatNgramMaskForward(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    int64_t n) {
  TORCH_CHECK(token_ids.defined(), "no_repeat_ngram_mask_forward: token_ids must be defined");
  TORCH_CHECK(token_ids.dim() == 2, "no_repeat_ngram_mask_forward: token_ids must be rank-2 (B, T)");
  TORCH_CHECK(vocab_size > 0, "no_repeat_ngram_mask_forward: vocab_size must be positive");
#if MODEL_STACK_WITH_CUDA
  if (token_ids.is_cuda() && HasCudaSamplingKernel()) {
    return CudaNoRepeatNgramMaskForward(token_ids, vocab_size, n);
  }
#endif
  auto token_ids_contig = token_ids.to(torch::kLong).contiguous();
  auto mask = torch::zeros(
      {token_ids_contig.size(0), vocab_size},
      torch::TensorOptions().dtype(torch::kBool).device(token_ids.device()));
  if (token_ids_contig.numel() == 0 || n <= 0 || token_ids_contig.size(1) < n) {
    return mask;
  }

  const auto* ids_ptr = token_ids_contig.data_ptr<int64_t>();
  auto* mask_ptr = mask.data_ptr<bool>();
  const auto batch_size = token_ids_contig.size(0);
  const auto seq_len = token_ids_contig.size(1);
  if (n == 1) {
    for (int64_t b = 0; b < batch_size; ++b) {
      const int64_t base = b * seq_len;
      const int64_t mask_base = b * vocab_size;
      for (int64_t pos = 0; pos < seq_len; ++pos) {
        const int64_t token = ids_ptr[base + pos];
        if (token >= 0 && token < vocab_size) {
          mask_ptr[mask_base + token] = true;
        }
      }
    }
    return mask;
  }

  const int64_t prefix_len = n - 1;
  for (int64_t b = 0; b < batch_size; ++b) {
    const int64_t base = b * seq_len;
    const int64_t mask_base = b * vocab_size;
    const int64_t recent_base = base + seq_len - prefix_len;
    for (int64_t start = 0; start <= seq_len - n; ++start) {
      bool match = true;
      for (int64_t offset = 0; offset < prefix_len; ++offset) {
        if (ids_ptr[base + start + offset] != ids_ptr[recent_base + offset]) {
          match = false;
          break;
        }
      }
      if (!match) {
        continue;
      }
      const int64_t token = ids_ptr[base + start + prefix_len];
      if (token >= 0 && token < vocab_size) {
        mask_ptr[mask_base + token] = true;
      }
    }
  }
  return mask;
}

torch::Tensor SampleNextTokenForward(const torch::Tensor& logits, bool do_sample) {
  TORCH_CHECK(logits.defined(), "sample_next_token_forward: logits must be defined");
  TORCH_CHECK(logits.dim() == 2, "sample_next_token_forward: logits must be rank-2 (B, V)");
#if MODEL_STACK_WITH_CUDA
  if (logits.is_cuda() && HasCudaSamplingKernel()) {
    return do_sample ? CudaMultinomialSampleForward(logits) : CudaGreedyNextTokenForward(logits);
  }
#endif
  if (do_sample) {
    auto probs = torch::softmax(logits.to(torch::kFloat32), -1);
    return torch::multinomial(probs, 1);
  }
  return std::get<1>(torch::max(logits, -1, true));
}

torch::Tensor RepetitionPenaltyForward(
    const torch::Tensor& logits,
    const torch::Tensor& token_ids,
    double penalty) {
  TORCH_CHECK(logits.defined() && token_ids.defined(), "repetition_penalty_forward: tensors must be defined");
  TORCH_CHECK(logits.dim() == 2, "repetition_penalty_forward: logits must be rank-2 (B, V)");
  TORCH_CHECK(token_ids.dim() == 2, "repetition_penalty_forward: token_ids must be rank-2 (B, T)");
  TORCH_CHECK(logits.size(0) == token_ids.size(0), "repetition_penalty_forward: batch mismatch");
  TORCH_CHECK(std::isfinite(penalty) && penalty > 0.0, "repetition_penalty_forward: penalty must be positive and finite");
#if MODEL_STACK_WITH_CUDA
  if (logits.is_cuda() && token_ids.is_cuda() && HasCudaSamplingKernel()) {
    return CudaRepetitionPenaltyForward(logits, token_ids, penalty);
  }
#endif

  auto out = logits.clone();
  if (penalty == 1.0 || token_ids.size(1) == 0) {
    return out;
  }

  for (int64_t b = 0; b < token_ids.size(0); ++b) {
    auto row_ids = token_ids[b].to(torch::kLong).reshape({-1});
    auto unique_result = at::_unique2(row_ids, true, false, false);
    auto unique_ids = std::get<0>(unique_result);
    if (unique_ids.numel() == 0) {
      continue;
    }
    auto row = out[b];
    auto values = at::index_select(row, 0, unique_ids);
    auto positive = values > 0;
    auto adjusted = torch::where(positive, values / penalty, values * penalty);
    row.index_put_({at::indexing::TensorIndex(unique_ids)}, adjusted);
  }
  return out;
}

torch::Tensor TokenCountsForward(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    torch::ScalarType counts_dtype) {
  TORCH_CHECK(token_ids.defined(), "token_counts_forward: token_ids must be defined");
  TORCH_CHECK(token_ids.dim() == 2, "token_counts_forward: token_ids must be rank-2 (B, T)");
  TORCH_CHECK(vocab_size > 0, "token_counts_forward: vocab_size must be positive");
#if MODEL_STACK_WITH_CUDA
  if (token_ids.is_cuda() && HasCudaSamplingKernel()) {
    return CudaTokenCountsForward(token_ids, vocab_size, counts_dtype);
  }
#endif
  auto counts = torch::zeros(
      {token_ids.size(0), vocab_size},
      torch::TensorOptions().dtype(counts_dtype).device(token_ids.device()));
  if (token_ids.numel() == 0) {
    return counts;
  }
  auto ones = torch::ones(token_ids.sizes(), torch::TensorOptions().dtype(counts_dtype).device(token_ids.device()));
  return counts.scatter_add(1, token_ids.to(torch::kLong), ones);
}

py::tuple BeamSearchStepForward(
    const torch::Tensor& beams,
    const torch::Tensor& logits,
    const torch::Tensor& raw_scores,
    const torch::Tensor& finished,
    const torch::Tensor& lengths,
    int64_t beam_size,
    int64_t eos_id,
    int64_t pad_id) {
  TORCH_CHECK(beams.defined(), "beam_search_step_forward: beams must be defined");
  TORCH_CHECK(logits.defined(), "beam_search_step_forward: logits must be defined");
  TORCH_CHECK(raw_scores.defined(), "beam_search_step_forward: raw_scores must be defined");
  TORCH_CHECK(finished.defined(), "beam_search_step_forward: finished must be defined");
  TORCH_CHECK(lengths.defined(), "beam_search_step_forward: lengths must be defined");
  TORCH_CHECK(beam_size > 0, "beam_search_step_forward: beam_size must be positive");
  TORCH_CHECK(beams.dim() == 2, "beam_search_step_forward: beams must be rank-2 (B*beam, T)");
  TORCH_CHECK(raw_scores.dim() == 2, "beam_search_step_forward: raw_scores must be rank-2 (B, beam)");
  TORCH_CHECK(finished.dim() == 2, "beam_search_step_forward: finished must be rank-2 (B, beam)");
  TORCH_CHECK(lengths.dim() == 2, "beam_search_step_forward: lengths must be rank-2 (B, beam)");
  TORCH_CHECK(raw_scores.size(1) == beam_size, "beam_search_step_forward: raw_scores beam dimension mismatch");
  TORCH_CHECK(finished.sizes() == raw_scores.sizes(), "beam_search_step_forward: finished shape mismatch");
  TORCH_CHECK(lengths.sizes() == raw_scores.sizes(), "beam_search_step_forward: lengths shape mismatch");
  const auto batch_size = raw_scores.size(0);
  TORCH_CHECK(beams.size(0) == batch_size * beam_size, "beam_search_step_forward: beams batch mismatch");

  torch::Tensor next_logits;
  if (logits.dim() == 3) {
    TORCH_CHECK(logits.size(0) == batch_size * beam_size, "beam_search_step_forward: logits batch mismatch");
    TORCH_CHECK(logits.size(1) > 0, "beam_search_step_forward: logits time dimension must be non-empty");
    next_logits = logits.select(1, logits.size(1) - 1);
  } else if (logits.dim() == 2) {
    TORCH_CHECK(logits.size(0) == batch_size * beam_size, "beam_search_step_forward: logits batch mismatch");
    next_logits = logits;
  } else {
    TORCH_CHECK(false, "beam_search_step_forward: logits must be rank-2 or rank-3");
  }
  TORCH_CHECK(next_logits.size(1) > 0, "beam_search_step_forward: logits vocab dimension must be non-empty");
  TORCH_CHECK(pad_id >= 0 && pad_id < next_logits.size(1), "beam_search_step_forward: pad_id out of range");

  auto logp = torch::log_softmax(next_logits, -1).view({batch_size, beam_size, next_logits.size(1)});
  auto finished_bool = finished.to(torch::kBool);
  if (finished_bool.any().item<bool>()) {
    logp = logp.masked_fill(finished_bool.unsqueeze(-1), -std::numeric_limits<float>::infinity());
    auto pad_scores = logp.select(2, pad_id);
    pad_scores.copy_(torch::where(finished_bool, torch::zeros_like(pad_scores), pad_scores));
  }

  auto candidate_scores = raw_scores.unsqueeze(-1) + logp;
  auto flat_scores = candidate_scores.view({batch_size, beam_size * next_logits.size(1)});
  auto topk = torch::topk(flat_scores, beam_size, -1);
  auto best_raw_scores = std::get<0>(topk);
  auto best_idx = std::get<1>(topk);

  auto parent_beams = torch::floor_divide(best_idx, next_logits.size(1));
  auto next_tokens = best_idx.remainder(next_logits.size(1));

  auto prev_beams = beams.view({batch_size, beam_size, beams.size(1)});
  auto gather_index = parent_beams.unsqueeze(-1).expand({batch_size, beam_size, prev_beams.size(2)});
  auto gathered = prev_beams.gather(1, gather_index);

  auto parent_finished = finished_bool.gather(1, parent_beams);
  auto appended_tokens = torch::where(parent_finished, torch::full_like(next_tokens, pad_id), next_tokens);
  auto next_beams = torch::cat(
      {gathered.reshape({batch_size * beam_size, gathered.size(2)}), appended_tokens.reshape({batch_size * beam_size, 1})},
      1);

  auto parent_lengths = lengths.gather(1, parent_beams);
  auto next_lengths = parent_lengths + parent_finished.logical_not().to(lengths.scalar_type());
  auto next_finished = parent_finished.logical_or(next_tokens.eq(eos_id));
  return py::make_tuple(next_beams, best_raw_scores, next_finished, next_lengths);
}


py::tuple AppendTokensForward(
    const torch::Tensor& seq,
    const torch::Tensor& next_id,
    const c10::optional<torch::Tensor>& attention_mask) {
  TORCH_CHECK(seq.defined() && next_id.defined(), "append_tokens_forward: seq and next_id must be defined");
  TORCH_CHECK(seq.dim() == 2 && next_id.dim() == 2, "append_tokens_forward: seq and next_id must be rank-2");
  TORCH_CHECK(seq.size(0) == next_id.size(0), "append_tokens_forward: batch mismatch");
#if MODEL_STACK_WITH_CUDA
  if (seq.is_cuda() && next_id.is_cuda() && HasCudaAppendTokensKernel()) {
    auto next = CudaAppendTokensForward(seq, next_id, attention_mask);
    py::object next_mask = py::none();
    if (next.size() > 1 && next[1].defined()) {
      next_mask = py::cast(next[1]);
    }
    return py::make_tuple(next[0], next_mask);
  }
#endif
  auto next_seq = torch::cat({seq, next_id}, 1);
  py::object next_mask = py::none();
  if (attention_mask.has_value() && attention_mask.value().defined()) {
    auto ones = torch::ones(
        {next_id.size(0), next_id.size(1)},
        torch::TensorOptions().dtype(attention_mask.value().scalar_type()).device(attention_mask.value().device()));
    next_mask = py::cast(torch::cat({attention_mask.value(), ones}, 1));
  }
  return py::make_tuple(next_seq, next_mask);
}

py::tuple DecodePositionsForward(
    int64_t batch_size,
    int64_t seq_len,
    const torch::Tensor& reference) {
  TORCH_CHECK(reference.defined(), "decode_positions_forward: reference tensor must be defined");
  TORCH_CHECK(batch_size > 0 && seq_len > 0, "decode_positions_forward: batch_size and seq_len must be positive");
#if MODEL_STACK_WITH_CUDA
  if (reference.is_cuda() && HasCudaDecodePositionsKernel()) {
    auto next = CudaDecodePositionsForward(batch_size, seq_len, reference);
    return py::make_tuple(next[0], next[1]);
  }
#endif
  auto pos = torch::full(
      {batch_size, 1},
      seq_len - 1,
      torch::TensorOptions().dtype(torch::kLong).device(reference.device()));
  return py::make_tuple(pos, pos.view({-1}));
}

torch::Tensor LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const std::string& backend) {
  TORCH_CHECK(x.defined() && weight.defined(), "linear_forward: x and weight must be defined");
  TORCH_CHECK(x.dim() >= 2, "linear_forward: x must have rank >= 2");
  TORCH_CHECK(weight.dim() == 2, "linear_forward: weight must be rank-2");
  TORCH_CHECK(x.size(-1) == weight.size(1), "linear_forward: input feature size mismatch");
  if (bias.has_value() && bias.value().defined()) {
    const auto& b = bias.value();
    TORCH_CHECK(b.dim() == 1, "linear_forward: bias must be rank-1");
    TORCH_CHECK(b.size(0) == weight.size(0), "linear_forward: bias size mismatch");
  }
  const auto resolved_backend = ResolveLinearBackend(backend);
#if MODEL_STACK_WITH_CUDA
  if (resolved_backend == "cublaslt" && x.is_cuda()) {
    return CublasLtLinearForward(x, weight, bias);
  }
#endif
  return ReferenceLinearForward(x, weight, bias);
}

torch::Tensor MlpForward(
    const torch::Tensor& x,
    const torch::Tensor& w_in_weight,
    const c10::optional<torch::Tensor>& w_in_bias,
    const torch::Tensor& w_out_weight,
    const c10::optional<torch::Tensor>& w_out_bias,
    const std::string& activation,
    bool gated,
    const std::string& backend) {
  const auto normalized_backend = NormalizeBackendName(backend);
  const bool auto_backend = normalized_backend.empty() || normalized_backend == "auto";
  if (auto_backend && gated && x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
      ForceReferenceGatedMlpFp16()) {
    return ReferenceMlpForward(x, w_in_weight, w_in_bias, w_out_weight, w_out_bias, activation, gated);
  }
  const auto resolved_backend = ResolveLinearBackend(backend);
  auto hidden = LinearForward(x, w_in_weight, w_in_bias, resolved_backend);
  torch::Tensor projected;
  if (gated) {
    projected = ApplyGatedActivation(hidden, activation);
  } else {
    projected = ApplyActivation(hidden, activation);
  }
  return LinearForward(projected, w_out_weight, w_out_bias, resolved_backend);
}

std::vector<torch::Tensor> QkvProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& q_weight,
    const c10::optional<torch::Tensor>& q_bias,
    const torch::Tensor& k_weight,
    const c10::optional<torch::Tensor>& k_bias,
    const torch::Tensor& v_weight,
    const c10::optional<torch::Tensor>& v_bias,
    const std::string& backend) {
  const auto resolved_backend = ResolveLinearBackend(backend);
#if MODEL_STACK_WITH_CUDA
  if (resolved_backend == "cublaslt" && x.is_cuda()) {
    TORCH_CHECK(
        q_weight.size(1) == k_weight.size(1) && q_weight.size(1) == v_weight.size(1),
        "qkv_projection_forward: q/k/v input feature size mismatch");
    auto fused_weight = torch::cat({q_weight, k_weight, v_weight}, 0);
    c10::optional<torch::Tensor> fused_bias = c10::nullopt;
    if ((q_bias.has_value() && q_bias.value().defined()) ||
        (k_bias.has_value() && k_bias.value().defined()) ||
        (v_bias.has_value() && v_bias.value().defined())) {
      auto bias_options = fused_weight.options().device(x.device()).dtype(x.scalar_type());
      auto make_bias = [&](const c10::optional<torch::Tensor>& maybe_bias, int64_t size) {
        if (maybe_bias.has_value() && maybe_bias.value().defined()) {
          return maybe_bias.value().to(bias_options);
        }
        return torch::zeros({size}, bias_options);
      };
      fused_bias = torch::cat({
          make_bias(q_bias, q_weight.size(0)),
          make_bias(k_bias, k_weight.size(0)),
          make_bias(v_bias, v_weight.size(0)),
      });
    }
    auto fused = LinearForward(x, fused_weight, fused_bias, resolved_backend);
    auto q_size = q_weight.size(0);
    auto k_size = k_weight.size(0);
    auto v_size = v_weight.size(0);
    return {
        fused.slice(-1, 0, q_size),
        fused.slice(-1, q_size, q_size + k_size),
        fused.slice(-1, q_size + k_size, q_size + k_size + v_size),
    };
  }
#endif
  return {
      LinearForward(x, q_weight, q_bias, resolved_backend),
      LinearForward(x, k_weight, k_bias, resolved_backend),
      LinearForward(x, v_weight, v_bias, resolved_backend),
  };
}

py::tuple PackLinearWeightForward(
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(weight.defined(), "pack_linear_weight_forward: weight must be defined");
  TORCH_CHECK(weight.dim() == 2, "pack_linear_weight_forward: weight must be rank-2");
  py::object packed_bias = py::none();
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().dim() == 1, "pack_linear_weight_forward: bias must be rank-1");
    TORCH_CHECK(bias.value().size(0) == weight.size(0), "pack_linear_weight_forward: bias size mismatch");
    packed_bias = py::cast(bias.value().contiguous());
  }
  return py::make_tuple(weight.contiguous(), packed_bias);
}

py::tuple PackQkvWeightsForward(
    const torch::Tensor& q_weight,
    const c10::optional<torch::Tensor>& q_bias,
    const torch::Tensor& k_weight,
    const c10::optional<torch::Tensor>& k_bias,
    const torch::Tensor& v_weight,
    const c10::optional<torch::Tensor>& v_bias) {
  TORCH_CHECK(q_weight.defined() && k_weight.defined() && v_weight.defined(),
              "pack_qkv_weights_forward: q/k/v weights must be defined");
  TORCH_CHECK(q_weight.dim() == 2 && k_weight.dim() == 2 && v_weight.dim() == 2,
              "pack_qkv_weights_forward: q/k/v weights must be rank-2");
  TORCH_CHECK(q_weight.size(1) == k_weight.size(1) && q_weight.size(1) == v_weight.size(1),
              "pack_qkv_weights_forward: q/k/v input feature size mismatch");
  auto fused_weight = torch::cat({q_weight, k_weight, v_weight}, 0).contiguous();
  py::object fused_bias = py::none();
  if ((q_bias.has_value() && q_bias.value().defined()) ||
      (k_bias.has_value() && k_bias.value().defined()) ||
      (v_bias.has_value() && v_bias.value().defined())) {
    auto bias_options = fused_weight.options();
    auto make_bias = [&](const c10::optional<torch::Tensor>& maybe_bias, int64_t size) {
      if (maybe_bias.has_value() && maybe_bias.value().defined()) {
        TORCH_CHECK(maybe_bias.value().dim() == 1, "pack_qkv_weights_forward: bias must be rank-1");
        TORCH_CHECK(maybe_bias.value().size(0) == size, "pack_qkv_weights_forward: bias size mismatch");
        return maybe_bias.value().to(bias_options).contiguous();
      }
      return torch::zeros({size}, bias_options);
    };
    fused_bias = py::cast(torch::cat({
        make_bias(q_bias, q_weight.size(0)),
        make_bias(k_bias, k_weight.size(0)),
        make_bias(v_bias, v_weight.size(0)),
    }).contiguous());
  }
  return py::make_tuple(
      fused_weight,
      fused_bias,
      q_weight.size(0),
      k_weight.size(0),
      v_weight.size(0));
}

std::vector<torch::Tensor> QkvPackedHeadsProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const c10::optional<torch::Tensor>& packed_bias,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads,
    const std::string& backend) {
  TORCH_CHECK(packed_weight.defined(), "qkv_packed_heads_projection_forward: packed_weight must be defined");
  TORCH_CHECK(packed_weight.dim() == 2, "qkv_packed_heads_projection_forward: packed_weight must be rank-2");
  TORCH_CHECK(q_size > 0 && k_size > 0 && v_size > 0,
              "qkv_packed_heads_projection_forward: q/k/v sizes must be positive");
  TORCH_CHECK(packed_weight.size(0) == q_size + k_size + v_size,
              "qkv_packed_heads_projection_forward: packed weight row count mismatch");
  if (packed_bias.has_value() && packed_bias.value().defined()) {
    TORCH_CHECK(packed_bias.value().dim() == 1, "qkv_packed_heads_projection_forward: packed_bias must be rank-1");
    TORCH_CHECK(packed_bias.value().size(0) == packed_weight.size(0),
                "qkv_packed_heads_projection_forward: packed_bias size mismatch");
  }
  auto fused = LinearForward(x, packed_weight, packed_bias, backend);
  return {
      SplitHeadsForward(fused.slice(-1, 0, q_size), q_heads),
      SplitHeadsForward(fused.slice(-1, q_size, q_size + k_size), kv_heads),
      SplitHeadsForward(fused.slice(-1, q_size + k_size, q_size + k_size + v_size), kv_heads),
  };
}

torch::Tensor EmbeddingForward(
    const torch::Tensor& weight,
    const torch::Tensor& indices,
    int64_t padding_idx) {
  TORCH_CHECK(weight.defined() && indices.defined(), "embedding_forward: weight and indices must be defined");
  TORCH_CHECK(weight.dim() == 2, "embedding_forward: weight must be rank-2");
  TORCH_CHECK(indices.scalar_type() == torch::kLong || indices.scalar_type() == torch::kInt,
              "embedding_forward: indices must be int32 or int64");
#if MODEL_STACK_WITH_CUDA
  if (weight.is_cuda() && indices.is_cuda() && HasCudaEmbeddingKernel()) {
    return CudaEmbeddingForward(weight, indices, padding_idx);
  }
#endif
  return at::embedding(weight, indices.to(torch::kLong), padding_idx, false, false);
}

torch::Tensor SplitHeadsForward(
    const torch::Tensor& x,
    int64_t num_heads) {
  TORCH_CHECK(x.defined(), "split_heads_forward: x must be defined");
  TORCH_CHECK(x.dim() == 3, "split_heads_forward: x must have shape (B, T, D)");
  TORCH_CHECK(num_heads > 0, "split_heads_forward: num_heads must be positive");
  TORCH_CHECK(x.size(2) % num_heads == 0, "split_heads_forward: model dim must be divisible by num_heads");
  const auto head_dim = x.size(2) / num_heads;
  return x.contiguous().view({x.size(0), x.size(1), num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous();
}

torch::Tensor MergeHeadsForward(const torch::Tensor& x) {
  TORCH_CHECK(x.defined(), "merge_heads_forward: x must be defined");
  TORCH_CHECK(x.dim() == 4, "merge_heads_forward: x must have shape (B, H, T, Dh)");
  return x.permute({0, 2, 1, 3}).contiguous().view({x.size(0), x.size(2), x.size(1) * x.size(3)});
}

std::vector<torch::Tensor> QkvHeadsProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& q_weight,
    const c10::optional<torch::Tensor>& q_bias,
    const torch::Tensor& k_weight,
    const c10::optional<torch::Tensor>& k_bias,
    const torch::Tensor& v_weight,
    const c10::optional<torch::Tensor>& v_bias,
    int64_t q_heads,
    int64_t kv_heads,
    const std::string& backend) {
  auto qkv = QkvProjectionForward(x, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, backend);
  return {
      SplitHeadsForward(qkv[0], q_heads),
      SplitHeadsForward(qkv[1], kv_heads),
      SplitHeadsForward(qkv[2], kv_heads),
  };
}

torch::Tensor HeadOutputProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const std::string& backend) {
  return LinearForward(MergeHeadsForward(x), weight, bias, backend);
}

std::string ResolveLinearBackendPy(const std::string& requested) {
  return ResolveLinearBackend(requested);
}

}  // namespace

PYBIND11_MODULE(_model_stack_native, m) {
  m.doc() = "Model-stack native C++/CUDA extension boundary.";
  py::class_<PagedKvLayerState, std::shared_ptr<PagedKvLayerState>>(m, "PagedKvLayerState")
      .def(py::init<int64_t, int64_t, int64_t, int64_t, const torch::Tensor&>(),
           py::arg("batch"),
           py::arg("heads"),
           py::arg("head_dim"),
           py::arg("page_size"),
           py::arg("example"))
      .def("reset", &PagedKvLayerState::Reset)
      .def("append",
           &PagedKvLayerState::Append,
           py::arg("k_chunk"),
           py::arg("v_chunk"),
           py::arg("block_ids") = py::none())
      .def("read_range",
           &PagedKvLayerState::ReadRange,
           py::arg("start"),
           py::arg("end"),
           py::arg("block_ids") = py::none())
      .def("read_last",
           &PagedKvLayerState::ReadLast,
           py::arg("keep"),
           py::arg("block_ids") = py::none())
      .def("compact", &PagedKvLayerState::Compact, py::arg("keep"))
      .def("k_pages", &PagedKvLayerState::KPages)
      .def("v_pages", &PagedKvLayerState::VPages)
      .def("block_table", &PagedKvLayerState::BlockTable)
      .def("lengths", &PagedKvLayerState::Lengths)
      .def("page_count", &PagedKvLayerState::PageCount);
  py::class_<PagedKvCacheState, std::shared_ptr<PagedKvCacheState>>(m, "PagedKvCacheState")
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, const torch::Tensor&>(),
           py::arg("batch"),
           py::arg("layers"),
           py::arg("heads"),
           py::arg("head_dim"),
           py::arg("page_size"),
           py::arg("example"))
      .def("reset", &PagedKvCacheState::Reset)
      .def("reset_layer", &PagedKvCacheState::ResetLayer, py::arg("layer_idx"))
      .def("append",
           &PagedKvCacheState::Append,
           py::arg("layer_idx"),
           py::arg("k_chunk"),
           py::arg("v_chunk"),
           py::arg("block_ids") = py::none())
      .def("read_range",
           &PagedKvCacheState::ReadRange,
           py::arg("layer_idx"),
           py::arg("start"),
           py::arg("end"),
           py::arg("block_ids") = py::none())
      .def("read_last",
           &PagedKvCacheState::ReadLast,
           py::arg("layer_idx"),
           py::arg("keep"),
           py::arg("block_ids") = py::none())
      .def("compact", &PagedKvCacheState::Compact, py::arg("keep"))
      .def("compact_layer", &PagedKvCacheState::CompactLayer, py::arg("layer_idx"), py::arg("keep"))
      .def("k_pages", &PagedKvCacheState::KPages, py::arg("layer_idx"))
      .def("v_pages", &PagedKvCacheState::VPages, py::arg("layer_idx"))
      .def("block_table", &PagedKvCacheState::BlockTable, py::arg("layer_idx"))
      .def("lengths", &PagedKvCacheState::Lengths, py::arg("layer_idx"))
      .def("page_count", &PagedKvCacheState::PageCount, py::arg("layer_idx"))
      .def("max_length", &PagedKvCacheState::MaxLength, py::arg("layer_idx"))
      .def("num_layers", &PagedKvCacheState::NumLayers);
  m.def("runtime_info", &RuntimeInfo);
  m.def("has_op", &HasOp, py::arg("name"));
  m.def("resolve_linear_backend", &ResolveLinearBackendPy, py::arg("requested") = "auto");
  m.def("rms_norm_forward", &RmsNormForward, py::arg("x"), py::arg("weight") = py::none(),
        py::arg("eps") = 1e-6);
  m.def("add_rms_norm_forward", &AddRmsNormForward, py::arg("x"), py::arg("update"),
        py::arg("weight") = py::none(), py::arg("residual_scale") = 1.0, py::arg("eps") = 1e-6);
  m.def("residual_add_forward", &ResidualAddForward, py::arg("x"), py::arg("update"),
        py::arg("residual_scale") = 1.0);
  m.def("layer_norm_forward", &LayerNormForward, py::arg("x"), py::arg("weight") = py::none(),
        py::arg("bias") = py::none(), py::arg("eps") = 1e-5);
  m.def("add_layer_norm_forward", &AddLayerNormForward, py::arg("x"), py::arg("update"),
        py::arg("weight") = py::none(), py::arg("bias") = py::none(), py::arg("residual_scale") = 1.0,
        py::arg("eps") = 1e-5);
  m.def("apply_rotary_forward", &ApplyRotaryForward, py::arg("q"), py::arg("k"), py::arg("cos"),
        py::arg("sin"));
  m.def("kv_cache_append_forward", &KvCacheAppendForward, py::arg("k_cache") = py::none(),
        py::arg("v_cache") = py::none(), py::arg("k_new"), py::arg("v_new"));
  m.def("kv_cache_write_forward", &KvCacheWriteForward, py::arg("cache"), py::arg("chunk"), py::arg("start"));
  m.def("kv_cache_gather_forward", &KvCacheGatherForward, py::arg("cache"), py::arg("positions"));
  m.def("paged_kv_assign_blocks_forward", &PagedKvAssignBlocksForward, py::arg("block_table"),
        py::arg("block_ids"), py::arg("starts"), py::arg("total"), py::arg("page_size"),
        py::arg("next_page_id"));
  m.def("paged_kv_reserve_pages_forward", &PagedKvReservePagesForward, py::arg("pages"),
        py::arg("used_pages"), py::arg("needed_pages"));
  m.def("paged_kv_read_range_forward", &PagedKvReadRangeForward, py::arg("pages"), py::arg("block_table"),
        py::arg("lengths"), py::arg("start"), py::arg("end"));
  m.def("paged_kv_read_last_forward", &PagedKvReadLastForward, py::arg("pages"), py::arg("block_table"),
        py::arg("lengths"), py::arg("keep"));
  m.def("paged_kv_append_forward", &PagedKvAppendForward, py::arg("k_pages"), py::arg("v_pages"),
        py::arg("block_table"), py::arg("lengths"), py::arg("page_count"), py::arg("k_chunk"),
        py::arg("v_chunk"), py::arg("block_ids"));
  m.def("paged_kv_compact_forward", &PagedKvCompactForward, py::arg("k_pages"), py::arg("v_pages"),
        py::arg("block_table"), py::arg("lengths"), py::arg("keep"));
  m.def("paged_kv_gather_forward", &PagedKvGatherForward, py::arg("pages"), py::arg("block_table"),
        py::arg("positions"));
  m.def("paged_kv_write_forward", &PagedKvWriteForward, py::arg("pages"), py::arg("block_table"),
        py::arg("positions"), py::arg("values"));
  m.def("prepare_attention_mask_forward", &PrepareAttentionMaskForward, py::arg("mask"), py::arg("batch_size"),
        py::arg("num_heads"), py::arg("tgt_len"), py::arg("src_len"), py::arg("position_ids") = py::none());
  m.def("resolve_position_ids_forward", &ResolvePositionIdsForward, py::arg("batch_size"), py::arg("seq_len"),
        py::arg("reference"), py::arg("attention_mask") = py::none(), py::arg("cache_position") = py::none(),
        py::arg("past_length") = 0);
  m.def("create_causal_mask_forward", &CreateCausalMaskForward, py::arg("reference"),
        py::arg("attention_mask") = py::none(), py::arg("cache_position") = py::none(),
        py::arg("position_ids") = py::none());
  m.def("resolve_rotary_embedding_forward", &ResolveRotaryEmbeddingForward, py::arg("reference"),
        py::arg("head_dim"), py::arg("base_theta"), py::arg("attention_scaling") = 1.0,
        py::arg("position_ids") = py::none());
  m.def("attention_forward", &NativeAttentionForward, py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("attn_mask") = py::none(), py::arg("is_causal") = false, py::arg("scale") = py::none());
  m.def("attention_plan_info", &NativeAttentionPlanInfo, py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("attn_mask") = py::none(), py::arg("is_causal") = false);
  m.def("temperature_forward", &TemperatureForward, py::arg("logits"), py::arg("tau"));
  m.def("topk_mask_forward", &TopkMaskForward, py::arg("logits"), py::arg("k"));
  m.def("topp_mask_forward", &ToppMaskForward, py::arg("logits"), py::arg("p"));
  m.def("apply_sampling_mask_forward", &ApplySamplingMaskForward, py::arg("logits"),
        py::arg("topk_mask") = py::none(), py::arg("topp_mask") = py::none(),
        py::arg("no_repeat_mask") = py::none());
  m.def("sample_with_policies_forward", &SampleWithPoliciesForward, py::arg("logits"), py::arg("token_ids"),
        py::arg("do_sample"), py::arg("temperature") = 1.0, py::arg("top_k") = py::none(),
        py::arg("top_p") = py::none(), py::arg("no_repeat_ngram") = 0,
        py::arg("repetition_penalty") = 1.0, py::arg("presence_penalty") = 0.0,
        py::arg("frequency_penalty") = 0.0);
  m.def("presence_frequency_penalty_forward", &PresenceFrequencyPenaltyForward, py::arg("logits"),
        py::arg("counts"), py::arg("alpha_presence"), py::arg("alpha_frequency"));
  m.def("no_repeat_ngram_mask_forward", &NoRepeatNgramMaskForward, py::arg("token_ids"), py::arg("vocab_size"),
        py::arg("n"));
  m.def("sample_next_token_forward", &SampleNextTokenForward, py::arg("logits"), py::arg("do_sample"));
  m.def("beam_search_step_forward", &BeamSearchStepForward, py::arg("beams"), py::arg("logits"),
        py::arg("raw_scores"), py::arg("finished"), py::arg("lengths"), py::arg("beam_size"),
        py::arg("eos_id"), py::arg("pad_id"));
  m.def("repetition_penalty_forward", &RepetitionPenaltyForward, py::arg("logits"), py::arg("token_ids"),
        py::arg("penalty"));
  m.def("token_counts_forward", &TokenCountsForward, py::arg("token_ids"), py::arg("vocab_size"),
        py::arg("counts_dtype"));
  m.def("append_tokens_forward", &AppendTokensForward, py::arg("seq"), py::arg("next_id"),
        py::arg("attention_mask") = py::none());
  m.def("decode_positions_forward", &DecodePositionsForward, py::arg("batch_size"), py::arg("seq_len"),
        py::arg("reference"));
  m.def("activation_forward", &ApplyActivation, py::arg("x"), py::arg("activation") = "gelu");
  m.def("gated_activation_forward", &ApplyGatedActivation, py::arg("x"), py::arg("activation") = "swiglu");
  m.def("linear_forward", &LinearForward, py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
        py::arg("backend") = "auto");
  m.def("pack_linear_weight_forward", &PackLinearWeightForward, py::arg("weight"), py::arg("bias") = py::none());
  m.def("mlp_forward", &MlpForward, py::arg("x"), py::arg("w_in_weight"), py::arg("w_in_bias") = py::none(),
        py::arg("w_out_weight"), py::arg("w_out_bias") = py::none(), py::arg("activation") = "gelu",
        py::arg("gated") = false, py::arg("backend") = "auto");
  m.def("pack_qkv_weights_forward", &PackQkvWeightsForward, py::arg("q_weight"),
        py::arg("q_bias") = py::none(), py::arg("k_weight"), py::arg("k_bias") = py::none(),
        py::arg("v_weight"), py::arg("v_bias") = py::none());
  m.def("qkv_projection_forward", &QkvProjectionForward, py::arg("x"), py::arg("q_weight"),
        py::arg("q_bias") = py::none(), py::arg("k_weight"), py::arg("k_bias") = py::none(),
        py::arg("v_weight"), py::arg("v_bias") = py::none(), py::arg("backend") = "auto");
  m.def("qkv_packed_heads_projection_forward", &QkvPackedHeadsProjectionForward, py::arg("x"),
        py::arg("packed_weight"), py::arg("packed_bias") = py::none(), py::arg("q_size"), py::arg("k_size"),
        py::arg("v_size"), py::arg("q_heads"), py::arg("kv_heads"), py::arg("backend") = "auto");
  m.def("qkv_heads_projection_forward", &QkvHeadsProjectionForward, py::arg("x"), py::arg("q_weight"),
        py::arg("q_bias") = py::none(), py::arg("k_weight"), py::arg("k_bias") = py::none(),
        py::arg("v_weight"), py::arg("v_bias") = py::none(), py::arg("q_heads"), py::arg("kv_heads"),
        py::arg("backend") = "auto");
  m.def("split_heads_forward", &SplitHeadsForward, py::arg("x"), py::arg("num_heads"));
  m.def("merge_heads_forward", &MergeHeadsForward, py::arg("x"));
  m.def("head_output_projection_forward", &HeadOutputProjectionForward, py::arg("x"), py::arg("weight"),
        py::arg("bias") = py::none(), py::arg("backend") = "auto");
  m.def("embedding_forward", &EmbeddingForward, py::arg("weight"), py::arg("indices"),
        py::arg("padding_idx") = -1);
}
