#include <torch/extension.h>
#include <ATen/TensorIndexing.h>
#include <ATen/ops/_unique2.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/silu.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "descriptors/attention_desc.h"
#include "policy/attention_policy.h"
#include "reference/aten_reference.h"
#include "backend/bitnet/bitnet_formats.h"
#if MODEL_STACK_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "backend/cuda_device_arch.cuh"
#endif

namespace py = pybind11;
namespace torch_ext = torch;
using namespace pybind11::literals;

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
torch::Tensor CudaEmbeddingForwardUnchecked(
    const torch::Tensor& weight,
    const torch::Tensor& indices);
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CudaSpeculativeAcceptForward(
    const torch::Tensor& target_probs,
    const torch::Tensor& draft_probs,
    const torch::Tensor& draft_token_ids,
    const c10::optional<torch::Tensor>& bonus_probs,
    const c10::optional<torch::Tensor>& bonus_enabled,
    const std::string& method,
    double posterior_threshold,
    double posterior_alpha);
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
std::vector<torch::Tensor> CudaBeamSearchStepForward(
    const torch::Tensor& beams,
    const torch::Tensor& logits,
    const torch::Tensor& raw_scores,
    const torch::Tensor& finished,
    const torch::Tensor& lengths,
    int64_t beam_size,
    int64_t eos_id,
    int64_t pad_id);
bool HasCudaDecodePositionsKernel();
torch::Tensor CublasLtLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias);
bool HasCublasLtLinearBackend();
torch::Tensor CudaInt4LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias);
torch::Tensor CudaInt4LinearGradInputForward(
    const torch::Tensor& grad_out,
    const torch::Tensor& packed_weight,
    const torch::Tensor& inv_scale,
    int64_t in_features);
bool HasCudaInt4LinearKernel();
torch::Tensor CutlassInt4PackShuffledForward(const torch::Tensor& qweight);
torch::Tensor CutlassInt4Bf16LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight_rowmajor,
    const torch::Tensor& scale,
    const c10::optional<torch::Tensor>& bias,
    bool packed_weight_is_shuffled);
torch::Tensor CudaNf4LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias);
bool HasCudaNf4LinearKernel();
torch::Tensor CudaFp8LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    double weight_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);
bool HasCudaFp8LinearKernel();
torch::Tensor CudaInt8LinearForward(
    const torch::Tensor& qx,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);
torch::Tensor CublasLtInt8LinearAccumForward(
    const torch::Tensor& qx,
    const torch::Tensor& qweight);
bool HasCudaInt8LinearKernel();
std::vector<torch::Tensor> CudaInt8QuantizeActivationForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& provided_scale);
std::vector<torch::Tensor> CudaInt8QuantizeActivationTransposeForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& provided_scale);
std::vector<torch::Tensor> CudaInt8QuantizeActivationColumnwiseForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& provided_scale);
std::vector<torch::Tensor> CudaInt8QuantizeRelu2ActivationForward(
    const torch::Tensor& x,
    int64_t act_quant_bits);
std::vector<torch::Tensor> CudaInt8QuantizeLeakyReluHalf2ActivationForward(
    const torch::Tensor& x,
    int64_t act_quant_bits);
torch::Tensor CudaInt8LinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& provided_scale,
    const c10::optional<torch::ScalarType>& out_dtype);
torch::Tensor CudaInt8LinearFromFloatPreScaleForward(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& pre_scale,
    const c10::optional<torch::ScalarType>& out_dtype);
bool HasCudaInt8QuantFrontendKernel();
torch::Tensor CudaInt8AttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& q_scale,
    const torch::Tensor& k,
    const torch::Tensor& k_scale,
    const torch::Tensor& v,
    const torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    const c10::optional<torch::ScalarType>& out_dtype);
bool HasCudaInt8AttentionKernel();
torch::Tensor CudaInt8AttentionFromFloatForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    const c10::optional<torch::ScalarType>& out_dtype,
    const c10::optional<torch::Tensor>& q_provided_scale,
    const c10::optional<torch::Tensor>& k_provided_scale,
    const c10::optional<torch::Tensor>& v_provided_scale);
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
std::vector<torch::Tensor> CudaInt3PackLastDimForward(const torch::Tensor& x);
torch::Tensor CudaInt3DequantizeLastDimForward(
    const torch::Tensor& packed,
    const torch::Tensor& scale,
    int64_t original_last_dim,
    const c10::optional<torch::ScalarType>& out_dtype);
std::vector<torch::Tensor> CudaProjectedQkvRotaryPagedWriteForward(
    const torch::Tensor& projected,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads);
bool HasCudaKvCacheKernel();
torch::Tensor CudaAttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale);
bool HasCudaAttentionKernel();
torch::Tensor CudaPagedAttentionDecodeForward(
    const torch::Tensor& q,
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    const c10::optional<torch::Tensor>& attn_mask,
    const c10::optional<double>& scale,
    int64_t known_mask_seq = -1);
bool HasCudaPagedAttentionDecodeKernel();
namespace t10::bitnet {
torch::Tensor CudaBitNetLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);
torch::Tensor CudaBitNetLinearForwardComputePacked(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);
torch::Tensor CudaBitNetRmsNormLinearForwardDecodeRows(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& rms_weight,
    double eps,
    const torch::Tensor& layout_header,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);
std::vector<torch::Tensor> CudaBitNetAddRmsNormLinearForwardDecodeRows(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& rms_weight,
    double residual_scale,
    double eps,
    const torch::Tensor& layout_header,
    const torch::Tensor& compute_packed_words,
    const torch::Tensor& compute_row_scales,
    const torch::Tensor& decode_nz_masks,
    const torch::Tensor& decode_sign_masks,
    const torch::Tensor& decode_row_scales,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);
torch::Tensor CudaBitNetTransformInputForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale);
std::tuple<torch::Tensor, torch::Tensor> CudaBitNetQuantizeActivationInt8CodesForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale);
std::tuple<torch::Tensor, torch::Tensor> CudaBitNetQuantizeGatedActivationInt8CodesForward(
    const torch::Tensor& x,
    const std::string& activation,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale);
torch::Tensor CudaBitNetInt8LinearFromFloatRow1Forward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype);
torch::Tensor CudaBitNetLinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    const c10::optional<torch::Tensor>& act_scale,
    const c10::optional<torch::ScalarType>& out_dtype);
torch::Tensor BitNetInt8LinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    double act_quant_percentile,
    const c10::optional<torch::Tensor>& act_scale,
    const c10::optional<torch::ScalarType>& out_dtype);
std::vector<torch::Tensor> BitNetInt8FusedQkvPackedHeadsProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& packed_bias,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    double act_quant_percentile,
    const c10::optional<torch::Tensor>& act_scale,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads,
    const c10::optional<torch::ScalarType>& out_dtype);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CudaPackBitNetWeightForward(
    const torch::Tensor& weight);
std::tuple<torch::Tensor, torch::Tensor> CudaBitNetRuntimeRowQuantizeForward(
    const torch::Tensor& weight,
    double eps);
std::vector<torch::Tensor> CudaBitNetFusedQkvPackedHeadsProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& packed_bias,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads);
bool HasCudaBitNetLinearKernel();
bool HasCudaBitNetInputFrontendKernel();
bool HasCudaBitNetFusedQkvKernel();
}  // namespace t10::bitnet
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
bool HasCudaInt4LinearKernel() {
  return false;
}
bool HasCudaNf4LinearKernel() {
  return false;
}
bool HasCudaFp8LinearKernel() {
  return false;
}
bool HasCudaInt8LinearKernel() {
  return false;
}
bool HasCudaInt8QuantFrontendKernel() {
  return false;
}
bool HasCudaInt8AttentionKernel() {
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
bool HasCudaPagedAttentionDecodeKernel() {
  return false;
}
namespace t10::bitnet {
bool HasCudaBitNetInputFrontendKernel() {
  return false;
}
bool HasCudaBitNetLinearKernel() {
  return false;
}
bool HasCudaBitNetFusedQkvKernel() {
  return false;
}
}  // namespace t10::bitnet
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

bool CanUseCudaInt8AttentionPath(
    const torch::Tensor& q,
    const torch::Tensor& q_scale,
    const torch::Tensor& k,
    const torch::Tensor& k_scale,
    const torch::Tensor& v,
    const torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& attn_mask) {
#if MODEL_STACK_WITH_CUDA
  if (!(q.is_cuda() && q_scale.is_cuda() && k.is_cuda() && k_scale.is_cuda() && v.is_cuda() && v_scale.is_cuda() &&
        HasCudaInt8AttentionKernel())) {
    return false;
  }
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    const auto& mask = attn_mask.value();
    if (!mask.is_cuda() || mask.dim() != 4) {
      return false;
    }
    if (!(mask.scalar_type() == torch::kBool || mask.scalar_type() == torch::kFloat32 ||
          mask.scalar_type() == torch::kFloat16 || mask.scalar_type() == torch::kBFloat16)) {
      return false;
    }
    if (mask.size(0) != q.size(0) || mask.size(2) != q.size(2) || mask.size(3) != k.size(2)) {
      return false;
    }
    if (!(mask.size(1) == 1 || mask.size(1) == q.size(1))) {
      return false;
    }
  }
  return true;
#else
  (void)q;
  (void)q_scale;
  (void)k;
  (void)k_scale;
  (void)v;
  (void)v_scale;
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
      {"bitnet_transform_input", true},
      {"bitnet_linear", true},
      {"bitnet_linear_compute_packed", true},
      {"bitnet_linear_from_float", true},
      {"bitnet_int8_linear_from_float", true},
      {"bitnet_int8_fused_qkv_packed_heads_projection", true},
      {"int4_linear", true},
      {"int4_linear_grad_input", true},
      {"nf4_linear", true},
      {"fp8_linear", true},
      {"int8_quantize_activation", true},
      {"int8_quantize_activation_transpose", true},
      {"int8_quantize_activation_columnwise", true},
      {"int8_quantize_relu2_activation", true},
      {"int8_quantize_leaky_relu_half2_activation", true},
      {"int8_linear", true},
      {"int8_linear_from_float", true},
      {"int8_linear_grad_weight_from_float", true},
      {"int8_attention", true},
      {"int8_attention_from_float", true},
      {"pack_bitnet_weight", true},
      {"bitnet_runtime_row_quantize", true},
      {"pack_linear_weight", true},
      {"mlp", true},
      {"qkv_projection", true},
      {"pack_qkv_weights", true},
      {"qkv_packed_heads_projection", true},
      {"bitnet_qkv_packed_heads_projection", true},
      {"bitnet_fused_qkv_packed_heads_projection", true},
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
      {"int3_kv_pack", true},
      {"int3_kv_dequantize", true},
      {"paged_attention_decode", true},
      {"attention_decode", true},
      {"attention_prefill", true},
      {"sampling", true},
      {"beam_search_step", true},
      {"incremental_beam_search", true},
  };
}

std::vector<std::string> SupportedLinearBackends() {
#if MODEL_STACK_WITH_CUDA
  if (HasCublasLtLinearBackend()) {
    return {"aten", "bitnet", "cublaslt"};
  }
#endif
  return {"aten", "bitnet"};
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

bool EnvFlagEnabled(const char* name) {
  const char* env_value = std::getenv(name);
  if (env_value == nullptr) {
    return false;
  }
  const auto normalized = NormalizeBackendName(env_value);
  return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

int64_t EnvInt64OrDefault(const char* name, int64_t default_value) {
  const char* env_value = std::getenv(name);
  if (env_value == nullptr || *env_value == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const auto parsed = std::strtoll(env_value, &end, 10);
  if (end == env_value) {
    return default_value;
  }
  return static_cast<int64_t>(parsed);
}

std::string HopperDenseBitNetDecodeCacheMode() {
  if (EnvFlagEnabled("MODEL_STACK_DISABLE_HOPPER_DENSE_DECODE_CACHE")) {
    return "off";
  }
  const char* env_value = std::getenv("MODEL_STACK_HOPPER_DENSE_DECODE_CACHE_MODE");
  if (env_value == nullptr) {
    return "full";
  }
  const auto normalized = NormalizeBackendName(env_value);
  if (normalized == "full") {
    return "full";
  }
  if (normalized.empty() || normalized == "qkv" || normalized == "qkv_only") {
    return "qkv_only";
  }
  if (normalized == "off" || normalized == "disabled" || normalized == "none" ||
      normalized == "0" || normalized == "false" || normalized == "no") {
    return "off";
  }
  return "full";
}

bool HopperDenseBitNetDecodeQkvCacheEnabled() {
  const auto mode = HopperDenseBitNetDecodeCacheMode();
  return mode == "full" || mode == "qkv_only";
}

bool HopperDenseBitNetDecodeLinearCacheEnabled() {
  return HopperDenseBitNetDecodeCacheMode() == "full";
}

int64_t HopperDenseBitNetDecodeFusedAppendMaxCacheLength() {
  return std::max<int64_t>(
      0,
      EnvInt64OrDefault("MODEL_STACK_HOPPER_DENSE_DECODE_FUSED_APPEND_MAX_CACHE_LENGTH", 4096));
}

bool BitNetProjectedQkvDecodeFusedAppendEnabled() {
  return !EnvFlagEnabled("MODEL_STACK_DISABLE_BITNET_PROJECTED_QKV_DECODE_FUSED_APPEND");
}

bool BitNetDecodeFusedNormRowsEnabled(const torch::Tensor& x) {
#if MODEL_STACK_WITH_CUDA
  if (EnvFlagEnabled("MODEL_STACK_DISABLE_BITNET_DECODE_FUSED_NORM_ROWS")) {
    return false;
  }
  if (EnvFlagEnabled("MODEL_STACK_ENABLE_BITNET_DECODE_FUSED_NORM_ROWS")) {
    return true;
  }
  return x.is_cuda() && t10::cuda::DeviceIsSm90OrLater(x);
#else
  (void)x;
  return false;
#endif
}

int64_t PagedDecodeSdpaMaxLength() {
  if (EnvFlagEnabled("MODEL_STACK_DISABLE_PAGED_DECODE_SDPA_BRIDGE")) {
    return 0;
  }
  return std::max<int64_t>(
      0,
      EnvInt64OrDefault("MODEL_STACK_PAGED_DECODE_SDPA_MAX_LENGTH", 8192));
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
      candidate == "aten" || candidate == "bitnet" || (candidate == "cublaslt" && HasCublasLtLinearBackend()),
      "Unsupported linear backend request: ",
      candidate,
      " (supported backends: aten, bitnet; planned backends: cublaslt)");
  return candidate;
}

bool PreferAtenLinearBackendForAuto(const torch::Tensor& reference) {
#if MODEL_STACK_WITH_CUDA
  if (!reference.is_cuda() || !HasCublasLtLinearBackend()) {
    return false;
  }
  if (EnvFlagEnabled("MODEL_STACK_DISABLE_SM8X_ATEN_LINEAR_AUTO")) {
    return false;
  }
  int major = 0;
  return t10::cuda::DeviceComputeCapability(reference, &major, nullptr) && major == 8;
#else
  (void)reference;
  return false;
#endif
}

std::string ResolveLinearBackendForTensor(const std::string& requested, const torch::Tensor& reference) {
  const auto normalized = NormalizeBackendName(requested);
  const bool auto_backend = normalized.empty() || normalized == "auto";
  auto resolved = ResolveLinearBackend(requested);
  if (auto_backend && resolved == "cublaslt" && PreferAtenLinearBackendForAuto(reference)) {
    return "aten";
  }
  return resolved;
}

torch::Tensor ReferenceInt4LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias) {
  const auto in_features = x.size(-1);
  auto packed_cpu = packed_weight.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8)).contiguous();
  auto scale_cpu = inv_scale.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
  auto weight_cpu = torch::empty(
      {packed_cpu.size(0), in_features},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));

  auto packed_acc = packed_cpu.accessor<uint8_t, 2>();
  auto scale_acc = scale_cpu.accessor<float, 1>();
  auto weight_acc = weight_cpu.accessor<float, 2>();
  for (int64_t out_idx = 0; out_idx < packed_cpu.size(0); ++out_idx) {
    for (int64_t in_idx = 0; in_idx < in_features; ++in_idx) {
      const auto packed_value = packed_acc[out_idx][in_idx / 2];
      const auto nibble = (in_idx % 2) == 0 ? (packed_value & 0x0F) : ((packed_value >> 4) & 0x0F);
      const auto q = static_cast<int>(nibble) - 8;
      weight_acc[out_idx][in_idx] = static_cast<float>(q) * scale_acc[out_idx];
    }
  }

  auto weight = weight_cpu.to(x.device(), x.scalar_type());
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    bias_cast = bias.value().to(x.device(), x.scalar_type());
  }
  return ReferenceLinearForward(x, weight, bias_cast);
}

std::pair<torch::Tensor, torch::Tensor> ReferenceQuantizeActivationInt8Rowwise(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& provided_scale,
    int act_quant_bits = 8) {
  TORCH_CHECK(act_quant_bits >= 2 && act_quant_bits <= 8,
              "ReferenceQuantizeActivationInt8Rowwise: act_quant_bits must be in [2, 8]");
  const float qmax = static_cast<float>((1 << (act_quant_bits - 1)) - 1);
  const auto cols = x.size(-1);
  const auto rows = x.numel() / cols;
  auto x_float = x.to(torch::kFloat32);
  auto x_2d = x_float.reshape({rows, cols});
  torch::Tensor row_scale;
  if (provided_scale.has_value() && provided_scale.value().defined()) {
    row_scale = provided_scale.value().to(x.device(), torch::kFloat32).reshape({-1});
    TORCH_CHECK(
        row_scale.numel() == 1 || row_scale.numel() == rows,
        "ReferenceQuantizeActivationInt8Rowwise: provided_scale must have 1 or rows elements");
    if (row_scale.numel() == 1) {
      row_scale = row_scale.expand({rows}).clone();
    }
    row_scale = row_scale.clamp_min(1e-8).contiguous();
  } else {
    row_scale = (x_2d.abs().amax(-1).clamp_min(1e-8) / qmax).to(torch::kFloat32).contiguous();
  }
  auto qx = torch::round(x_2d / row_scale.unsqueeze(1)).clamp(-qmax, qmax).to(torch::kInt8);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  return {qx.view(out_sizes), row_scale};
}

torch::Tensor ReferenceNf4LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias) {
  constexpr float kNf4Codebook[16] = {
      -1.0f, -0.6961928f, -0.52507305f, -0.3949175f,
      -0.28444138f, -0.18477343f, -0.09105004f, 0.0f,
      0.0795803f, 0.1609302f, 0.2461123f, 0.33791524f,
      0.44070983f, 0.5626170f, 0.72295684f, 1.0f};
  const auto in_features = x.size(-1);
  auto packed_cpu = packed_weight.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8)).contiguous();
  auto scale_cpu = weight_scale.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
  auto weight_cpu = torch::empty(
      {packed_cpu.size(0), in_features},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));

  auto packed_acc = packed_cpu.accessor<uint8_t, 2>();
  auto scale_acc = scale_cpu.accessor<float, 1>();
  auto weight_acc = weight_cpu.accessor<float, 2>();
  for (int64_t out_idx = 0; out_idx < packed_cpu.size(0); ++out_idx) {
    for (int64_t in_idx = 0; in_idx < in_features; ++in_idx) {
      const auto packed_value = packed_acc[out_idx][in_idx / 2];
      const auto nibble = (in_idx % 2) == 0 ? (packed_value & 0x0F) : ((packed_value >> 4) & 0x0F);
      weight_acc[out_idx][in_idx] = kNf4Codebook[nibble] * scale_acc[out_idx];
    }
  }

  auto weight = weight_cpu.to(x.device(), x.scalar_type());
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    bias_cast = bias.value().to(x.device(), x.scalar_type());
  }
  return ReferenceLinearForward(x, weight, bias_cast);
}

torch::Tensor ReferenceFp8LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    double weight_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  const auto output_dtype = out_dtype.value_or(x.scalar_type());
  auto x_cast = x.to(x.device(), output_dtype);
  auto weight = (weight_fp8.to(x.device(), output_dtype) * static_cast<float>(weight_scale)).contiguous();
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    bias_cast = bias.value().to(x.device(), output_dtype);
  }
  return ReferenceLinearForward(x_cast, weight, bias_cast);
}

torch::Tensor ReferenceInt8LinearForward(
    const torch::Tensor& qx,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(qx.scalar_type() == torch::kInt8, "int8_linear_forward: qx must use int8 storage");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt8, "int8_linear_forward: qweight must use int8 storage");
  TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32, "int8_linear_forward: x_scale must use float32 storage");
  TORCH_CHECK(inv_scale.scalar_type() == torch::kFloat32, "int8_linear_forward: inv_scale must use float32 storage");
  const auto in_features = qx.size(-1);
  const auto rows = qx.numel() / in_features;
  auto qx_2d = qx.reshape({rows, in_features}).to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt8)).contiguous();
  auto x_scale_cpu = x_scale.reshape({rows}).to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
  auto qweight_cpu = qweight.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt8)).contiguous();
  auto inv_scale_cpu = inv_scale.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();

  auto x_float_cpu = qx_2d.to(torch::kFloat32) * x_scale_cpu.unsqueeze(1);
  auto weight_float_cpu = qweight_cpu.to(torch::kFloat32) * inv_scale_cpu.unsqueeze(1);
  std::vector<int64_t> out_sizes(qx.sizes().begin(), qx.sizes().end());
  auto x_float = x_float_cpu.to(qx.device(), torch::kFloat32).view(out_sizes);
  auto weight_float = weight_float_cpu.to(qweight.device(), torch::kFloat32);

  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    bias_cast = bias.value().to(qx.device(), torch::kFloat32);
  }
  auto out = ReferenceLinearForward(x_float, weight_float, bias_cast);
  if (out_dtype.has_value()) {
    out = out.to(out_dtype.value());
  }
  return out;
}

torch::Tensor ReferenceInt8LinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& provided_scale,
    const c10::optional<torch::ScalarType>& out_dtype) {
  auto quantized = ReferenceQuantizeActivationInt8Rowwise(x, provided_scale);
  return ReferenceInt8LinearForward(
      quantized.first,
      quantized.second,
      qweight,
      inv_scale,
      bias,
      out_dtype);
}

torch::Tensor ReferenceInt8AttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& q_scale,
    const torch::Tensor& k,
    const torch::Tensor& k_scale,
    const torch::Tensor& v,
    const torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(q.scalar_type() == torch::kInt8, "int8_attention_forward: q must use int8 storage");
  TORCH_CHECK(k.scalar_type() == torch::kInt8, "int8_attention_forward: k must use int8 storage");
  TORCH_CHECK(v.scalar_type() == torch::kInt8, "int8_attention_forward: v must use int8 storage");
  TORCH_CHECK(q_scale.scalar_type() == torch::kFloat32, "int8_attention_forward: q_scale must use float32 storage");
  TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32, "int8_attention_forward: k_scale must use float32 storage");
  TORCH_CHECK(v_scale.scalar_type() == torch::kFloat32, "int8_attention_forward: v_scale must use float32 storage");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "int8_attention_forward: q, k, and v must be rank-4");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "int8_attention_forward: batch size mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "int8_attention_forward: kv head mismatch");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "int8_attention_forward: q heads must be a multiple of kv heads");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "int8_attention_forward: head_dim mismatch");
  const auto q_rows = q.size(0) * q.size(1) * q.size(2);
  const auto kv_rows = k.size(0) * k.size(1) * k.size(2);
  TORCH_CHECK(q_scale.numel() == q_rows, "int8_attention_forward: q_scale size mismatch");
  TORCH_CHECK(k_scale.numel() == kv_rows, "int8_attention_forward: k_scale size mismatch");
  TORCH_CHECK(v_scale.numel() == kv_rows, "int8_attention_forward: v_scale size mismatch");

  auto q_2d = q.reshape({q_rows, q.size(3)}).to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt8)).contiguous();
  auto k_2d = k.reshape({kv_rows, k.size(3)}).to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt8)).contiguous();
  auto v_2d = v.reshape({kv_rows, v.size(3)}).to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt8)).contiguous();
  auto q_scale_cpu = q_scale.reshape({q_rows, 1}).to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
  auto k_scale_cpu = k_scale.reshape({kv_rows, 1}).to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
  auto v_scale_cpu = v_scale.reshape({kv_rows, 1}).to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();

  auto q_float = (q_2d.to(torch::kFloat32) * q_scale_cpu).view({q.size(0), q.size(1), q.size(2), q.size(3)}).to(q.device(), torch::kFloat32);
  auto k_float = (k_2d.to(torch::kFloat32) * k_scale_cpu).view({k.size(0), k.size(1), k.size(2), k.size(3)}).to(k.device(), torch::kFloat32);
  auto v_float = (v_2d.to(torch::kFloat32) * v_scale_cpu).view({v.size(0), v.size(1), v.size(2), v.size(3)}).to(v.device(), torch::kFloat32);

  c10::optional<torch::Tensor> mask_cast = c10::nullopt;
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    mask_cast = attn_mask.value().to(q.device());
  }
  auto out = ReferenceAttentionForward(q_float, k_float, v_float, mask_cast, is_causal, scale);
  if (out_dtype.has_value()) {
    out = out.to(out_dtype.value());
  }
  return out;
}

torch::Tensor ReferenceInt8AttentionFromFloatForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    const c10::optional<torch::ScalarType>& out_dtype,
    const c10::optional<torch::Tensor>& q_provided_scale,
    const c10::optional<torch::Tensor>& k_provided_scale,
    const c10::optional<torch::Tensor>& v_provided_scale) {
  auto qq = ReferenceQuantizeActivationInt8Rowwise(q, q_provided_scale);
  auto kk = ReferenceQuantizeActivationInt8Rowwise(k, k_provided_scale);
  auto vv = ReferenceQuantizeActivationInt8Rowwise(v, v_provided_scale);
  return ReferenceInt8AttentionForward(
      qq.first,
      qq.second,
      kk.first,
      kk.second,
      vv.first,
      vv.second,
      attn_mask,
      is_causal,
      scale,
      out_dtype);
}

torch::Tensor ReferenceBitNetLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias) {
  const auto layout = t10::bitnet::ParseLayoutHeader(layout_header);
  t10::bitnet::ValidateSegmentOffsets(segment_offsets, layout);
  TORCH_CHECK(packed_weight.dim() == 2, "bitnet_linear_forward: packed_weight must be rank-2");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
              "bitnet_linear_forward: packed_weight must use uint8 storage");
  TORCH_CHECK(scale_values.dim() == 1, "bitnet_linear_forward: scale_values must be rank-1");
  TORCH_CHECK(x.size(-1) == layout.logical_in_features,
              "bitnet_linear_forward: input feature size mismatch");
  TORCH_CHECK(packed_weight.size(0) == layout.padded_out_features,
              "bitnet_linear_forward: packed_weight row count mismatch");
  TORCH_CHECK(packed_weight.size(1) == (layout.padded_in_features + 3) / 4,
              "bitnet_linear_forward: packed_weight column count mismatch");
  if (bias.has_value() && bias.value().defined()) {
    const auto& b = bias.value();
    TORCH_CHECK(b.dim() == 1, "bitnet_linear_forward: bias must be rank-1");
    TORCH_CHECK(b.size(0) == layout.logical_out_features,
                "bitnet_linear_forward: bias size mismatch");
  }

  auto packed_cpu = packed_weight.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8)).contiguous();
  auto weight_cpu = torch::empty(
      {layout.logical_out_features, layout.logical_in_features},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  auto packed_acc = packed_cpu.accessor<uint8_t, 2>();
  auto weight_acc = weight_cpu.accessor<float, 2>();
  for (int64_t out_idx = 0; out_idx < layout.logical_out_features; ++out_idx) {
    const float row_scale = t10::bitnet::ResolveRowScale(out_idx, layout, scale_values, segment_offsets);
    for (int64_t in_idx = 0; in_idx < layout.logical_in_features; ++in_idx) {
      const auto packed_value = packed_acc[out_idx][in_idx / 4];
      const auto code = static_cast<int>((packed_value >> ((in_idx % 4) * 2)) & 0x03);
      const auto q = code == 0 ? -1 : (code == 2 ? 1 : 0);
      weight_acc[out_idx][in_idx] = static_cast<float>(q) * row_scale;
    }
  }

  auto weight = weight_cpu.to(x.device(), x.scalar_type());
  c10::optional<torch::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value() && bias.value().defined()) {
    bias_cast = bias.value().to(x.device(), x.scalar_type());
  }
  return ReferenceLinearForward(x, weight, bias_cast);
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
  if (act == "relu2" || act == "squared_relu" || act == "squared-relu") {
    auto y = at::relu(x);
    return y * y;
  }
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
  if (act == "leaky_relu_0p5_squared" || act == "leaky-relu-0p5-squared" ||
      act == "leaky_relu2" || act == "leaky-relu2" ||
      act == "leaky_relu_0.5_squared" || act == "leaky-relu-0.5-squared") {
    auto y = at::leaky_relu(x, 0.5);
    return y * y;
  }
  return at::gelu(x, "none");
}

torch::Tensor ApplyGatedActivation(const torch::Tensor& x, const std::string& activation) {
  TORCH_CHECK(x.size(-1) % 2 == 0, "apply_gated_activation: hidden width must be even");
  const auto act = NormalizeBackendName(activation);
  if (act == "relu2" || act == "squared_relu" || act == "squared-relu") {
    const auto split = x.size(-1) / 2;
    auto a = x.slice(-1, 0, split);
    auto b = x.slice(-1, split, x.size(-1));
    return ApplyActivation(a, act) * b;
  }
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

c10::optional<torch::Tensor> OptionalTensorFromPyObject(const py::object& obj) {
  if (obj.is_none()) {
    return c10::nullopt;
  }
  return py::cast<torch::Tensor>(obj);
}

py::tuple AppendTokensForward(
    const torch::Tensor& seq,
    const torch::Tensor& next_id,
    const c10::optional<torch::Tensor>& attention_mask,
    const c10::optional<torch::Tensor>& row_ids);

py::tuple DecodePositionsForward(
    int64_t batch_size,
    int64_t seq_len,
    const torch::Tensor& reference);

torch::Tensor EmbeddingForward(
    const torch::Tensor& weight,
    const torch::Tensor& indices,
    int64_t padding_idx);

torch::Tensor RmsNormForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    double eps);

torch::Tensor LayerNormForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps);

std::vector<torch::Tensor> AddRmsNormForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& weight,
    double residual_scale,
    double eps);

std::vector<torch::Tensor> AddLayerNormForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double residual_scale,
    double eps);

torch::Tensor ResidualAddForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    double residual_scale);

std::vector<torch::Tensor> ApplyRotaryForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin);

torch::Tensor ResolvePositionIdsForward(
    int64_t batch_size,
    int64_t seq_len,
    const torch::Tensor& reference,
    const c10::optional<torch::Tensor>& attention_mask,
    const c10::optional<torch::Tensor>& cache_position,
    int64_t past_length);

torch::Tensor CreateCausalMaskForward(
    const torch::Tensor& reference,
    const c10::optional<torch::Tensor>& attention_mask,
    const c10::optional<torch::Tensor>& cache_position,
    const c10::optional<torch::Tensor>& position_ids);

int64_t ResolveRopeSequenceLength(
    int64_t fallback_seq_len,
    const c10::optional<torch::Tensor>& position_ids,
    const c10::optional<int64_t>& known_single_position = c10::nullopt) {
  int64_t needed = fallback_seq_len;
  if (known_single_position.has_value()) {
    return std::max<int64_t>(needed, known_single_position.value() + 1);
  }
  if (!position_ids.has_value() || !position_ids.value().defined()) {
    return needed;
  }
  try {
    auto gather_pos = position_ids.value().to(torch::kLong);
    if (gather_pos.dim() == 2) {
      gather_pos = gather_pos[0];
    }
    if (gather_pos.numel() > 0) {
      needed = std::max<int64_t>(needed, gather_pos.max().item<int64_t>() + 1);
    }
  } catch (...) {
  }
  return needed;
}

std::string LowerAscii(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

double ResolveRopeNtkScaling(
    int64_t seq_len,
    double base_theta,
    double factor,
    const c10::optional<int64_t>& original_max_position_embeddings,
    int64_t head_dim) {
  if (!(factor > 1.0)) {
    return base_theta;
  }
  const auto original = original_max_position_embeddings.has_value()
      ? std::max<int64_t>(original_max_position_embeddings.value(), 1)
      : std::max<int64_t>(seq_len, 1);
  if (seq_len <= original) {
    return base_theta;
  }
  const auto ratio = std::max(
      (factor * static_cast<double>(seq_len) / static_cast<double>(original)) - (factor - 1.0),
      1.0);
  double exponent = 1.0;
  if (head_dim > 2) {
    exponent = static_cast<double>(head_dim) / static_cast<double>(head_dim - 2);
  }
  return base_theta * std::pow(ratio, exponent);
}

std::pair<double, double> ResolveNativeRopeParameters(
    int64_t seq_len,
    int64_t head_dim,
    double base_theta,
    double attention_scaling,
    const std::string& scaling_type,
    bool has_scaling_factor,
    double scaling_factor,
    const c10::optional<int64_t>& original_max_position_embeddings,
    const c10::optional<double>& low_freq_factor,
    const c10::optional<double>& high_freq_factor) {
  (void)low_freq_factor;
  (void)high_freq_factor;
  double base = base_theta;
  double attn = attention_scaling;
  const auto st = LowerAscii(scaling_type);
  if (st == "linear" && has_scaling_factor) {
    base *= scaling_factor;
  } else if ((st == "dynamic" || st == "ntk") && has_scaling_factor) {
    base = ResolveRopeNtkScaling(
        seq_len,
        base,
        scaling_factor,
        original_max_position_embeddings,
        head_dim);
  } else if (st == "yarn" && has_scaling_factor) {
    double scale = scaling_factor;
    if (original_max_position_embeddings.has_value() && original_max_position_embeddings.value() > 0) {
      scale = std::max(
          scale,
          static_cast<double>(seq_len) / static_cast<double>(original_max_position_embeddings.value()));
    }
    scale = std::max(scale, 1.0);
    if (scale > 1.0) {
      attn *= 0.1 * std::log(scale) + 1.0;
    }
  }
  return {base, attn};
}

std::vector<torch::Tensor> ResolveRotaryEmbeddingForward(
    const torch::Tensor& reference,
    int64_t head_dim,
    double base_theta,
    double attention_scaling,
    const c10::optional<torch::Tensor>& position_ids);

torch::Tensor NativeAttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale);

torch::Tensor MlpForward(
    const torch::Tensor& x,
    const torch::Tensor& w_in_weight,
    const c10::optional<torch::Tensor>& w_in_bias,
    const torch::Tensor& w_out_weight,
    const c10::optional<torch::Tensor>& w_out_bias,
    const std::string& activation,
    bool gated,
    const std::string& backend);

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
    const std::string& backend);

torch::Tensor HeadOutputProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const std::string& backend);

std::string DetectNativeExecutorKind(const py::object& model);
bool ModelUsesAttentionBiases(const py::object& model);
c10::optional<torch::Tensor> TryNativeCausalModelForward(
    const py::object& model,
    const torch::Tensor& input_ids,
    const py::object& attention_mask,
    const py::object& cache,
    const py::object& position_ids,
    const py::object& cache_position,
    bool skip_model_support_check = false,
    bool model_uses_attention_biases = false);
struct PreparedNativeCausalModel;
std::shared_ptr<PreparedNativeCausalModel> TryPrepareNativeCausalModel(const py::object& model);
c10::optional<torch::Tensor> TryPreparedNativeCausalModelForward(
    const std::shared_ptr<PreparedNativeCausalModel>& prepared,
    const torch::Tensor& input_ids,
    const py::object& attention_mask,
    const py::object& cache,
    const py::object& position_ids_obj,
    const py::object& cache_position_obj,
    bool model_uses_attention_biases,
    int64_t known_decode_position = -1);
std::string ResolveAttentionMaskMode(
    const py::object& attention_mask,
    int64_t batch_size,
    int64_t seq_len);
torch::Tensor AppendExplicitDecodeAttentionMaskForward(
    const torch::Tensor& attention_mask,
    const c10::optional<torch::Tensor>& row_ids);

class NativeModelSession {
 public:
  NativeModelSession(
      py::object model,
      const torch::Tensor& seq,
      py::object attention_mask = py::none(),
      py::object cache = py::none(),
      bool trace = false)
      : model_(std::move(model)),
        seq_(seq),
        attention_mask_(std::move(attention_mask)),
        cache_(std::move(cache)),
        trace_(trace),
        native_executor_kind_(DetectNativeExecutorKind(model_)),
        model_uses_attention_biases_(ModelUsesAttentionBiases(model_)),
        attention_mask_mode_(ResolveAttentionMaskMode(attention_mask_, BatchSize(), SeqLen())),
        prepared_native_causal_model_(
            native_executor_kind_ == "causal_lm" ? TryPrepareNativeCausalModel(model_) : nullptr),
        decode_token_buffer_() {
    UpdateDecodeStepBuffers();
  }

  int64_t BatchSize() const { return seq_.defined() ? seq_.size(0) : 0; }
  int64_t SeqLen() const { return seq_.defined() ? seq_.size(1) : 0; }

  torch::Tensor GetSeq() const { return seq_; }
  void SetSeq(const torch::Tensor& seq) {
    seq_ = seq;
    UpdateDecodeStepBuffers();
    RefreshDecodeGraphState();
  }

  py::object GetAttentionMask() const { return attention_mask_; }
  void SetAttentionMask(py::object attention_mask) {
    attention_mask_ = std::move(attention_mask);
    attention_mask_mode_ = ResolveAttentionMaskMode(attention_mask_, BatchSize(), SeqLen());
    RefreshDecodeGraphState();
  }

  py::object GetCache() const { return cache_; }
  void SetCache(py::object cache) {
    cache_ = std::move(cache);
    UpdateDecodeStepBuffers();
    RefreshDecodeGraphState();
  }

  std::string NativeExecutorKind() const { return native_executor_kind_; }

  bool DecodeGraphEligible() const {
    if (cache_.is_none()) {
      return false;
    }
    if (native_executor_kind_ != "causal_lm" || prepared_native_causal_model_ == nullptr) {
      return false;
    }
    if (attention_mask_mode_ != "none" || model_uses_attention_biases_) {
      return false;
    }
    if (!seq_.defined() || !seq_.is_cuda() || seq_.dim() < 2 || SeqLen() <= 0) {
      return false;
    }
    if (!decode_token_buffer_.defined() || !decode_token_buffer_.is_cuda() || decode_token_buffer_.dim() != 2) {
      return false;
    }
    return decode_token_buffer_.size(0) == BatchSize() && decode_token_buffer_.size(1) == 1;
  }

  void SetDecodeGraphEnabled(bool enabled = true) {
    if (!enabled) {
      decode_graph_enabled_ = false;
      decode_position_ids_buffer_ = torch::Tensor();
      decode_cache_position_buffer_ = torch::Tensor();
      return;
    }
    decode_graph_enabled_ = false;
    UpdateDecodeStepBuffers();
    if (!DecodeGraphEligible()) {
      decode_position_ids_buffer_ = torch::Tensor();
      decode_cache_position_buffer_ = torch::Tensor();
      return;
    }
    decode_graph_enabled_ = true;
    UpdateDecodeStepBuffers();
  }

  void DisableCache() {
    cache_ = py::none();
    decode_token_buffer_ = torch::Tensor();
    decode_graph_enabled_ = false;
    decode_position_ids_buffer_ = torch::Tensor();
    decode_cache_position_buffer_ = torch::Tensor();
  }

  py::object PrefillNextLogits() const {
    if (cache_.is_none()) {
      return py::none();
    }
    auto logits = CallModel(seq_.contiguous(), attention_mask_, cache_, py::none(), py::none());
    if (logits.dim() == 2) {
      return py::cast(logits);
    }
    return py::cast(logits.index({at::indexing::Slice(), -1, at::indexing::Slice()}));
  }

  torch::Tensor FullNextLogits() const {
    auto logits = CallModel(seq_.contiguous(), attention_mask_, py::none(), py::none(), py::none());
    return logits.index({at::indexing::Slice(), -1, at::indexing::Slice()});
  }

  void Append(const torch::Tensor& next_id) {
    if (attention_mask_mode_ == "explicit" && !attention_mask_.is_none()) {
      auto next = AppendTokensForward(seq_, next_id, c10::nullopt, c10::nullopt);
      seq_ = py::cast<torch::Tensor>(next[0]);
      attention_mask_ = py::cast(AppendExplicitDecodeAttentionMaskForward(py::cast<torch::Tensor>(attention_mask_), c10::nullopt));
      attention_mask_mode_ = "explicit";
      UpdateDecodeStepBuffers(next_id);
      RefreshDecodeGraphState(next_id);
      return;
    }
    auto next = AppendTokensForward(seq_, next_id, OptionalTensorFromPyObject(attention_mask_), c10::nullopt);
    seq_ = py::cast<torch::Tensor>(next[0]);
    attention_mask_ = next[1].is_none() ? py::none() : next[1];
    attention_mask_mode_ = attention_mask_.is_none() ? std::string("none") : std::string("token");
    UpdateDecodeStepBuffers(next_id);
    RefreshDecodeGraphState(next_id);
  }

  py::tuple DecodePositions() const {
    if (cache_.is_none()) {
      return py::make_tuple(py::none(), py::none());
    }
    return DecodePositionsForward(BatchSize(), SeqLen(), seq_);
  }

  py::object DecodeNextLogits() const {
    if (cache_.is_none()) {
      return py::none();
    }
    auto decode_tokens = decode_token_buffer_.defined()
        ? decode_token_buffer_
        : seq_.slice(1, std::max<int64_t>(SeqLen() - 1, 0), SeqLen()).contiguous();
    torch::Tensor logits;
    if (native_executor_kind_ == "causal_lm" &&
        prepared_native_causal_model_ != nullptr &&
        attention_mask_mode_ == "none" &&
        !model_uses_attention_biases_) {
      if (decode_graph_enabled_ &&
          decode_position_ids_buffer_.defined() &&
          decode_cache_position_buffer_.defined()) {
        logits = CallModel(
            decode_tokens,
            attention_mask_,
            cache_,
            py::cast(decode_position_ids_buffer_),
            py::cast(decode_cache_position_buffer_));
      } else {
        logits = CallModel(
            decode_tokens,
            attention_mask_,
            cache_,
            py::none(),
            py::none(),
            SeqLen() - 1);
      }
    } else {
      auto pos = DecodePositions();
      logits = CallModel(
          decode_tokens,
          attention_mask_,
          cache_,
          pos[0],
          pos[1]);
    }
    if (logits.dim() == 2) {
      return py::cast(logits);
    }
    return py::cast(logits.index({at::indexing::Slice(), -1, at::indexing::Slice()}));
  }

  void ReorderCache(const torch::Tensor& row_ids, py::object source_cache = py::none()) {
    py::object cache_in = source_cache.is_none() ? cache_ : source_cache;
    if (cache_in.is_none()) {
      cache_ = py::none();
      RefreshDecodeGraphState();
      return;
    }
    cache_ = py::module_::import("runtime.kv_cache").attr("reorder_kv_cache_rows_")(cache_in, row_ids);
    RefreshDecodeGraphState();
  }

  void AppendAttentionMask(py::object row_ids = py::none(), py::object source_attention_mask = py::none()) {
    py::object mask_in = source_attention_mask.is_none() ? attention_mask_ : source_attention_mask;
    if (mask_in.is_none()) {
      attention_mask_ = py::none();
      attention_mask_mode_ = "none";
      RefreshDecodeGraphState();
      return;
    }
    auto mask = py::cast<torch::Tensor>(mask_in);
    auto row_ids_tensor = OptionalTensorFromPyObject(row_ids);
    const bool token_mask = attention_mask_mode_ == "token" || (attention_mask_mode_ != "explicit" && mask.dim() == 2 && mask.size(0) != mask.size(1));
    if (!token_mask) {
      attention_mask_ = py::cast(AppendExplicitDecodeAttentionMaskForward(mask, row_ids_tensor));
      attention_mask_mode_ = "explicit";
      RefreshDecodeGraphState();
      return;
    }
    const auto rows = row_ids_tensor.has_value() ? row_ids_tensor->numel() : mask.size(0);
    auto ones = torch::ones(
        {rows, 1},
        torch::TensorOptions().dtype(mask.scalar_type()).device(mask.device()));
    auto next = AppendTokensForward(mask, ones, c10::nullopt, row_ids_tensor);
    attention_mask_ = next[0];
    attention_mask_mode_ = attention_mask_.is_none() ? std::string("none") : std::string("token");
    RefreshDecodeGraphState();
  }

  void EvictIfNeeded(int64_t max_tokens, const std::string& policy = "sliding-window") {
    if (cache_.is_none() || max_tokens < 0) {
      return;
    }
    py::module_::import("runtime.cache").attr("evict_kv_cache")(cache_, max_tokens, policy);
  }

  py::object AdvanceBeamDecode(
      const torch::Tensor& next_beams,
      const torch::Tensor& cache_row_ids,
      py::object mask_row_ids = py::none(),
      py::object source_attention_mask = py::none(),
      py::object source_cache = py::none(),
      int64_t max_tokens = -1,
      const std::string& policy = "sliding-window") {
    seq_ = next_beams;
    AppendAttentionMask(std::move(mask_row_ids), std::move(source_attention_mask));
    ReorderCache(cache_row_ids, std::move(source_cache));
    EvictIfNeeded(max_tokens, policy);
    return DecodeNextLogits();
  }

 private:
  torch::Tensor CallModel(
      const torch::Tensor& tokens,
      const py::object& attention_mask,
      const py::object& cache,
      const py::object& position_ids,
      const py::object& cache_position,
      int64_t known_decode_position = -1) const {
    if (native_executor_kind_ == "causal_lm") {
      auto native_logits = prepared_native_causal_model_ != nullptr
          ? TryPreparedNativeCausalModelForward(
                prepared_native_causal_model_,
                tokens,
                attention_mask,
                cache,
                position_ids,
                cache_position,
                model_uses_attention_biases_,
                known_decode_position)
          : TryNativeCausalModelForward(
                model_,
                tokens,
                attention_mask,
                cache,
                position_ids,
                cache_position,
                true,
                model_uses_attention_biases_);
      if (native_logits.has_value()) {
        return native_logits.value();
      }
    }
    py::object out = model_(
        tokens,
        "attn_mask"_a = py::none(),
        "attention_mask"_a = attention_mask,
        "cache"_a = cache,
        "position_ids"_a = position_ids,
        "cache_position"_a = cache_position,
        "return_dict"_a = false);
    return py::cast<torch::Tensor>(out);
  }

  py::object model_;
  torch::Tensor seq_;
  py::object attention_mask_;
  py::object cache_;
  bool trace_;
  std::string native_executor_kind_;
  bool model_uses_attention_biases_;
  std::string attention_mask_mode_;
  std::shared_ptr<PreparedNativeCausalModel> prepared_native_causal_model_;
  bool decode_graph_enabled_ = false;
  torch::Tensor decode_token_buffer_;
  torch::Tensor decode_position_ids_buffer_;
  torch::Tensor decode_cache_position_buffer_;

  static torch::Tensor ResolveDecodeTokenSource(const torch::Tensor& seq, const py::object& cache) {
    if (!seq.defined() || cache.is_none() || seq.dim() < 2 || seq.size(1) <= 0) {
      return torch::Tensor();
    }
    return seq.slice(1, std::max<int64_t>(seq.size(1) - 1, 0), seq.size(1));
  }

  static bool SameShape(const torch::Tensor& a, const torch::Tensor& b) {
    if (a.dim() != b.dim()) {
      return false;
    }
    for (int64_t dim = 0; dim < a.dim(); ++dim) {
      if (a.size(dim) != b.size(dim)) {
        return false;
      }
    }
    return true;
  }

  static void CopyIntoStableBuffer(torch::Tensor* buffer, const torch::Tensor& source) {
    auto contiguous = source.contiguous();
    if (!buffer->defined() ||
        buffer->device() != contiguous.device() ||
        buffer->scalar_type() != contiguous.scalar_type() ||
        !buffer->is_contiguous() ||
        !SameShape(*buffer, contiguous)) {
      *buffer = torch::empty_like(contiguous);
    }
    buffer->copy_(contiguous);
  }

  void UpdateDecodeStepBuffers(c10::optional<torch::Tensor> decode_tokens_override = c10::nullopt) {
    auto decode_source = decode_tokens_override.has_value()
        ? decode_tokens_override.value()
        : ResolveDecodeTokenSource(seq_, cache_);
    if (!decode_source.defined()) {
      decode_token_buffer_ = torch::Tensor();
      decode_position_ids_buffer_ = torch::Tensor();
      decode_cache_position_buffer_ = torch::Tensor();
      return;
    }
    CopyIntoStableBuffer(&decode_token_buffer_, decode_source);
    if (!decode_graph_enabled_) {
      return;
    }
    auto pos = DecodePositionsForward(BatchSize(), SeqLen(), seq_);
    CopyIntoStableBuffer(&decode_position_ids_buffer_, py::cast<torch::Tensor>(pos[0]));
    CopyIntoStableBuffer(&decode_cache_position_buffer_, py::cast<torch::Tensor>(pos[1]));
  }

  void RefreshDecodeGraphState(c10::optional<torch::Tensor> decode_tokens_override = c10::nullopt) {
    if (!decode_graph_enabled_) {
      return;
    }
    UpdateDecodeStepBuffers(decode_tokens_override);
    if (!DecodeGraphEligible()) {
      decode_graph_enabled_ = false;
      decode_position_ids_buffer_ = torch::Tensor();
      decode_cache_position_buffer_ = torch::Tensor();
    }
  }
};


py::dict RuntimeInfo() {
  py::dict info;
  info["abi_version"] = kAbiVersion;
#if MODEL_STACK_WITH_CUDA
  info["compiled_with_cuda"] = true;
#else
  info["compiled_with_cuda"] = false;
#endif
  info["native_ops"] = std::vector<std::string>{
      "activation", "gated_activation", "embedding", "linear", "linear_module", "bitnet_transform_input", "bitnet_linear", "bitnet_linear_compute_packed", "bitnet_linear_from_float", "bitnet_int8_linear_from_float", "bitnet_int8_fused_qkv_packed_heads_projection", "int4_linear", "int4_linear_grad_input", "nf4_linear", "fp8_linear", "int8_quantize_activation", "int8_quantize_activation_transpose", "int8_quantize_relu2_activation", "int8_quantize_leaky_relu_half2_activation", "int8_linear", "int8_linear_from_float", "int8_linear_grad_weight_from_float", "int8_attention", "int8_attention_from_float", "pack_bitnet_weight", "bitnet_runtime_row_quantize", "pack_linear_weight", "mlp", "qkv_projection", "pack_qkv_weights", "qkv_packed_heads_projection", "bitnet_qkv_packed_heads_projection", "bitnet_fused_qkv_packed_heads_projection", "qkv_heads_projection", "split_heads", "merge_heads", "head_output_projection", "prepare_attention_mask", "resolve_position_ids", "create_causal_mask", "resolve_rotary_embedding", "token_counts", "append_tokens", "decode_positions", "rms_norm", "add_rms_norm", "residual_add", "layer_norm", "add_layer_norm", "rope", "kv_cache_append", "kv_cache_write", "kv_cache_gather", "paged_kv_assign_blocks", "paged_kv_reserve_pages", "paged_kv_read_range", "paged_kv_read_last", "paged_kv_append", "paged_kv_compact", "paged_kv_gather", "paged_kv_write", "int3_kv_pack", "int3_kv_dequantize", "paged_attention_decode", "attention_decode", "attention_prefill", "sampling", "beam_search_step", "incremental_beam_search"};
  info["planned_ops"] = std::vector<std::string>{
      "activation", "gated_activation", "embedding", "linear", "linear_module", "bitnet_transform_input", "bitnet_linear", "bitnet_linear_compute_packed", "bitnet_linear_from_float", "bitnet_int8_linear_from_float", "bitnet_int8_fused_qkv_packed_heads_projection", "int4_linear", "int4_linear_grad_input", "nf4_linear", "fp8_linear", "int8_quantize_activation", "int8_quantize_activation_transpose", "int8_quantize_relu2_activation", "int8_quantize_leaky_relu_half2_activation", "int8_linear", "int8_linear_from_float", "int8_linear_grad_weight_from_float", "int8_attention", "int8_attention_from_float", "pack_bitnet_weight", "bitnet_runtime_row_quantize", "pack_linear_weight", "mlp", "qkv_projection", "pack_qkv_weights", "qkv_packed_heads_projection", "bitnet_qkv_packed_heads_projection", "bitnet_fused_qkv_packed_heads_projection", "qkv_heads_projection", "split_heads", "merge_heads", "head_output_projection", "prepare_attention_mask", "resolve_position_ids", "create_causal_mask", "resolve_rotary_embedding", "token_counts", "append_tokens", "decode_positions", "rms_norm", "add_rms_norm", "residual_add", "layer_norm", "add_layer_norm", "rope", "kv_cache_append", "kv_cache_write", "kv_cache_gather", "paged_kv_assign_blocks", "paged_kv_reserve_pages", "paged_kv_read_range", "paged_kv_read_last", "paged_kv_append", "paged_kv_compact", "paged_kv_gather", "paged_kv_write", "int3_kv_pack", "int3_kv_dequantize", "paged_attention_decode", "attention_decode",
      "attention_prefill", "sampling", "beam_search_step", "incremental_beam_search"};
  info["linear_backend_default"] = HasCublasLtLinearBackend() ? "cublaslt" : "aten";
  info["linear_sm8x_aten_auto_disable_env"] = std::string("MODEL_STACK_DISABLE_SM8X_ATEN_LINEAR_AUTO");
  info["linear_backends_supported"] = SupportedLinearBackends();
  info["linear_backends_planned"] = PlannedLinearBackends();
  info["cublaslt_linear_dtypes"] = HasCublasLtLinearBackend()
      ? std::vector<std::string>{"float32", "float16", "bfloat16"}
      : std::vector<std::string>{};
  info["bitnet_available"] = true;
  info["bitnet_arches"] = std::vector<std::string>{"sm80+"};
  info["bitnet_linear_dtypes"] = std::vector<std::string>{"float32", "float16", "bfloat16"};
  info["bitnet_weight_storage"] = std::string("uint8_packed_2bit_ternary");
  info["bitnet_layout_version"] = 1;
  info["bitnet_kernel_family"] = std::string("decode_persistent_prefill_tiled_splitk");
  info["bitnet_decode_rows_buckets"] = std::vector<int>{1, 2, 4, 8};
  info["bitnet_decode_scheduler"] = std::string("persistent_cta");
  info["bitnet_prefill_scheduler"] = std::string("tiled_or_splitk");
  info["bitnet_attention_kernel_family"] = std::string("fused_qkv_projection_plus_native_attention");
  info["bitnet_attention_qkv_scheduler"] = std::string("decode_or_prefill_fused_qkv_dispatch");
  info["bitnet_splitk_env"] = std::string("MODEL_STACK_DISABLE_BITNET_SPLITK");
  info["bitnet_persistent_decode_env"] = std::string("MODEL_STACK_DISABLE_BITNET_PERSISTENT_DECODE");
  info["bitnet_hopper_dense_decode_cache_disable_env"] =
      std::string("MODEL_STACK_DISABLE_HOPPER_DENSE_DECODE_CACHE");
  info["bitnet_hopper_dense_decode_cache_mode_env"] =
      std::string("MODEL_STACK_HOPPER_DENSE_DECODE_CACHE_MODE");
  info["bitnet_hopper_dense_decode_fused_append_max_cache_length_env"] =
      std::string("MODEL_STACK_HOPPER_DENSE_DECODE_FUSED_APPEND_MAX_CACHE_LENGTH");
  info["bitnet_hopper_dense_decode_fused_append_default_max_cache_length"] = int64_t(4096);
  info["bitnet_projected_qkv_decode_fused_append_disable_env"] =
      std::string("MODEL_STACK_DISABLE_BITNET_PROJECTED_QKV_DECODE_FUSED_APPEND");
  info["bitnet_decode_fused_norm_rows_enable_env"] =
      std::string("MODEL_STACK_ENABLE_BITNET_DECODE_FUSED_NORM_ROWS");
  info["bitnet_decode_fused_norm_rows_disable_env"] =
      std::string("MODEL_STACK_DISABLE_BITNET_DECODE_FUSED_NORM_ROWS");
  info["attention_arches"] =
#if defined(MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL) && MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL
      std::vector<std::string>{"sm50+", "sm90_specialized", "sm90a_experimental"};
#else
      std::vector<std::string>{"sm50+", "sm90_specialized"};
#endif
  info["attention_sm80_inference_prefill_compiled"] =
#if defined(MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA) && MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA
      true;
#else
      false;
#endif
  info["attention_pytorch_memeff_prefill_compiled"] =
#if defined(MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA) && MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA
      true;
#else
      false;
#endif
  info["attention_cutlass_prefill_compiled"] =
#if defined(MODEL_STACK_WITH_CUTLASS_FMHA) && MODEL_STACK_WITH_CUTLASS_FMHA
      true;
#else
      false;
#endif
  info["attention_sm80_flash_prefill_compiled"] =
#if defined(MODEL_STACK_WITH_LOCAL_FLASH_STYLE_PREFILL) && MODEL_STACK_WITH_LOCAL_FLASH_STYLE_PREFILL
      true;
#else
      false;
#endif
  info["attention_sm80_inference_prefill_kernel_family"] =
      std::string("model_stack_sm80_causal_prefill_64x64_rf");
  info["attention_sm80_inference_prefill_kernel_env"] =
      std::string("MODEL_STACK_SM80_INFERENCE_PREFILL_KERNEL");
  info["attention_sm80_flash_prefill_enable_env"] =
      std::string("MODEL_STACK_ENABLE_ATTENTION_PREFILL_SM80_FLASH");
  info["sm90a_experimental_build_requested"] =
#if defined(MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL) && MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL
      true;
#else
      false;
#endif
  info["int4_linear_dtypes"] = HasCudaInt4LinearKernel()
      ? std::vector<std::string>{"float32", "float16", "bfloat16"}
      : std::vector<std::string>{};
  info["int4_linear_arches"] =
#if defined(MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL) && MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL
      std::vector<std::string>{"sm50+", "sm90_specialized", "sm90a_experimental"};
#else
      std::vector<std::string>{"sm50+", "sm90_specialized"};
#endif
  info["int4_linear_kernel_family"] = std::string("packed_byte_decode_sm90_with_default_w4a4_imma_path");
  info["int4_linear_sm90_tile"] = std::vector<int>{8, 128, 128};
  info["int4_linear_imma_tile"] = std::vector<int>{8, 8, 32};
  info["int4_linear_imma_requires"] = std::vector<std::string>{"sm90"};
  info["int4_linear_imma_disable_env"] = std::string("MODEL_STACK_DISABLE_INT4_IMMA_ACT_QUANT");
  info["int4_linear_weight_storage"] = std::string("uint8_packed_symmetric_per_channel");
  info["nf4_linear_dtypes"] = HasCudaNf4LinearKernel()
      ? std::vector<std::string>{"float32", "float16", "bfloat16"}
      : std::vector<std::string>{};
  info["nf4_linear_arches"] = std::vector<std::string>{"sm50+"};
  info["nf4_linear_kernel_family"] = std::string("packed_nf4_codebook_decode");
  info["nf4_linear_weight_storage"] = std::string("uint8_packed_nf4_per_channel");
  info["fp8_linear_dtypes"] = HasCudaFp8LinearKernel()
      ? std::vector<std::string>{"float32", "float16", "bfloat16"}
      : std::vector<std::string>{};
  info["fp8_linear_arches"] = std::vector<std::string>{"sm50+"};
  info["fp8_linear_kernel_family"] = std::string("scaled_fp8_weight_linear");
  info["fp8_linear_weight_storage"] = std::string("fake_fp8_float_tensor_plus_scalar_scale");
  info["int8_linear_dtypes"] = HasCudaInt8LinearKernel()
      ? std::vector<std::string>{"float32", "float16", "bfloat16"}
      : std::vector<std::string>{};
  info["int8_linear_arches"] =
#if defined(MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL) && MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL
      std::vector<std::string>{"sm61+", "sm90_specialized", "sm90a_experimental"};
#else
      std::vector<std::string>{"sm61+", "sm90_specialized"};
#endif
  info["int8_linear_kernel_family"] = std::string("sm90a_wgmma_experimental_with_wmma_sm90_dp4a_tiled_fallback");
  info["int8_linear_tensorcore_tile"] = std::vector<int>{16, 16, 16};
  info["int8_linear_tensorcore_arches"] = std::vector<std::string>{"sm90_specialized"};
  info["int8_linear_wgmma_tile"] = std::vector<int>{64, 8, 32};
  info["int8_linear_wgmma_requires"] =
      std::vector<std::string>{"sm90a", "u8_activation_rebias", "s8_weight_correction"};
  info["int8_linear_wgmma_env"] = std::string("MODEL_STACK_ENABLE_INT8_LINEAR_WGMMA");
  info["int8_linear_wgmma_min_ops_env"] = std::string("MODEL_STACK_INT8_LINEAR_WGMMA_MIN_OPS");
  info["int8_linear_large_gemm_backend"] = std::string("cublaslt_int8");
  info["int8_linear_wgmma_build_requested"] = info["sm90a_experimental_build_requested"];
  info["int8_linear_weight_storage"] = std::string("int8_symmetric_per_channel");
  info["int8_linear_activation_storage"] = std::string("int8_symmetric_per_row");
  info["int8_linear_frontend_kernel_family"] =
      std::string("warp_rowwise_quant_frontend_provided_scale_onepass_shared_cache_wide2_plus_native_int8_linear");
  info["int8_linear_frontend_warp_row_disable_env"] = std::string("MODEL_STACK_DISABLE_INT8_QUANT_WARP_ROW");
  info["int8_linear_frontend_warp_rows_per_block_env"] =
      std::string("MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK");
  info["int8_linear_frontend_shared_cache_disable_env"] =
      std::string("MODEL_STACK_DISABLE_INT8_QUANT_SHARED_CACHE");
  info["int8_linear_frontend_vec4_enable_env"] = std::string("MODEL_STACK_ENABLE_INT8_QUANT_VEC4");
  info["int8_linear_grad_weight_transpose_tile_env"] = std::string("MODEL_STACK_INT8_TRANSPOSE_TILE_DIM");
  info["int8_linear_grad_weight_transpose_tile_options"] = std::vector<int>{32, 64};
  info["int8_linear_wgmma_activation_strategy"] =
      std::string("rebias_s8_to_u8_minus_128_weight_sum");
  info["int8_attention_dtypes"] = HasCudaInt8AttentionKernel()
      ? std::vector<std::string>{"float32", "float16", "bfloat16"}
      : std::vector<std::string>{};
  info["int8_attention_qkv_storage"] = std::string("int8_symmetric_per_row");
  info["int8_attention_frontend_kernel_family"] =
      std::string("rowwise_quant_frontend_plus_native_int8_attention");
  info["int8_attention_kernel_family"] =
      std::string("sm90a_wgmma_experimental_with_sm90_bulk_async_wmma_online_softmax_and_generic_fallback");
  info["int8_attention_tensorcore_tile"] = std::vector<int>{16, 16, 16};
  info["int8_attention_tensorcore_head_dims"] = std::vector<int>{32, 64, 96, 128, 192, 256};
  info["int8_attention_wgmma_tile"] = std::vector<int>{64, 8, 32};
  info["int8_attention_wgmma_head_dims"] = std::vector<int>{32, 64, 96, 128};
  info["int8_attention_wgmma_requires"] =
      std::vector<std::string>{"sm90a", "u8_q_rebias", "s8_k_correction", "shared_descriptor_wgmma"};
  info["int8_attention_wgmma_env"] = std::string("MODEL_STACK_ENABLE_INT8_ATTENTION_WGMMA");
  info["int8_attention_wgmma_disable_env"] = std::string("MODEL_STACK_DISABLE_INT8_ATTENTION_WGMMA");
  info["int8_attention_wgmma_min_work_env"] = std::string("MODEL_STACK_INT8_ATTENTION_WGMMA_MIN_WORK");
  info["int8_attention_sm90_pipeline_stages"] = 2;
  info["int8_attention_sm90_bulk_async"] = true;
  info["int8_attention_sm90_bulk_async_requires"] = std::vector<std::string>{"sm90", "16b_alignment", "bulk_async_barrier"};
  info["int8_attention_scheduler"] = std::string("cta_grid_or_persistent");
  info["int8_attention_persistent_env"] = std::string("MODEL_STACK_ENABLE_INT8_ATTENTION_PERSISTENT");
  info["int8_attention_persistent_disable_env"] = std::string("MODEL_STACK_DISABLE_INT8_ATTENTION_PERSISTENT");
  info["int8_attention_persistent_waves_env"] = std::string("MODEL_STACK_INT8_ATTENTION_PERSISTENT_WAVES");
  info["int8_attention_persistent_waves_default"] = static_cast<int64_t>(2);
  info["int8_attention_persistent_requires"] = std::vector<std::string>{"sm90", "optimized_path", "grid_stride_tiles"};
  info["int8_attention_wgmma_build_requested"] = info["sm90a_experimental_build_requested"];
  info["int8_attention_optimized_default"] = true;
  info["int8_attention_optimized_min_work_default"] = static_cast<int64_t>(32768);
  info["int8_attention_optimized_small_seq_min_head_dim_default"] = static_cast<int64_t>(256);
  info["int8_attention_optimized_min_work_env"] =
      std::string("MODEL_STACK_INT8_ATTENTION_OPTIMIZED_MIN_WORK");
  info["int8_attention_optimized_small_seq_min_head_dim_env"] =
      std::string("MODEL_STACK_INT8_ATTENTION_OPTIMIZED_SMALL_SEQ_MIN_HEAD_DIM");
  info["int8_attention_specializations"] = std::vector<std::string>{
      "nomask",
      "bool_mask",
      "additive_mask",
      "causal",
      "decode_q1_nomask_opt_in",
  };
  info["int8_attention_decode_specialized_env"] =
      std::string("MODEL_STACK_ENABLE_INT8_ATTENTION_DECODE_SPECIALIZED");
  info["sm90_specialized_ops"] = std::vector<std::string>{
      "attention_decode",
      "attention_prefill",
      "int8_attention",
      "int4_linear",
      "int8_linear",
  };
  info["sm90a_advanced_ops"] =
#if defined(MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL) && MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL
      std::vector<std::string>{"int8_attention", "int8_linear"};
#else
      std::vector<std::string>{};
#endif
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
    cuda_backend_ops.push_back("beam_search_step");
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
    cuda_backend_ops.push_back("int3_kv_pack");
    cuda_backend_ops.push_back("int3_kv_dequantize");
  }
  if (HasCudaAttentionKernel()) {
    cuda_backend_ops.push_back("attention");
  }
  if (HasCudaPagedAttentionDecodeKernel()) {
    cuda_backend_ops.push_back("paged_attention_decode");
  }
  if (HasCublasLtLinearBackend()) {
    cuda_backend_ops.push_back("linear");
    cuda_backend_ops.push_back("linear_module");
    cuda_backend_ops.push_back("qkv_projection");
  }
  if (t10::bitnet::HasCudaBitNetFusedQkvKernel()) {
    cuda_backend_ops.push_back("bitnet_fused_qkv_packed_heads_projection");
  }
  if (t10::bitnet::HasCudaBitNetInputFrontendKernel()) {
    cuda_backend_ops.push_back("bitnet_transform_input");
  }
  if (t10::bitnet::HasCudaBitNetLinearKernel()) {
    cuda_backend_ops.push_back("bitnet_linear");
    cuda_backend_ops.push_back("bitnet_linear_compute_packed");
    cuda_backend_ops.push_back("bitnet_runtime_row_quantize");
  }
  if (t10::bitnet::HasCudaBitNetLinearKernel() && t10::bitnet::HasCudaBitNetInputFrontendKernel()) {
    cuda_backend_ops.push_back("bitnet_linear_from_float");
  }
  if (HasCudaInt4LinearKernel()) {
    cuda_backend_ops.push_back("int4_linear");
    cuda_backend_ops.push_back("int4_linear_grad_input");
  }
  if (HasCudaNf4LinearKernel()) {
    cuda_backend_ops.push_back("nf4_linear");
  }
  if (HasCudaFp8LinearKernel()) {
    cuda_backend_ops.push_back("fp8_linear");
  }
  if (HasCudaInt8LinearKernel()) {
    cuda_backend_ops.push_back("int8_linear");
  }
  if (HasCudaInt8QuantFrontendKernel()) {
    cuda_backend_ops.push_back("int8_quantize_activation");
    cuda_backend_ops.push_back("int8_quantize_activation_transpose");
    cuda_backend_ops.push_back("int8_quantize_relu2_activation");
    cuda_backend_ops.push_back("int8_quantize_leaky_relu_half2_activation");
  }
  if (HasCudaInt8LinearKernel()) {
    cuda_backend_ops.push_back("bitnet_int8_linear_from_float");
  }
  if (HasCudaInt8LinearKernel()) {
    cuda_backend_ops.push_back("bitnet_int8_fused_qkv_packed_heads_projection");
  }
  if (HasCudaInt8LinearKernel() && HasCudaInt8QuantFrontendKernel()) {
    cuda_backend_ops.push_back("int8_linear_from_float");
    cuda_backend_ops.push_back("int8_linear_grad_weight_from_float");
  }
  if (HasCudaInt8AttentionKernel()) {
    cuda_backend_ops.push_back("int8_attention");
  }
  if (HasCudaInt8AttentionKernel() && HasCudaInt8QuantFrontendKernel()) {
    cuda_backend_ops.push_back("int8_attention_from_float");
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SpeculativeAcceptForward(
    const torch::Tensor& target_probs,
    const torch::Tensor& draft_probs,
    const torch::Tensor& draft_token_ids,
    const c10::optional<torch::Tensor>& bonus_probs,
    const c10::optional<torch::Tensor>& bonus_enabled,
    const std::string& method,
    double posterior_threshold,
    double posterior_alpha);
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

std::vector<torch::Tensor> Int3KvPackForward(const torch::Tensor& x) {
  TORCH_CHECK(x.defined(), "int3_kv_pack_forward: x must be defined");
  TORCH_CHECK(x.dim() >= 1, "int3_kv_pack_forward: x must have rank >= 1");
  TORCH_CHECK(x.size(-1) > 0, "int3_kv_pack_forward: last dim must be non-empty");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
          x.scalar_type() == torch::kBFloat16,
      "int3_kv_pack_forward: x must be float32, float16, or bfloat16");

  if (x.is_cuda() && HasCudaKvCacheKernel()) {
    return CudaInt3PackLastDimForward(x);
  }
  TORCH_CHECK(!x.is_cuda(), "int3_kv_pack_forward: CUDA int3 kernel is unavailable");

  auto x_cpu = x.to(torch::TensorOptions().device(torch::kCPU).dtype(x.scalar_type())).contiguous();
  const auto dim = x_cpu.size(-1);
  const auto vectors = x_cpu.numel() / dim;
  const auto groups = (dim + 7) / 8;
  const auto packed_dim = groups * 3;
  std::vector<int64_t> packed_sizes(x_cpu.sizes().begin(), x_cpu.sizes().end());
  packed_sizes.back() = packed_dim;
  std::vector<int64_t> scale_sizes(x_cpu.sizes().begin(), x_cpu.sizes().end() - 1);
  auto packed = torch::empty(packed_sizes, torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
  auto scale = torch::empty(scale_sizes, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  auto packed_2d = packed.reshape({vectors, packed_dim});
  auto scale_flat = scale.reshape({vectors});
  auto packed_ptr = packed_2d.data_ptr<uint8_t>();
  auto scale_ptr = scale_flat.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_cpu.scalar_type(),
      "model_stack_cpu_int3_pack_lastdim",
      [&] {
        const auto* x_ptr = x_cpu.data_ptr<scalar_t>();
        for (int64_t vector = 0; vector < vectors; ++vector) {
          const int64_t base = vector * dim;
          float max_abs = 0.0f;
          for (int64_t d = 0; d < dim; ++d) {
            max_abs = std::max(max_abs, std::fabs(static_cast<float>(x_ptr[base + d])));
          }
          const float s = max_abs > 0.0f ? max_abs / 3.0f : 1.0f;
          scale_ptr[vector] = s;
          for (int64_t group = 0; group < groups; ++group) {
            uint32_t word = 0;
            for (int i = 0; i < 8; ++i) {
              const int64_t d = group * 8 + i;
              int code = 3;
              if (d < dim) {
                int q = static_cast<int>(std::nearbyint(static_cast<float>(x_ptr[base + d]) / s));
                q = std::max(-3, std::min(3, q));
                code = q + 3;
              }
              word |= (static_cast<uint32_t>(code) & 0x7u) << (i * 3);
            }
            const int64_t out = vector * packed_dim + group * 3;
            packed_ptr[out + 0] = static_cast<uint8_t>(word & 0xFFu);
            packed_ptr[out + 1] = static_cast<uint8_t>((word >> 8) & 0xFFu);
            packed_ptr[out + 2] = static_cast<uint8_t>((word >> 16) & 0xFFu);
          }
        }
      });
  return {packed.contiguous(), scale.contiguous()};
}

torch::Tensor Int3KvDequantizeForward(
    const torch::Tensor& packed,
    const torch::Tensor& scale,
    int64_t original_last_dim,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(packed.defined() && scale.defined(), "int3_kv_dequantize_forward: packed and scale must be defined");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "int3_kv_dequantize_forward: packed must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "int3_kv_dequantize_forward: scale must be float32");
  TORCH_CHECK(packed.dim() >= 1, "int3_kv_dequantize_forward: packed must have rank >= 1");
  TORCH_CHECK(original_last_dim > 0, "int3_kv_dequantize_forward: original_last_dim must be positive");
  TORCH_CHECK(packed.size(-1) % 3 == 0, "int3_kv_dequantize_forward: packed last dim must be divisible by 3");
  TORCH_CHECK(packed.device() == scale.device(), "int3_kv_dequantize_forward: device mismatch");

  if (packed.is_cuda() && scale.is_cuda() && HasCudaKvCacheKernel()) {
    return CudaInt3DequantizeLastDimForward(packed, scale, original_last_dim, out_dtype);
  }
  TORCH_CHECK(!packed.is_cuda(), "int3_kv_dequantize_forward: CUDA int3 kernel is unavailable");

  auto packed_cpu = packed.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8)).contiguous();
  auto scale_cpu = scale.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
  const auto packed_dim = packed_cpu.size(-1);
  const auto groups = packed_dim / 3;
  const auto vectors = packed_cpu.numel() / packed_dim;
  TORCH_CHECK(scale_cpu.numel() == vectors, "int3_kv_dequantize_forward: scale shape mismatch");
  std::vector<int64_t> out_sizes(packed_cpu.sizes().begin(), packed_cpu.sizes().end());
  out_sizes.back() = original_last_dim;
  const auto dtype = out_dtype.has_value() ? out_dtype.value() : torch::kFloat32;
  auto out = torch::empty(out_sizes, torch::TensorOptions().device(torch::kCPU).dtype(dtype));
  auto out_2d = out.reshape({vectors, original_last_dim});
  auto packed_2d = packed_cpu.reshape({vectors, packed_dim});
  auto scale_flat = scale_cpu.reshape({vectors});
  const auto* packed_ptr = packed_2d.data_ptr<uint8_t>();
  const auto* scale_ptr = scale_flat.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "model_stack_cpu_int3_dequantize_lastdim",
      [&] {
        auto* out_ptr = out_2d.data_ptr<scalar_t>();
        for (int64_t vector = 0; vector < vectors; ++vector) {
          for (int64_t d = 0; d < original_last_dim; ++d) {
            const int64_t group = d / 8;
            const int in_group = static_cast<int>(d % 8);
            const int64_t p = vector * packed_dim + group * 3;
            const uint32_t word = static_cast<uint32_t>(packed_ptr[p + 0]) |
                (static_cast<uint32_t>(packed_ptr[p + 1]) << 8) |
                (static_cast<uint32_t>(packed_ptr[p + 2]) << 16);
            const int code = static_cast<int>((word >> (in_group * 3)) & 0x7u);
            out_ptr[vector * original_last_dim + d] =
                static_cast<scalar_t>(static_cast<float>(code - 3) * scale_ptr[vector]);
          }
        }
      });
  return out;
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

std::tuple<torch::Tensor, torch::Tensor, int64_t> PagedKvAssignBlocksForwardUnchecked(
    const torch::Tensor& block_table,
    const torch::Tensor& block_ids,
    const torch::Tensor& starts,
    int64_t total,
    int64_t page_size,
    int64_t next_page_id) {
  auto table_long = block_table.to(torch::kLong).contiguous();
  auto block_ids_long = block_ids.to(torch::kLong).contiguous();
  auto starts_long = starts.to(torch::kLong).contiguous();
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
torch::Tensor PagedKvWriteForwardUnchecked(
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t> PagedKvAppendForwardUnchecked(
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
        page_size_(std::max<int64_t>(page_size, 1)),
        max_length_(0) {
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
    max_length_ = 0;
  }

  void Append(
      const torch::Tensor& k_chunk,
      const torch::Tensor& v_chunk,
      const c10::optional<torch::Tensor>& block_ids) {
    const auto total = k_chunk.size(2);
    const bool single_row_fast_path = batch_ == 1 && (!block_ids.has_value() || !block_ids.value().defined());
    if (single_row_fast_path) {
      int64_t next_max_length = max_length_;
      if (total > 0) {
        next_max_length = max_length_ + total;
      }
      auto k_chunk_local = k_chunk.to(k_pages_.options()).contiguous();
      auto v_chunk_local = v_chunk.to(v_pages_.options()).contiguous();
      if (total > 0) {
        const int64_t start = max_length_;
        const int64_t end = start + total;
        const int64_t assigned_blocks = (max_length_ + page_size_ - 1) / page_size_;
        const int64_t needed_blocks = (end + page_size_ - 1) / page_size_;
        int64_t next_blocks = block_table_.size(1);
        if (next_blocks < needed_blocks) {
          next_blocks = std::max<int64_t>(1, next_blocks > 0 ? next_blocks : 1);
          while (next_blocks < needed_blocks) {
            next_blocks *= 2;
          }
        }

        auto next_block_table = block_table_;
        if (next_blocks != block_table_.size(1)) {
          next_block_table = torch::full({batch_, next_blocks}, -1, block_table_.options());
          if (block_table_.numel() > 0) {
            next_block_table.slice(1, 0, block_table_.size(1)).copy_(block_table_);
          }
        } else {
          next_block_table = block_table_.clone();
        }

        const int64_t missing_count = std::max<int64_t>(0, needed_blocks - assigned_blocks);
        if (missing_count > 0) {
          auto new_ids = missing_count == 1
              ? torch::full({1}, page_count_, block_table_.options())
              : torch::arange(page_count_, page_count_ + missing_count, block_table_.options());
          next_block_table[0].slice(0, assigned_blocks, needed_blocks).copy_(new_ids);
        }

        auto selected_block_table = next_block_table.slice(1, 0, needed_blocks).contiguous();
        auto positions = total == 1
            ? torch::full({1, 1}, start, block_table_.options())
            : (torch::arange(total, block_table_.options()) + start).view({1, total});
        auto next_page_count = page_count_ + missing_count;
        auto next_k_pages = PagedKvReservePagesForward(k_pages_, page_count_, next_page_count);
        auto next_v_pages = PagedKvReservePagesForward(v_pages_, page_count_, next_page_count);
        next_k_pages = PagedKvWriteForwardUnchecked(next_k_pages, selected_block_table, positions, k_chunk_local);
        next_v_pages = PagedKvWriteForwardUnchecked(next_v_pages, selected_block_table, positions, v_chunk_local);
        k_pages_ = next_k_pages;
        v_pages_ = next_v_pages;
        block_table_ = next_block_table;
        lengths_ = torch::full({batch_}, end, block_table_.options());
        page_count_ = next_page_count;
      }
      max_length_ = next_max_length;
      return;
    }

    auto ids = NormalizeBlockIds(block_ids, k_chunk.size(0));
    auto starts_long = lengths_.index_select(0, ids);
    int64_t next_max_length = max_length_;
    const bool updates_all_rows =
        !block_ids.has_value() || !block_ids.value().defined() || ids.numel() == batch_;
    if (ids.numel() > 0 && total > 0) {
      if (updates_all_rows) {
        next_max_length = max_length_ + total;
      } else {
        next_max_length = std::max(max_length_, starts_long.max().item<int64_t>() + total);
      }
    }
    auto result = PagedKvAppendForwardUnchecked(
        k_pages_,
        v_pages_,
        block_table_,
        lengths_,
        page_count_,
        k_chunk.to(k_pages_.options()).contiguous(),
        v_chunk.to(v_pages_.options()).contiguous(),
        ids);
    k_pages_ = std::get<0>(result);
    v_pages_ = std::get<1>(result);
    block_table_ = std::get<2>(result);
    lengths_ = std::get<3>(result);
    page_count_ = std::get<4>(result);
    max_length_ = next_max_length;
  }

  c10::optional<torch::Tensor> AppendProjectedQkvRotary(
      const torch::Tensor& projected,
      const torch::Tensor& cos,
      const torch::Tensor& sin,
      int64_t q_size,
      int64_t k_size,
      int64_t v_size,
      int64_t q_heads,
      int64_t kv_heads) {
#if MODEL_STACK_WITH_CUDA
    if (!HasCudaKvCacheKernel() || !projected.is_cuda() || !k_pages_.is_cuda() ||
        !v_pages_.is_cuda() || !block_table_.is_cuda() || !lengths_.is_cuda()) {
      return c10::nullopt;
    }
    TORCH_CHECK(projected.dim() == 3 && projected.size(0) == batch_ && projected.size(1) == 1,
                "PagedKvLayerState: projected QKV must have shape (B,1,QKV)");
    TORCH_CHECK(q_heads > 0 && kv_heads == heads_,
                "PagedKvLayerState: projected QKV head count mismatch");
    TORCH_CHECK(q_size % q_heads == 0 && k_size % kv_heads == 0 && v_size % kv_heads == 0,
                "PagedKvLayerState: projected QKV sizes must divide by head counts");
    TORCH_CHECK(q_size / q_heads == head_dim_ && k_size / kv_heads == head_dim_ &&
                    v_size / kv_heads == head_dim_,
                "PagedKvLayerState: projected QKV head_dim mismatch");
    TORCH_CHECK(projected.size(2) == q_size + k_size + v_size,
                "PagedKvLayerState: projected QKV size mismatch");
    TORCH_CHECK(cos.defined() && sin.defined() && cos.is_cuda() && sin.is_cuda(),
                "PagedKvLayerState: rotary cos/sin must be CUDA tensors");
    TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2 && cos.size(0) == 1 && sin.size(0) == 1 &&
                    cos.size(1) == head_dim_ && sin.size(1) == head_dim_,
                "PagedKvLayerState: fused projected QKV append requires one RoPE row");

    auto ids = torch::arange(batch_, block_table_.options());
    auto starts_long = lengths_.to(torch::kLong).contiguous();
    auto assign_result = PagedKvAssignBlocksForwardUnchecked(
        block_table_,
        ids,
        starts_long,
        1,
        page_size_,
        page_count_);
    auto next_block_table = std::get<0>(assign_result);
    auto selected_block_table = std::get<1>(assign_result);
    auto next_page_count = std::get<2>(assign_result);
    auto next_k_pages = PagedKvReservePagesForward(k_pages_, page_count_, next_page_count);
    auto next_v_pages = PagedKvReservePagesForward(v_pages_, page_count_, next_page_count);
    auto projected_local = projected.to(k_pages_.options()).contiguous();
    auto cos_local = cos.to(projected_local.options()).contiguous();
    auto sin_local = sin.to(projected_local.options()).contiguous();
    auto fused = CudaProjectedQkvRotaryPagedWriteForward(
        projected_local,
        cos_local,
        sin_local,
        next_k_pages,
        next_v_pages,
        selected_block_table,
        starts_long.view({batch_, 1}),
        q_size,
        k_size,
        v_size,
        q_heads,
        kv_heads);
    k_pages_ = fused[1];
    v_pages_ = fused[2];
    block_table_ = next_block_table;
    lengths_ = starts_long + 1;
    page_count_ = next_page_count;
    max_length_ = lengths_.numel() > 0 ? lengths_.max().item<int64_t>() : 0;
    return fused[0];
#else
    (void)projected;
    (void)cos;
    (void)sin;
    (void)q_size;
    (void)k_size;
    (void)v_size;
    (void)q_heads;
    (void)kv_heads;
    return c10::nullopt;
#endif
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

  std::shared_ptr<PagedKvLayerState> CloneRows(const torch::Tensor& row_ids) const {
    auto ids = NormalizeSelectionRowIds(row_ids);
    auto out = std::make_shared<PagedKvLayerState>(ids.numel(), heads_, head_dim_, page_size_, k_pages_);
    if (ids.numel() == 0) {
      return out;
    }
    auto gathered_lengths = lengths_.index_select(0, ids);
    const auto max_len = gathered_lengths.numel() > 0 ? gathered_lengths.max().item<int64_t>() : 0;
    if (max_len <= 0) {
      return out;
    }
    auto gathered_table = block_table_.index_select(0, ids).contiguous().clone();
    gathered_table.clamp_min_(0);
    auto gathered_k = PagedKvReadRangeForward(k_pages_.slice(0, 0, page_count_), gathered_table, gathered_lengths, 0, max_len);
    auto gathered_v = PagedKvReadRangeForward(v_pages_.slice(0, 0, page_count_), gathered_table, gathered_lengths, 0, max_len);

    auto kept_cpu = gathered_lengths.to(torch::kCPU);
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
      auto rows = torch::nonzero(gathered_lengths == length).view(-1);
      if (rows.numel() == 0) {
        continue;
      }
      out->Append(
          gathered_k.index_select(0, rows).slice(2, 0, length).contiguous(),
          gathered_v.index_select(0, rows).slice(2, 0, length).contiguous(),
          rows);
    }
    return out;
  }

  void ReorderRowsInPlace(const torch::Tensor& row_ids) {
    auto reordered = CloneRows(row_ids);
    batch_ = reordered->batch_;
    k_pages_ = reordered->k_pages_;
    v_pages_ = reordered->v_pages_;
    block_table_ = reordered->block_table_;
    lengths_ = reordered->lengths_;
    page_count_ = reordered->page_count_;
    max_length_ = reordered->max_length_;
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
    max_length_ = std::min(max_length_, std::max<int64_t>(keep, 0));
  }

  torch::Tensor KPages() const { return k_pages_; }
  torch::Tensor VPages() const { return v_pages_; }
  torch::Tensor BlockTable() const { return block_table_; }
  torch::Tensor Lengths() const { return lengths_; }
  int64_t PageCount() const { return page_count_; }
  int64_t MaxLength() const { return max_length_; }

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

  torch::Tensor NormalizeSelectionRowIds(const torch::Tensor& row_ids) const {
    TORCH_CHECK(row_ids.defined(), "PagedKvLayerState: row_ids must be defined");
    auto ids = row_ids.to(torch::kLong).contiguous().view(-1);
    if (ids.numel() == 0) {
      return ids;
    }
    const auto min_id = ids.min().item<int64_t>();
    const auto max_id = ids.max().item<int64_t>();
    TORCH_CHECK(min_id >= 0, "PagedKvLayerState: row_ids must be non-negative");
    TORCH_CHECK(max_id < batch_, "PagedKvLayerState: row_ids exceed cache batch size");
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
  int64_t max_length_;
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

  c10::optional<torch::Tensor> AppendProjectedQkvRotary(
      int64_t layer_idx,
      const torch::Tensor& projected,
      const torch::Tensor& cos,
      const torch::Tensor& sin,
      int64_t q_size,
      int64_t k_size,
      int64_t v_size,
      int64_t q_heads,
      int64_t kv_heads) {
    return Layer(layer_idx)->AppendProjectedQkvRotary(
        projected,
        cos,
        sin,
        q_size,
        k_size,
        v_size,
        q_heads,
        kv_heads);
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

  std::shared_ptr<PagedKvCacheState> CloneRows(const torch::Tensor& row_ids) const {
    auto ids = NormalizeSelectionRowIds(row_ids);
    auto out = std::make_shared<PagedKvCacheState>(
        ids.numel(),
        layers_count_,
        heads_,
        head_dim_,
        page_size_,
        layers_count_ > 0 ? layers_[0]->KPages() : torch::empty({0, heads_, page_size_, head_dim_}, torch::TensorOptions().dtype(torch::kFloat32)));
    for (int64_t layer_idx = 0; layer_idx < layers_count_; ++layer_idx) {
      out->layers_[static_cast<size_t>(layer_idx)] = layers_[static_cast<size_t>(layer_idx)]->CloneRows(ids);
    }
    return out;
  }

  void ReorderRowsInPlace(const torch::Tensor& row_ids) {
    auto reordered = CloneRows(row_ids);
    batch_ = reordered->batch_;
    layers_ = reordered->layers_;
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
    return Layer(layer_idx)->MaxLength();
  }

  int64_t NumLayers() const { return layers_count_; }

 private:
  torch::Tensor NormalizeSelectionRowIds(const torch::Tensor& row_ids) const {
    TORCH_CHECK(row_ids.defined(), "PagedKvCacheState: row_ids must be defined");
    auto ids = row_ids.to(torch::kLong).contiguous().view(-1);
    if (ids.numel() == 0) {
      return ids;
    }
    const auto min_id = ids.min().item<int64_t>();
    const auto max_id = ids.max().item<int64_t>();
    TORCH_CHECK(min_id >= 0, "PagedKvCacheState: row_ids must be non-negative");
    TORCH_CHECK(max_id < batch_, "PagedKvCacheState: row_ids exceed cache batch size");
    return ids;
  }

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
  const auto& v_chunk_write = v_chunk;
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
  next_v_pages = PagedKvWriteForward(next_v_pages, selected_block_table, positions, v_chunk_write);
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t> PagedKvAppendForwardUnchecked(
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    int64_t page_count,
    const torch::Tensor& k_chunk,
    const torch::Tensor& v_chunk,
    const torch::Tensor& block_ids) {
  auto block_table_long = block_table.to(torch::kLong).contiguous();
  auto lengths_long = lengths.to(torch::kLong).contiguous();
  auto block_ids_long = block_ids.to(torch::kLong).contiguous();
  auto k_chunk_contig = k_chunk.contiguous();
  const auto& v_chunk_write = v_chunk;
  const auto total = k_chunk_contig.size(2);
  if (total == 0 || block_ids_long.numel() == 0) {
    return std::make_tuple(k_pages, v_pages, block_table_long, lengths_long, page_count);
  }

  auto starts_long = lengths_long.index_select(0, block_ids_long);
  auto base = torch::arange(total, block_table_long.options()).view({1, total});
  auto positions = starts_long.view({block_ids_long.size(0), 1}) + base;
  auto assign_result = PagedKvAssignBlocksForwardUnchecked(
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
  next_k_pages = PagedKvWriteForwardUnchecked(next_k_pages, selected_block_table, positions, k_chunk_contig);
  next_v_pages = PagedKvWriteForwardUnchecked(next_v_pages, selected_block_table, positions, v_chunk_write);
  auto next_lengths = lengths_long.clone();
  next_lengths.index_copy_(0, block_ids_long, starts_long + total);
  return std::make_tuple(next_k_pages, next_v_pages, next_block_table, next_lengths, next_page_count);
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

torch::Tensor PagedKvWriteForwardUnchecked(
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& positions,
    const torch::Tensor& values) {
  auto block_table_long = block_table.to(torch::kLong).contiguous();
  auto positions_long = positions.to(torch::kLong).contiguous();

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
  int64_t needed = ResolveRopeSequenceLength(seq_len, position_ids);
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

std::vector<torch::Tensor> ResolveRotaryEmbeddingRangeForward(
    const torch::Tensor& reference,
    int64_t head_dim,
    double base_theta,
    double attention_scaling,
    int64_t start,
    int64_t end) {
  TORCH_CHECK(reference.defined(), "resolve_rotary_embedding_range_forward: reference must be defined");
  TORCH_CHECK(reference.dim() == 3, "resolve_rotary_embedding_range_forward: reference must have shape (B, T, D)");
  TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0,
              "resolve_rotary_embedding_range_forward: head_dim must be positive and even");
  TORCH_CHECK(std::isfinite(base_theta) && base_theta > 0.0,
              "resolve_rotary_embedding_range_forward: base_theta must be positive and finite");
  TORCH_CHECK(std::isfinite(attention_scaling),
              "resolve_rotary_embedding_range_forward: attention_scaling must be finite");
  TORCH_CHECK(start >= 0 && end >= start,
              "resolve_rotary_embedding_range_forward: invalid range");

  const auto count = end - start;
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(reference.device());
  auto t = torch::arange(start, end, float_opts);
  auto idx = torch::arange(0, head_dim, 2, torch::TensorOptions().dtype(torch::kInt64).device(reference.device()))
                 .to(torch::kFloat32);
  auto inv_freq = torch::pow(
      torch::full({idx.size(0)}, base_theta, float_opts),
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
  if (count == 0) {
    cos = cos.view({0, head_dim});
    sin = sin.view({0, head_dim});
  }
  return {cos, sin};
}

torch::Tensor ToAdditiveMask(const torch::Tensor& mask);

torch::Tensor PagedKvReadRangeForwardTrusted(
    const torch::Tensor& pages,
    const torch::Tensor& block_table_safe,
    const torch::Tensor& lengths_long,
    int64_t start,
    int64_t end) {
  const auto gather_seq = end - start;
  auto out = torch::zeros({block_table_safe.size(0), pages.size(1), gather_seq, pages.size(3)}, pages.options());
  if (block_table_safe.size(0) == 0 || gather_seq == 0) {
    return out;
  }
#if MODEL_STACK_WITH_CUDA
  if (pages.is_cuda() && block_table_safe.is_cuda() && lengths_long.is_cuda() && HasCudaKvCacheKernel()) {
    return CudaPagedKvReadRangeForward(pages, block_table_safe, lengths_long, start, end);
  }
#endif
  return PagedKvReadRangeForward(pages, block_table_safe, lengths_long, start, end);
}

c10::optional<torch::Tensor> TryPagedAttentionDecodeSdpaBridgeForward(
    const torch::Tensor& q,
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table_safe,
    const torch::Tensor& lengths_long,
    int64_t max_len,
    const c10::optional<torch::Tensor>& attn_mask,
    const c10::optional<double>& scale) {
  const int64_t sdpa_max_len = PagedDecodeSdpaMaxLength();
  if (sdpa_max_len <= 0 || max_len <= 0 || max_len > sdpa_max_len ||
      !q.is_cuda() || !k_pages.is_cuda() || !v_pages.is_cuda() ||
      !block_table_safe.is_cuda() || !lengths_long.is_cuda()) {
    return c10::nullopt;
  }

  auto k = PagedKvReadRangeForwardTrusted(k_pages, block_table_safe, lengths_long, 0, max_len);
  auto v = PagedKvReadRangeForwardTrusted(v_pages, block_table_safe, lengths_long, 0, max_len);
  c10::optional<torch::Tensor> sdpa_mask = c10::nullopt;
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    auto invalid = torch::arange(max_len, lengths_long.options()).view({1, 1, 1, max_len}) >=
        lengths_long.view({q.size(0), 1, 1, 1});
    auto mask = attn_mask.value();
    sdpa_mask = mask.scalar_type() == torch::kBool ? ToAdditiveMask(mask) : mask.to(torch::kFloat32);
    sdpa_mask = sdpa_mask.value().to(q.scalar_type());
    sdpa_mask = sdpa_mask.value().masked_fill(invalid, -std::numeric_limits<float>::infinity());
  } else if (q.size(0) > 1) {
    auto invalid = torch::arange(max_len, lengths_long.options()).view({1, 1, 1, max_len}) >=
        lengths_long.view({q.size(0), 1, 1, 1});
    sdpa_mask = ToAdditiveMask(invalid).to(q.scalar_type());
  }
  return at::scaled_dot_product_attention(
      q,
      k,
      v,
      sdpa_mask,
      0.0,
      false,
      scale.has_value() ? std::optional<double>(scale.value()) : std::nullopt,
      q.size(1) != k.size(1));
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

torch::Tensor PagedAttentionDecodeForward(
    const torch::Tensor& q,
    const torch::Tensor& k_pages,
    const torch::Tensor& v_pages,
    const torch::Tensor& block_table,
    const torch::Tensor& lengths,
    const c10::optional<torch::Tensor>& attn_mask,
    const c10::optional<double>& scale) {
  TORCH_CHECK(q.defined() && k_pages.defined() && v_pages.defined() && block_table.defined() && lengths.defined(),
              "paged_attention_decode_forward: q, page tensors, block_table, and lengths must be defined");
  TORCH_CHECK(q.dim() == 4 && q.size(2) == 1, "paged_attention_decode_forward: q must be rank-4 with q_len=1");
  TORCH_CHECK(k_pages.dim() == 4 && v_pages.dim() == 4,
              "paged_attention_decode_forward: k_pages and v_pages must be rank-4 (P,H,page_size,D)");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(),
              "paged_attention_decode_forward: k_pages and v_pages shapes must match");
  TORCH_CHECK(block_table.dim() == 2, "paged_attention_decode_forward: block_table must be rank-2 (B,max_blocks)");
  TORCH_CHECK(lengths.dim() == 1, "paged_attention_decode_forward: lengths must be rank-1");
  TORCH_CHECK(q.device() == k_pages.device() && q.device() == v_pages.device() && q.device() == block_table.device() &&
                  q.device() == lengths.device(),
              "paged_attention_decode_forward: device mismatch");
  TORCH_CHECK(q.scalar_type() == k_pages.scalar_type() && q.scalar_type() == v_pages.scalar_type(),
              "paged_attention_decode_forward: dtype mismatch between q and page tensors");
  TORCH_CHECK(block_table.size(0) == q.size(0) && lengths.size(0) == q.size(0),
              "paged_attention_decode_forward: batch dimension mismatch");
  TORCH_CHECK(k_pages.size(1) == v_pages.size(1), "paged_attention_decode_forward: key/value page head mismatch");
  TORCH_CHECK(q.size(1) % k_pages.size(1) == 0,
              "paged_attention_decode_forward: query heads must be a multiple of kv page heads");
  TORCH_CHECK(q.size(3) == k_pages.size(3), "paged_attention_decode_forward: head_dim mismatch");

  auto block_table_long = block_table.to(torch::kLong).contiguous();
  auto block_table_safe = block_table_long.clone();
  block_table_safe.clamp_min_(0);
  auto lengths_long = lengths.to(torch::kLong).contiguous().view(-1);
  if (block_table_safe.numel() > 0) {
    const auto max_page = block_table_safe.max().item<int64_t>();
    TORCH_CHECK(max_page < k_pages.size(0), "paged_attention_decode_forward: block_table entries exceed page pool size");
  }
  if (lengths_long.numel() > 0) {
    const auto min_len = lengths_long.min().item<int64_t>();
    const auto max_len = lengths_long.max().item<int64_t>();
    TORCH_CHECK(min_len >= 0, "paged_attention_decode_forward: lengths must be non-negative");
    TORCH_CHECK(max_len <= block_table_long.size(1) * k_pages.size(2),
                "paged_attention_decode_forward: lengths exceed logical block-table capacity");
    if (attn_mask.has_value() && attn_mask.value().defined()) {
      const auto& mask = attn_mask.value();
      TORCH_CHECK(mask.dim() == 4, "paged_attention_decode_forward: attn_mask must be rank-4 when defined");
      TORCH_CHECK(mask.size(0) == q.size(0) && mask.size(1) == q.size(1) && mask.size(2) == 1 && mask.size(3) == max_len,
                  "paged_attention_decode_forward: attn_mask shape must match (B,H,1,max_len)");
    }
  }

  const auto max_len = lengths_long.numel() > 0 ? lengths_long.max().item<int64_t>() : 0;
  if (max_len <= 0 || q.size(0) == 0) {
    return torch::zeros_like(q);
  }

  if (auto bridged = TryPagedAttentionDecodeSdpaBridgeForward(
          q, k_pages, v_pages, block_table_safe, lengths_long, max_len, attn_mask, scale)) {
    return bridged.value();
  }

#if MODEL_STACK_WITH_CUDA
  if (q.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda() && block_table_safe.is_cuda() && lengths_long.is_cuda() &&
      (!attn_mask.has_value() || !attn_mask.value().defined() || attn_mask.value().is_cuda()) &&
      HasCudaPagedAttentionDecodeKernel()) {
    return CudaPagedAttentionDecodeForward(q, k_pages, v_pages, block_table_safe, lengths_long, attn_mask, scale);
  }
#endif

  auto k = PagedKvReadRangeForward(k_pages, block_table_safe, lengths_long, 0, max_len);
  auto v = PagedKvReadRangeForward(v_pages, block_table_safe, lengths_long, 0, max_len);
  auto invalid = torch::arange(max_len, lengths_long.options()).view({1, 1, 1, max_len}) >=
      lengths_long.view({q.size(0), 1, 1, 1});

  c10::optional<torch::Tensor> merged_mask = c10::nullopt;
  if (attn_mask.has_value() && attn_mask.value().defined()) {
    if (attn_mask.value().scalar_type() == torch::kBool) {
      merged_mask = attn_mask.value().to(torch::kBool) | invalid;
    } else {
      merged_mask = ToAdditiveMask(attn_mask.value()).masked_fill(invalid, -std::numeric_limits<float>::infinity());
    }
  } else {
    merged_mask = invalid;
  }
  return NativeAttentionForward(q, k, v, merged_mask, false, scale);
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
  info["effective_kv_len"] = t10::policy::AttentionEffectiveKvLen(desc);
  info["head_dim"] = desc.head_dim;
  info["causal"] = desc.causal;
  info["trimmed_causal_tail"] = t10::policy::AttentionHasDeadTopLeftCausalKvTail(desc);
  info["sm80_flash_prefill_disabled"] = t10::policy::AttentionSm80FlashPrefillDisabled();
  info["sm80_flash_prefill_min_seq"] = t10::policy::AttentionSm80FlashPrefillMinSeq();
  info["sm80_flash_prefill_eligible"] = t10::policy::SupportsAttentionSm80FlashPrefill(desc);
  info["sm80_flash_prefill_device_supported"] = false;
  info["sm80_flash_prefill_selected"] = false;
  info["split_kv_eligible"] = t10::policy::SupportsAttentionSplitKv(desc);
  info["split_kv_block_n"] = t10::policy::AttentionSplitKvBlockN(desc);
  info["split_kv_num_m_blocks"] = t10::policy::AttentionSplitKvNumMBlocks(desc);
  info["split_kv_num_n_blocks"] = t10::policy::AttentionSplitKvNumNBlocks(desc);
  info["split_kv_splits"] = 1;
#if MODEL_STACK_WITH_CUDA
  if (q.is_cuda()) {
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    const auto* prop = at::cuda::getCurrentDeviceProperties();
    const bool device_supported = prop->major >= 8 && prop->major < 9;
    const int effective_sms =
        std::max(1, static_cast<int>(prop->multiProcessorCount)) * 2;
    info["sm80_flash_prefill_device_supported"] = device_supported;
    info["sm80_flash_prefill_selected"] =
        device_supported && t10::policy::PreferAttentionSm80FlashPrefill(desc);
    info["split_kv_effective_sms"] = effective_sms;
    info["split_kv_splits"] =
        t10::policy::SelectAttentionSplitKvSplits(desc, effective_sms);
  } else {
    info["split_kv_effective_sms"] = 0;
  }
#else
  info["split_kv_effective_sms"] = 0;
#endif
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SpeculativeAcceptForward(
    const torch::Tensor& target_probs,
    const torch::Tensor& draft_probs,
    const torch::Tensor& draft_token_ids,
    const c10::optional<torch::Tensor>& bonus_probs,
    const c10::optional<torch::Tensor>& bonus_enabled,
    const std::string& method,
    double posterior_threshold,
    double posterior_alpha) {
  TORCH_CHECK(target_probs.defined(), "speculative_accept_forward: target_probs must be defined");
  TORCH_CHECK(draft_probs.defined(), "speculative_accept_forward: draft_probs must be defined");
  TORCH_CHECK(draft_token_ids.defined(), "speculative_accept_forward: draft_token_ids must be defined");
  TORCH_CHECK(target_probs.dim() == 3, "speculative_accept_forward: target_probs must be rank-3 (B, K, V)");
  TORCH_CHECK(draft_probs.dim() == 3, "speculative_accept_forward: draft_probs must be rank-3 (B, K, V)");
  TORCH_CHECK(draft_token_ids.dim() == 2, "speculative_accept_forward: draft_token_ids must be rank-2 (B, K)");
  TORCH_CHECK(target_probs.size(0) == draft_probs.size(0) && target_probs.size(1) == draft_probs.size(1) &&
                  target_probs.size(2) == draft_probs.size(2),
              "speculative_accept_forward: target/draft probability shapes must match");
  TORCH_CHECK(target_probs.size(0) == draft_token_ids.size(0) && target_probs.size(1) == draft_token_ids.size(1),
              "speculative_accept_forward: batch/speculative length mismatch");
  if (bonus_probs.has_value() && bonus_probs.value().defined()) {
    TORCH_CHECK(bonus_probs.value().dim() == 2, "speculative_accept_forward: bonus_probs must be rank-2 (B, V)");
    TORCH_CHECK(bonus_probs.value().size(0) == target_probs.size(0) && bonus_probs.value().size(1) == target_probs.size(2),
                "speculative_accept_forward: bonus_probs shape mismatch");
  }
  if (bonus_enabled.has_value() && bonus_enabled.value().defined()) {
    TORCH_CHECK(bonus_enabled.value().numel() == target_probs.size(0),
                "speculative_accept_forward: bonus_enabled must have one flag per batch row");
  }

#if MODEL_STACK_WITH_CUDA
  if (target_probs.is_cuda() && draft_probs.is_cuda() && draft_token_ids.is_cuda() && HasCudaSamplingKernel()) {
    TORCH_CHECK(!bonus_probs.has_value() || !bonus_probs.value().defined() || bonus_probs.value().is_cuda(),
                "speculative_accept_forward: CUDA path requires bonus_probs on CUDA when provided");
    TORCH_CHECK(!bonus_enabled.has_value() || !bonus_enabled.value().defined() || bonus_enabled.value().is_cuda(),
                "speculative_accept_forward: CUDA path requires bonus_enabled on CUDA when provided");
    return CudaSpeculativeAcceptForward(
        target_probs,
        draft_probs,
        draft_token_ids,
        bonus_probs,
        bonus_enabled,
        method,
        posterior_threshold,
        posterior_alpha);
  }
#endif

  std::string method_name = method;
  std::transform(method_name.begin(), method_name.end(), method_name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (method_name == "probabilistic" || method_name == "rejection") {
    method_name = "rejection_sampler";
  }
  if (method_name.empty()) {
    method_name = "rejection_sampler";
  }

  auto probs_t = target_probs.to(torch::kFloat32);
  auto probs_q = draft_probs.to(torch::kFloat32);
  auto token_ids = draft_token_ids.to(torch::kLong);
  torch::Tensor bonus = torch::Tensor();
  if (bonus_probs.has_value() && bonus_probs.value().defined()) {
    bonus = bonus_probs.value().to(torch::kFloat32);
  }
  torch::Tensor bonus_mask = torch::Tensor();
  if (bonus_enabled.has_value() && bonus_enabled.value().defined()) {
    bonus_mask = bonus_enabled.value().to(torch::kBool).reshape({-1});
  }

  const auto batch_size = token_ids.size(0);
  const auto draft_len = token_ids.size(1);
  auto out_tokens = torch::full({batch_size, draft_len + 1}, -1,
                                token_ids.options().dtype(torch::kLong));
  auto out_lengths = torch::zeros({batch_size}, token_ids.options().dtype(torch::kLong));
  auto accepted_counts = torch::zeros({batch_size}, token_ids.options().dtype(torch::kLong));

  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t accepted = 0;
    std::vector<int64_t> emitted;
    emitted.reserve(static_cast<size_t>(draft_len + 1));
    for (int64_t i = 0; i < draft_len; ++i) {
      const int64_t candidate = token_ids.index({b, i}).item<int64_t>();
      auto p = probs_t.index({b, i});
      if (method_name == "strict") {
        const int64_t expected = std::get<1>(torch::max(p, -1, true)).item<int64_t>();
        if (candidate == expected) {
          emitted.push_back(candidate);
          accepted += 1;
          continue;
        }
        emitted.push_back(expected);
        break;
      }
      if (method_name == "typical_acceptance_sampler") {
        constexpr double kEpsilon = 1e-5;
        const double candidate_prob = p.index({candidate}).item<double>();
        const double posterior_entropy = -(p * torch::log(p + kEpsilon)).sum().item<double>();
        const double threshold = std::min(posterior_threshold, std::exp(-posterior_entropy) * posterior_alpha);
        if (candidate_prob > threshold) {
          emitted.push_back(candidate);
          accepted += 1;
          continue;
        }
        auto sample_probs = p / p.sum().clamp_min(1e-8);
        emitted.push_back(torch::multinomial(sample_probs, 1).item<int64_t>());
        break;
      }
      auto q = probs_q.index({b, i});
      const double qx = q.index({candidate}).item<double>();
      const double px = p.index({candidate}).item<double>();
      const double accept_prob = qx <= 1e-12 ? 1.0 : std::min(px / qx, 1.0);
      const double draw = torch::rand({}, torch::TensorOptions().dtype(torch::kFloat32).device(token_ids.device())).item<double>();
      if (draw <= accept_prob) {
        emitted.push_back(candidate);
        accepted += 1;
        continue;
      }
      auto residual = torch::clamp(p - q, 0.0);
      const double residual_mass = residual.sum().item<double>();
      torch::Tensor sample_probs;
      if (residual_mass <= 1e-8) {
        sample_probs = p / p.sum().clamp_min(1e-8);
      } else {
        sample_probs = residual / residual_mass;
      }
      const int64_t sampled = torch::multinomial(sample_probs, 1).item<int64_t>();
      emitted.push_back(sampled);
      break;
    }

    if (accepted == draft_len && bonus.defined()) {
      bool enabled = true;
      if (bonus_mask.defined()) {
        enabled = bonus_mask.index({b}).item<bool>();
      }
      if (enabled) {
        auto bonus_row = bonus.index({b});
        if (method_name == "strict") {
          emitted.push_back(std::get<1>(torch::max(bonus_row, -1, true)).item<int64_t>());
        } else {
          auto normalized = bonus_row / bonus_row.sum().clamp_min(1e-8);
          emitted.push_back(torch::multinomial(normalized, 1).item<int64_t>());
        }
      }
    }

    if (!emitted.empty()) {
      auto emitted_tensor = torch::tensor(emitted, token_ids.options().dtype(torch::kLong));
      out_tokens.index_put_({b, torch::indexing::Slice(0, emitted_tensor.size(0))}, emitted_tensor);
      out_lengths.index_put_({b}, static_cast<int64_t>(emitted_tensor.numel()));
    }
    accepted_counts.index_put_({b}, accepted);
  }

  return std::make_tuple(out_tokens, out_lengths, accepted_counts);
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

#if MODEL_STACK_WITH_CUDA
  if (beams.is_cuda() && logits.is_cuda() && raw_scores.is_cuda() && finished.is_cuda() && lengths.is_cuda() &&
      HasCudaSamplingKernel() && beam_size <= 32) {
    auto next_logits = logits.dim() == 3 ? logits.select(1, logits.size(1) - 1) : logits;
    auto next = CudaBeamSearchStepForward(
        beams,
        next_logits,
        raw_scores,
        finished,
        lengths,
        beam_size,
        eos_id,
        pad_id);
    return py::make_tuple(next[0], next[1], next[2], next[3], next[4]);
  }
#endif

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
  auto batch_offsets = torch::arange(batch_size, parent_beams.options()).unsqueeze(1) * beam_size;
  auto parent_rows = (batch_offsets + parent_beams).reshape({batch_size * beam_size});
  return py::make_tuple(next_beams, best_raw_scores, next_finished, next_lengths, parent_rows);
}

py::object IncrementalBeamSearchForward(
    const torch::Tensor& initial_beams,
    const torch::Tensor& initial_logits,
    int64_t beam_size,
    int64_t max_new_tokens,
    int64_t prompt_length,
    int64_t eos_id,
    int64_t pad_id,
    const py::function& advance_fn) {
  TORCH_CHECK(initial_beams.defined(), "incremental_beam_search_forward: initial_beams must be defined");
  TORCH_CHECK(initial_logits.defined(), "incremental_beam_search_forward: initial_logits must be defined");
  TORCH_CHECK(initial_beams.dim() == 2, "incremental_beam_search_forward: initial_beams must be rank-2 (B*beam, T)");
  TORCH_CHECK(beam_size > 0, "incremental_beam_search_forward: beam_size must be positive");
  TORCH_CHECK(max_new_tokens > 0, "incremental_beam_search_forward: max_new_tokens must be positive");
  TORCH_CHECK(prompt_length >= 0, "incremental_beam_search_forward: prompt_length must be non-negative");
  const auto total_rows = initial_beams.size(0);
  TORCH_CHECK(total_rows % beam_size == 0, "incremental_beam_search_forward: initial_beams batch mismatch");
  const auto batch_size = total_rows / beam_size;

  auto raw_scores = torch::full(
      {batch_size, beam_size},
      -std::numeric_limits<float>::infinity(),
      torch::TensorOptions().dtype(torch::kFloat32).device(initial_beams.device()));
  raw_scores.select(1, 0).zero_();
  auto finished = torch::zeros(
      {batch_size, beam_size},
      torch::TensorOptions().dtype(torch::kBool).device(initial_beams.device()));
  auto lengths = torch::full(
      {batch_size, beam_size},
      prompt_length,
      torch::TensorOptions().dtype(torch::kLong).device(initial_beams.device()));

  auto current = BeamSearchStepForward(initial_beams, initial_logits, raw_scores, finished, lengths, beam_size, eos_id, pad_id);
  auto beams = py::cast<torch::Tensor>(current[0]);
  raw_scores = py::cast<torch::Tensor>(current[1]);
  finished = py::cast<torch::Tensor>(current[2]);
  lengths = py::cast<torch::Tensor>(current[3]);
  auto parent_rows = py::cast<torch::Tensor>(current[4]);

  for (int64_t step = 1; step < max_new_tokens; ++step) {
    if (finished.all().item<bool>()) {
      break;
    }
    py::object next_logits_obj = advance_fn(parent_rows, beams);
    if (next_logits_obj.is_none()) {
      return py::none();
    }
    auto next_logits = py::cast<torch::Tensor>(next_logits_obj);
    current = BeamSearchStepForward(beams, next_logits, raw_scores, finished, lengths, beam_size, eos_id, pad_id);
    beams = py::cast<torch::Tensor>(current[0]);
    raw_scores = py::cast<torch::Tensor>(current[1]);
    finished = py::cast<torch::Tensor>(current[2]);
    lengths = py::cast<torch::Tensor>(current[3]);
    parent_rows = py::cast<torch::Tensor>(current[4]);
  }

  return py::make_tuple(beams, raw_scores, finished, lengths, parent_rows);
}


py::tuple AppendTokensForward(
    const torch::Tensor& seq,
    const torch::Tensor& next_id,
    const c10::optional<torch::Tensor>& attention_mask,
    const c10::optional<torch::Tensor>& row_ids) {
  TORCH_CHECK(seq.defined() && next_id.defined(), "append_tokens_forward: seq and next_id must be defined");
  TORCH_CHECK(seq.dim() == 2 && next_id.dim() == 2, "append_tokens_forward: seq and next_id must be rank-2");

  auto selected_seq = seq;
  c10::optional<torch::Tensor> selected_mask = attention_mask;
  if (row_ids.has_value() && row_ids.value().defined()) {
    auto ids = row_ids.value().to(torch::kLong).contiguous().view({-1});
    if (ids.numel() > 0) {
      const auto min_id = ids.min().item<int64_t>();
      const auto max_id = ids.max().item<int64_t>();
      TORCH_CHECK(min_id >= 0, "append_tokens_forward: row_ids must be non-negative");
      TORCH_CHECK(max_id < seq.size(0), "append_tokens_forward: row_ids exceed source batch size");
    }
    selected_seq = seq.index_select(0, ids);
    if (attention_mask.has_value() && attention_mask.value().defined()) {
      selected_mask = attention_mask.value().index_select(0, ids);
    }
  }

  TORCH_CHECK(selected_seq.size(0) == next_id.size(0), "append_tokens_forward: batch mismatch");
#if MODEL_STACK_WITH_CUDA
  if (selected_seq.is_cuda() && next_id.is_cuda() && HasCudaAppendTokensKernel()) {
    auto next = CudaAppendTokensForward(selected_seq, next_id, selected_mask);
    py::object next_mask = py::none();
    if (next.size() > 1 && next[1].defined()) {
      next_mask = py::cast(next[1]);
    }
    return py::make_tuple(next[0], next_mask);
  }
#endif
  auto next_seq = torch::cat({selected_seq, next_id}, 1);
  py::object next_mask = py::none();
  if (selected_mask.has_value() && selected_mask.value().defined()) {
    auto ones = torch::ones(
        {next_id.size(0), next_id.size(1)},
        torch::TensorOptions().dtype(selected_mask.value().scalar_type()).device(selected_mask.value().device()));
    next_mask = py::cast(torch::cat({selected_mask.value(), ones}, 1));
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
  const auto resolved_backend = ResolveLinearBackendForTensor(backend, x);
#if MODEL_STACK_WITH_CUDA
  if (resolved_backend == "cublaslt" && x.is_cuda()) {
    return CublasLtLinearForward(x, weight, bias);
  }
#endif
  return ReferenceLinearForward(x, weight, bias);
}

torch::Tensor Int4LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(x.defined() && packed_weight.defined() && inv_scale.defined(),
              "int4_linear_forward: x, packed_weight, and inv_scale must be defined");
  TORCH_CHECK(x.dim() >= 2, "int4_linear_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "int4_linear_forward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(packed_weight.dim() == 2, "int4_linear_forward: packed_weight must be rank-2");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
              "int4_linear_forward: packed_weight must use uint8 storage");
  TORCH_CHECK(inv_scale.dim() == 1, "int4_linear_forward: inv_scale must be rank-1");
  TORCH_CHECK(inv_scale.size(0) == packed_weight.size(0),
              "int4_linear_forward: inv_scale size mismatch");
  TORCH_CHECK(packed_weight.size(1) == (x.size(-1) + 1) / 2,
              "int4_linear_forward: packed_weight column count mismatch");
  if (bias.has_value() && bias.value().defined()) {
    const auto& b = bias.value();
    TORCH_CHECK(b.dim() == 1, "int4_linear_forward: bias must be rank-1");
    TORCH_CHECK(b.size(0) == packed_weight.size(0), "int4_linear_forward: bias size mismatch");
  }
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && packed_weight.is_cuda() && inv_scale.is_cuda() && HasCudaInt4LinearKernel()) {
    return CudaInt4LinearForward(x, packed_weight, inv_scale, bias);
  }
#endif
  return ReferenceInt4LinearForward(x, packed_weight, inv_scale, bias);
}

torch::Tensor Int4LinearGradInputForward(
    const torch::Tensor& grad_out,
    const torch::Tensor& packed_weight,
    const torch::Tensor& inv_scale,
    int64_t in_features) {
  TORCH_CHECK(grad_out.defined() && packed_weight.defined() && inv_scale.defined(),
              "int4_linear_grad_input_forward: grad_out, packed_weight, and inv_scale must be defined");
  TORCH_CHECK(grad_out.dim() >= 2, "int4_linear_grad_input_forward: grad_out must have rank >= 2");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 || grad_out.scalar_type() == torch::kFloat16 ||
                  grad_out.scalar_type() == torch::kBFloat16,
              "int4_linear_grad_input_forward: grad_out must use float32, float16, or bfloat16");
  TORCH_CHECK(packed_weight.dim() == 2, "int4_linear_grad_input_forward: packed_weight must be rank-2");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
              "int4_linear_grad_input_forward: packed_weight must use uint8 storage");
  TORCH_CHECK(inv_scale.dim() == 1, "int4_linear_grad_input_forward: inv_scale must be rank-1");
  TORCH_CHECK(inv_scale.size(0) == packed_weight.size(0),
              "int4_linear_grad_input_forward: inv_scale size mismatch");
  TORCH_CHECK(grad_out.size(-1) == packed_weight.size(0),
              "int4_linear_grad_input_forward: grad_out feature size mismatch");
  TORCH_CHECK(in_features > 0, "int4_linear_grad_input_forward: in_features must be positive");
  TORCH_CHECK(packed_weight.size(1) == (in_features + 1) / 2,
              "int4_linear_grad_input_forward: packed weight column count mismatch");
#if MODEL_STACK_WITH_CUDA
  if (grad_out.is_cuda() && packed_weight.is_cuda() && inv_scale.is_cuda() && HasCudaInt4LinearKernel()) {
    return CudaInt4LinearGradInputForward(grad_out, packed_weight, inv_scale, in_features);
  }
#endif
  auto weight = ReferenceInt4LinearForward(
      torch::eye(in_features, torch::TensorOptions().device(grad_out.device()).dtype(grad_out.scalar_type())),
      packed_weight,
      inv_scale,
      c10::nullopt);
  return ReferenceLinearForward(grad_out, weight, c10::nullopt);
}

torch::Tensor CutlassInt4Bf16LinearForwardChecked(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight_rowmajor,
    const torch::Tensor& scale,
    const c10::optional<torch::Tensor>& bias,
    bool packed_weight_is_shuffled) {
  TORCH_CHECK(x.defined() && packed_weight_rowmajor.defined() && scale.defined(),
              "cutlass_int4_bf16_linear_forward: x, packed_weight_rowmajor, and scale must be defined");
  TORCH_CHECK(x.dim() >= 2, "cutlass_int4_bf16_linear_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "cutlass_int4_bf16_linear_forward: x must be bfloat16");
  TORCH_CHECK(packed_weight_rowmajor.dim() == 2,
              "cutlass_int4_bf16_linear_forward: packed_weight_rowmajor must be rank-2");
  TORCH_CHECK(packed_weight_rowmajor.scalar_type() == torch::kUInt8,
              "cutlass_int4_bf16_linear_forward: packed_weight_rowmajor must use uint8 storage");
  TORCH_CHECK(scale.dim() == 1, "cutlass_int4_bf16_linear_forward: scale must be rank-1");
  TORCH_CHECK(packed_weight_rowmajor.size(0) == scale.size(0),
              "cutlass_int4_bf16_linear_forward: packed N dimension mismatch");
  TORCH_CHECK(packed_weight_rowmajor.size(1) == (x.size(-1) + 1) / 2,
              "cutlass_int4_bf16_linear_forward: packed K dimension mismatch");
  TORCH_CHECK(!bias.has_value() || !bias.value().defined(),
              "cutlass_int4_bf16_linear_forward: bias is not supported yet");
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && packed_weight_rowmajor.is_cuda() && scale.is_cuda()) {
    auto out = CutlassInt4Bf16LinearForward(
        x,
        packed_weight_rowmajor,
        scale,
        bias,
        packed_weight_is_shuffled);
    if (out.defined()) {
      return out;
    }
  }
#endif
  TORCH_CHECK(false, "cutlass_int4_bf16_linear_forward: CUTLASS SM90 backend unavailable or shape unsupported");
  return torch::Tensor();
}

torch::Tensor CutlassInt4PackShuffledForwardChecked(const torch::Tensor& qweight) {
  TORCH_CHECK(qweight.defined(), "cutlass_int4_pack_shuffled_forward: qweight must be defined");
  TORCH_CHECK(qweight.dim() == 2, "cutlass_int4_pack_shuffled_forward: qweight must be rank-2");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt8,
              "cutlass_int4_pack_shuffled_forward: qweight must use int8 storage");
  TORCH_CHECK((qweight.size(1) % 2) == 0,
              "cutlass_int4_pack_shuffled_forward: K must be even for packed int4");
#if MODEL_STACK_WITH_CUDA
  if (qweight.is_cuda()) {
    auto out = CutlassInt4PackShuffledForward(qweight);
    if (out.defined()) {
      return out;
    }
  }
#endif
  TORCH_CHECK(false, "cutlass_int4_pack_shuffled_forward: CUTLASS SM90 backend unavailable or shape unsupported");
  return torch::Tensor();
}

torch::Tensor Nf4LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(x.defined() && packed_weight.defined() && weight_scale.defined(),
              "nf4_linear_forward: x, packed_weight, and weight_scale must be defined");
  TORCH_CHECK(x.dim() >= 2, "nf4_linear_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "nf4_linear_forward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(packed_weight.dim() == 2, "nf4_linear_forward: packed_weight must be rank-2");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
              "nf4_linear_forward: packed_weight must use uint8 storage");
  TORCH_CHECK(weight_scale.dim() == 1, "nf4_linear_forward: weight_scale must be rank-1");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32,
              "nf4_linear_forward: weight_scale must use float32 storage");
  TORCH_CHECK(weight_scale.size(0) == packed_weight.size(0),
              "nf4_linear_forward: weight_scale size mismatch");
  TORCH_CHECK(packed_weight.size(1) == (x.size(-1) + 1) / 2,
              "nf4_linear_forward: packed_weight column count mismatch");
  if (bias.has_value() && bias.value().defined()) {
    const auto& b = bias.value();
    TORCH_CHECK(b.dim() == 1, "nf4_linear_forward: bias must be rank-1");
    TORCH_CHECK(b.size(0) == packed_weight.size(0), "nf4_linear_forward: bias size mismatch");
  }
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && packed_weight.is_cuda() && weight_scale.is_cuda() && HasCudaNf4LinearKernel()) {
    return CudaNf4LinearForward(x, packed_weight, weight_scale, bias);
  }
#endif
  return ReferenceNf4LinearForward(x, packed_weight, weight_scale, bias);
}

torch::Tensor Fp8LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    double weight_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined() && weight_fp8.defined(), "fp8_linear_forward: x and weight_fp8 must be defined");
  TORCH_CHECK(x.dim() >= 2, "fp8_linear_forward: x must have rank >= 2");
  TORCH_CHECK(weight_fp8.dim() == 2, "fp8_linear_forward: weight_fp8 must be rank-2");
  TORCH_CHECK(x.size(-1) == weight_fp8.size(1), "fp8_linear_forward: input feature size mismatch");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "fp8_linear_forward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(weight_fp8.scalar_type() == torch::kFloat32 || weight_fp8.scalar_type() == torch::kFloat16 ||
                  weight_fp8.scalar_type() == torch::kBFloat16,
              "fp8_linear_forward: weight_fp8 must use float32, float16, or bfloat16");
  if (bias.has_value() && bias.value().defined()) {
    const auto& b = bias.value();
    TORCH_CHECK(b.dim() == 1, "fp8_linear_forward: bias must be rank-1");
    TORCH_CHECK(b.size(0) == weight_fp8.size(0), "fp8_linear_forward: bias size mismatch");
  }
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && weight_fp8.is_cuda() && HasCudaFp8LinearKernel()) {
    return CudaFp8LinearForward(x, weight_fp8, weight_scale, bias, out_dtype);
  }
#endif
  return ReferenceFp8LinearForward(x, weight_fp8, weight_scale, bias, out_dtype);
}

std::vector<torch::Tensor> Int8QuantizeActivationForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& provided_scale) {
  TORCH_CHECK(x.defined(), "int8_quantize_activation_forward: x must be defined");
  TORCH_CHECK(x.dim() >= 2, "int8_quantize_activation_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "int8_quantize_activation_forward: x must use float32, float16, or bfloat16");
  const auto rows = x.numel() / x.size(-1);
  if (provided_scale.has_value() && provided_scale.value().defined()) {
    const auto& s = provided_scale.value();
    TORCH_CHECK(s.scalar_type() == torch::kFloat32,
                "int8_quantize_activation_forward: provided_scale must use float32 storage");
    TORCH_CHECK(s.numel() == 1 || s.numel() == rows,
                "int8_quantize_activation_forward: provided_scale must have 1 or rows elements");
  }
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && HasCudaInt8QuantFrontendKernel()) {
    return CudaInt8QuantizeActivationForward(x, provided_scale);
  }
#endif
  auto quantized = ReferenceQuantizeActivationInt8Rowwise(x, provided_scale);
  return {quantized.first, quantized.second};
}

std::vector<torch::Tensor> Int8QuantizeActivationTransposeForward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& provided_scale) {
  TORCH_CHECK(x.defined(), "int8_quantize_activation_transpose_forward: x must be defined");
  TORCH_CHECK(x.dim() >= 2, "int8_quantize_activation_transpose_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "int8_quantize_activation_transpose_forward: x must use float32, float16, or bfloat16");
  const auto cols = x.size(-1);
  if (provided_scale.has_value() && provided_scale.value().defined()) {
    const auto& s = provided_scale.value();
    TORCH_CHECK(s.scalar_type() == torch::kFloat32,
                "int8_quantize_activation_transpose_forward: provided_scale must use float32 storage");
    TORCH_CHECK(s.numel() == cols,
                "int8_quantize_activation_transpose_forward: provided_scale must have cols elements");
  }
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && HasCudaInt8QuantFrontendKernel()) {
    return CudaInt8QuantizeActivationTransposeForward(x, provided_scale);
  }
#endif
  auto x_t = x.reshape({x.numel() / x.size(-1), x.size(-1)}).transpose(0, 1).contiguous();
  auto quantized = ReferenceQuantizeActivationInt8Rowwise(x_t, provided_scale);
  return {quantized.first, quantized.second};
}

std::vector<torch::Tensor> Int8QuantizeRelu2ActivationForward(const torch::Tensor& x, int64_t act_quant_bits) {
  TORCH_CHECK(x.defined(), "int8_quantize_relu2_activation_forward: x must be defined");
  TORCH_CHECK(x.dim() >= 2, "int8_quantize_relu2_activation_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "int8_quantize_relu2_activation_forward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(act_quant_bits >= 2 && act_quant_bits <= 8,
              "int8_quantize_relu2_activation_forward: act_quant_bits must be in [2, 8]");
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && HasCudaInt8QuantFrontendKernel()) {
    return CudaInt8QuantizeRelu2ActivationForward(x, act_quant_bits);
  }
#endif
  auto y = at::relu(x);
  y = y * y;
  auto quantized = ReferenceQuantizeActivationInt8Rowwise(y, c10::nullopt, static_cast<int>(act_quant_bits));
  return {quantized.first, quantized.second};
}

std::vector<torch::Tensor> Int8QuantizeLeakyReluHalf2ActivationForward(const torch::Tensor& x, int64_t act_quant_bits) {
  TORCH_CHECK(x.defined(), "int8_quantize_leaky_relu_half2_activation_forward: x must be defined");
  TORCH_CHECK(x.dim() >= 2, "int8_quantize_leaky_relu_half2_activation_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "int8_quantize_leaky_relu_half2_activation_forward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(act_quant_bits >= 2 && act_quant_bits <= 8,
              "int8_quantize_leaky_relu_half2_activation_forward: act_quant_bits must be in [2, 8]");
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && HasCudaInt8QuantFrontendKernel()) {
    return CudaInt8QuantizeLeakyReluHalf2ActivationForward(x, act_quant_bits);
  }
#endif
  auto y = at::leaky_relu(x, 0.5);
  y = y * y;
  auto quantized = ReferenceQuantizeActivationInt8Rowwise(y, c10::nullopt, static_cast<int>(act_quant_bits));
  return {quantized.first, quantized.second};
}

torch::Tensor Int8LinearForward(
    const torch::Tensor& qx,
    const torch::Tensor& x_scale,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(qx.defined() && x_scale.defined() && qweight.defined() && inv_scale.defined(),
              "int8_linear_forward: qx, x_scale, qweight, and inv_scale must be defined");
  TORCH_CHECK(qx.dim() >= 2, "int8_linear_forward: qx must have rank >= 2");
  TORCH_CHECK(qx.scalar_type() == torch::kInt8, "int8_linear_forward: qx must use int8 storage");
  TORCH_CHECK(x_scale.dim() == 1, "int8_linear_forward: x_scale must be rank-1");
  TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32, "int8_linear_forward: x_scale must use float32 storage");
  TORCH_CHECK(qweight.dim() == 2, "int8_linear_forward: qweight must be rank-2");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt8, "int8_linear_forward: qweight must use int8 storage");
  TORCH_CHECK(inv_scale.dim() == 1, "int8_linear_forward: inv_scale must be rank-1");
  TORCH_CHECK(inv_scale.scalar_type() == torch::kFloat32, "int8_linear_forward: inv_scale must use float32 storage");
  const auto in_features = qx.size(-1);
  const auto rows = qx.numel() / in_features;
  TORCH_CHECK(x_scale.size(0) == rows, "int8_linear_forward: x_scale size mismatch");
  TORCH_CHECK(qweight.size(1) == in_features, "int8_linear_forward: qweight column count mismatch");
  TORCH_CHECK(inv_scale.size(0) == qweight.size(0), "int8_linear_forward: inv_scale size mismatch");
  if (bias.has_value() && bias.value().defined()) {
    const auto& b = bias.value();
    TORCH_CHECK(b.dim() == 1, "int8_linear_forward: bias must be rank-1");
    TORCH_CHECK(b.size(0) == qweight.size(0), "int8_linear_forward: bias size mismatch");
  }
#if MODEL_STACK_WITH_CUDA
  if (qx.is_cuda() && x_scale.is_cuda() && qweight.is_cuda() && inv_scale.is_cuda() && HasCudaInt8LinearKernel()) {
    return CudaInt8LinearForward(qx, x_scale, qweight, inv_scale, bias, out_dtype);
  }
#endif
  return ReferenceInt8LinearForward(qx, x_scale, qweight, inv_scale, bias, out_dtype);
}

torch::Tensor Int8LinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& provided_scale,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined() && qweight.defined() && inv_scale.defined(),
              "int8_linear_from_float_forward: x, qweight, and inv_scale must be defined");
  TORCH_CHECK(x.dim() >= 2, "int8_linear_from_float_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "int8_linear_from_float_forward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(qweight.dim() == 2, "int8_linear_from_float_forward: qweight must be rank-2");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt8,
              "int8_linear_from_float_forward: qweight must use int8 storage");
  TORCH_CHECK(inv_scale.dim() == 1, "int8_linear_from_float_forward: inv_scale must be rank-1");
  TORCH_CHECK(inv_scale.scalar_type() == torch::kFloat32,
              "int8_linear_from_float_forward: inv_scale must use float32 storage");
  TORCH_CHECK(inv_scale.size(0) == qweight.size(0),
              "int8_linear_from_float_forward: inv_scale size mismatch");
  TORCH_CHECK(qweight.size(1) == x.size(-1), "int8_linear_from_float_forward: qweight column count mismatch");
  if (provided_scale.has_value() && provided_scale.value().defined()) {
    const auto& s = provided_scale.value();
    TORCH_CHECK(s.scalar_type() == torch::kFloat32,
                "int8_linear_from_float_forward: provided_scale must use float32 storage");
    TORCH_CHECK(s.numel() == 1 || s.numel() == (x.numel() / x.size(-1)),
                "int8_linear_from_float_forward: provided_scale must have 1 or rows elements");
  }
  if (bias.has_value() && bias.value().defined()) {
    const auto& b = bias.value();
    TORCH_CHECK(b.dim() == 1, "int8_linear_from_float_forward: bias must be rank-1");
    TORCH_CHECK(b.size(0) == qweight.size(0), "int8_linear_from_float_forward: bias size mismatch");
  }
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && qweight.is_cuda() && inv_scale.is_cuda() && HasCudaInt8QuantFrontendKernel()) {
    return CudaInt8LinearFromFloatForward(x, qweight, inv_scale, bias, provided_scale, out_dtype);
  }
#endif
  return ReferenceInt8LinearFromFloatForward(x, qweight, inv_scale, bias, provided_scale, out_dtype);
}

torch::Tensor Int8LinearGradWeightFromFloatForward(
    const torch::Tensor& grad_out,
    const torch::Tensor& x,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(grad_out.defined() && x.defined(),
              "int8_linear_grad_weight_from_float_forward: grad_out and x must be defined");
  TORCH_CHECK(grad_out.dim() >= 2 && x.dim() >= 2,
              "int8_linear_grad_weight_from_float_forward: grad_out and x must have rank >= 2");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 || grad_out.scalar_type() == torch::kFloat16 ||
                  grad_out.scalar_type() == torch::kBFloat16,
              "int8_linear_grad_weight_from_float_forward: grad_out must use float32, float16, or bfloat16");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "int8_linear_grad_weight_from_float_forward: x must use float32, float16, or bfloat16");
  const auto x_cols = x.size(-1);
  const auto go_cols = grad_out.size(-1);
  const auto rows = x.numel() / x_cols;
  TORCH_CHECK(grad_out.numel() / go_cols == rows,
              "int8_linear_grad_weight_from_float_forward: grad_out and x row counts must match");
  auto qx_t = Int8QuantizeActivationTransposeForward(x, c10::nullopt);
  auto qgo_t = Int8QuantizeActivationTransposeForward(grad_out, c10::nullopt);
  return Int8LinearForward(
      qgo_t[0],
      qgo_t[1],
      qx_t[0],
      qx_t[1],
      c10::nullopt,
      out_dtype);
}

torch::Tensor Int8AttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& q_scale,
    const torch::Tensor& k,
    const torch::Tensor& k_scale,
    const torch::Tensor& v,
    const torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(q.defined() && q_scale.defined() && k.defined() && k_scale.defined() && v.defined() && v_scale.defined(),
              "int8_attention_forward: q, q_scale, k, k_scale, v, and v_scale must be defined");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "int8_attention_forward: q, k, and v must be rank-4");
  TORCH_CHECK(q.scalar_type() == torch::kInt8 && k.scalar_type() == torch::kInt8 && v.scalar_type() == torch::kInt8,
              "int8_attention_forward: q, k, and v must use int8 storage");
  TORCH_CHECK(q_scale.dim() == 1 && k_scale.dim() == 1 && v_scale.dim() == 1,
              "int8_attention_forward: q_scale, k_scale, and v_scale must be rank-1");
  TORCH_CHECK(q_scale.scalar_type() == torch::kFloat32 && k_scale.scalar_type() == torch::kFloat32 &&
                  v_scale.scalar_type() == torch::kFloat32,
              "int8_attention_forward: q_scale, k_scale, and v_scale must use float32 storage");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "int8_attention_forward: batch size mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "int8_attention_forward: kv head mismatch");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "int8_attention_forward: q heads must be a multiple of kv heads");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "int8_attention_forward: head_dim mismatch");
  TORCH_CHECK(q_scale.numel() == q.size(0) * q.size(1) * q.size(2), "int8_attention_forward: q_scale size mismatch");
  TORCH_CHECK(k_scale.numel() == k.size(0) * k.size(1) * k.size(2), "int8_attention_forward: k_scale size mismatch");
  TORCH_CHECK(v_scale.numel() == v.size(0) * v.size(1) * v.size(2), "int8_attention_forward: v_scale size mismatch");
#if MODEL_STACK_WITH_CUDA
  if (CanUseCudaInt8AttentionPath(q, q_scale, k, k_scale, v, v_scale, attn_mask)) {
    return CudaInt8AttentionForward(q, q_scale, k, k_scale, v, v_scale, attn_mask, is_causal, scale, out_dtype);
  }
#endif
  return ReferenceInt8AttentionForward(q, q_scale, k, k_scale, v, v_scale, attn_mask, is_causal, scale, out_dtype);
}

torch::Tensor Int8AttentionFromFloatForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    const c10::optional<torch::ScalarType>& out_dtype,
    const c10::optional<torch::Tensor>& q_provided_scale,
    const c10::optional<torch::Tensor>& k_provided_scale,
    const c10::optional<torch::Tensor>& v_provided_scale) {
  TORCH_CHECK(q.defined() && k.defined() && v.defined(),
              "int8_attention_from_float_forward: q, k, and v must be defined");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "int8_attention_from_float_forward: q, k, and v must be rank-4");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16 ||
                  q.scalar_type() == torch::kBFloat16,
              "int8_attention_from_float_forward: q must use float32, float16, or bfloat16");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "int8_attention_from_float_forward: q, k, and v must share dtype");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "int8_attention_from_float_forward: batch size mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "int8_attention_from_float_forward: kv head mismatch");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "int8_attention_from_float_forward: q heads must be a multiple of kv heads");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
              "int8_attention_from_float_forward: head_dim mismatch");
#if MODEL_STACK_WITH_CUDA
  if (q.is_cuda() && k.is_cuda() && v.is_cuda() && HasCudaInt8QuantFrontendKernel()) {
    return CudaInt8AttentionFromFloatForward(
        q,
        k,
        v,
        attn_mask,
        is_causal,
        scale,
        out_dtype,
        q_provided_scale,
        k_provided_scale,
        v_provided_scale);
  }
#endif
  return ReferenceInt8AttentionFromFloatForward(
      q,
      k,
      v,
      attn_mask,
      is_causal,
      scale,
      out_dtype,
      q_provided_scale,
      k_provided_scale,
      v_provided_scale);
}

torch::Tensor BitNetLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::ScalarType>& out_dtype,
    bool debug_dense_fallback) {
  TORCH_CHECK(x.defined() && packed_weight.defined() && scale_values.defined() && layout_header.defined() &&
                  segment_offsets.defined(),
              "bitnet_linear_forward: x, packed_weight, scale_values, layout_header, and segment_offsets must be defined");
  TORCH_CHECK(x.dim() >= 2, "bitnet_linear_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "bitnet_linear_forward: x must use float32, float16, or bfloat16");
#if MODEL_STACK_WITH_CUDA
  if (!debug_dense_fallback &&
      x.is_cuda() && packed_weight.is_cuda() && scale_values.is_cuda() && layout_header.is_cuda() &&
      segment_offsets.is_cuda() && t10::bitnet::HasCudaBitNetLinearKernel()) {
    return t10::bitnet::CudaBitNetLinearForward(
        x,
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias,
        out_dtype);
  }
#endif
  auto out = ReferenceBitNetLinearForward(x, packed_weight, scale_values, layout_header, segment_offsets, bias);
  if (out_dtype.has_value()) {
    out = out.to(out_dtype.value());
  }
  return out;
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
  const auto resolved_backend = ResolveLinearBackendForTensor(backend, x);
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
  const auto resolved_backend = ResolveLinearBackendForTensor(backend, x);
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

py::tuple PackBitNetWeightForward(
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& scale_values,
    const c10::optional<torch::Tensor>& layout_header,
    const c10::optional<torch::Tensor>& segment_offsets) {
  TORCH_CHECK(weight.defined(), "pack_bitnet_weight_forward: weight must be defined");
  TORCH_CHECK(weight.dim() == 2, "pack_bitnet_weight_forward: weight must be rank-2");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32 || weight.scalar_type() == torch::kFloat16 ||
                  weight.scalar_type() == torch::kBFloat16,
              "pack_bitnet_weight_forward: weight must use float32, float16, or bfloat16");
  TORCH_CHECK(!scale_values.has_value() && !layout_header.has_value() && !segment_offsets.has_value(),
              "pack_bitnet_weight_forward: explicit metadata overrides are not supported yet");
#if MODEL_STACK_WITH_CUDA
  if (weight.is_cuda() && t10::bitnet::HasCudaBitNetLinearKernel()) {
    auto packed = t10::bitnet::CudaPackBitNetWeightForward(weight);
    return py::make_tuple(
        std::get<0>(packed),
        std::get<1>(packed),
        std::get<2>(packed),
        std::get<3>(packed));
  }
#endif

  const auto logical_out = weight.size(0);
  const auto logical_in = weight.size(1);
  const auto padded_out = ((logical_out + 15) / 16) * 16;
  const auto padded_in = ((logical_in + 31) / 32) * 32;

  auto weight_cpu = weight.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
  const auto max_abs = weight_cpu.abs().amax().item<float>();
  const float scale = std::max(max_abs, 1.0e-8f);

  auto packed_cpu = torch::full(
      {padded_out, (padded_in + 3) / 4},
      static_cast<uint8_t>(0x55),
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
  auto weight_acc = weight_cpu.accessor<float, 2>();
  auto packed_acc = packed_cpu.accessor<uint8_t, 2>();
  for (int64_t out_idx = 0; out_idx < logical_out; ++out_idx) {
    for (int64_t in_idx = 0; in_idx < logical_in; ++in_idx) {
      const auto normalized = weight_acc[out_idx][in_idx] / scale;
      const auto rounded = static_cast<int>(std::llround(normalized));
      const auto clamped = std::max(-1, std::min(1, rounded));
      const auto code = static_cast<uint8_t>(clamped + 1);
      auto& packed_value = packed_acc[out_idx][in_idx / 4];
      const auto shift = static_cast<int>((in_idx % 4) * 2);
      packed_value = static_cast<uint8_t>((packed_value & ~(0x03 << shift)) | (code << shift));
    }
  }

  auto packed_scale_values = torch::tensor({scale}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  auto packed_layout_header = torch::zeros(
      {t10::bitnet::kLayoutHeaderLen},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
  auto header_acc = packed_layout_header.accessor<int32_t, 1>();
  header_acc[t10::bitnet::kIdxFormatVersion] = 1;
  header_acc[t10::bitnet::kIdxTileN] = 16;
  header_acc[t10::bitnet::kIdxTileK] = 32;
  header_acc[t10::bitnet::kIdxLogicalOut] = static_cast<int32_t>(logical_out);
  header_acc[t10::bitnet::kIdxLogicalIn] = static_cast<int32_t>(logical_in);
  header_acc[t10::bitnet::kIdxPaddedOut] = static_cast<int32_t>(padded_out);
  header_acc[t10::bitnet::kIdxPaddedIn] = static_cast<int32_t>(padded_in);
  header_acc[t10::bitnet::kIdxScaleGranularity] = 0;
  header_acc[t10::bitnet::kIdxScaleGroupSize] = static_cast<int32_t>(logical_out);
  header_acc[t10::bitnet::kIdxInterleaveMode] = 1;
  header_acc[t10::bitnet::kIdxArchMin] = 80;
  header_acc[t10::bitnet::kIdxSegmentCount] = 1;
  header_acc[t10::bitnet::kIdxFlags] = 0;
  auto packed_segment_offsets = torch::zeros(
      {2},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
  auto offsets_acc = packed_segment_offsets.accessor<int32_t, 1>();
  offsets_acc[0] = 0;
  offsets_acc[1] = static_cast<int32_t>(logical_out);

  return py::make_tuple(
      packed_cpu.to(weight.device()),
      packed_scale_values.to(weight.device()),
      packed_layout_header.to(weight.device()),
      packed_segment_offsets.to(weight.device()));
}

py::tuple BitNetRuntimeRowQuantizeForward(
    const torch::Tensor& weight,
    double eps) {
  TORCH_CHECK(weight.defined(), "bitnet_runtime_row_quantize_forward: weight must be defined");
  TORCH_CHECK(weight.dim() == 2, "bitnet_runtime_row_quantize_forward: weight must be rank-2");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32 || weight.scalar_type() == torch::kFloat16 ||
                  weight.scalar_type() == torch::kBFloat16,
              "bitnet_runtime_row_quantize_forward: weight must use float32, float16, or bfloat16");
#if MODEL_STACK_WITH_CUDA
  if (weight.is_cuda() && t10::bitnet::HasCudaBitNetLinearKernel()) {
    auto quantized = t10::bitnet::CudaBitNetRuntimeRowQuantizeForward(weight, eps);
    return py::make_tuple(std::get<0>(quantized), std::get<1>(quantized));
  }
#endif
  TORCH_CHECK(false, "bitnet_runtime_row_quantize_forward: CUDA native backend is required");
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

std::vector<torch::Tensor> BitNetQkvPackedHeadsProjectionForward(
    const torch::Tensor& q_x,
    const torch::Tensor& q_packed_weight,
    const torch::Tensor& q_scale_values,
    const torch::Tensor& q_layout_header,
    const torch::Tensor& q_segment_offsets,
    const c10::optional<torch::Tensor>& q_bias,
    const torch::Tensor& k_x,
    const torch::Tensor& k_packed_weight,
    const torch::Tensor& k_scale_values,
    const torch::Tensor& k_layout_header,
    const torch::Tensor& k_segment_offsets,
    const c10::optional<torch::Tensor>& k_bias,
    const torch::Tensor& v_x,
    const torch::Tensor& v_packed_weight,
    const torch::Tensor& v_scale_values,
    const torch::Tensor& v_layout_header,
    const torch::Tensor& v_segment_offsets,
    const c10::optional<torch::Tensor>& v_bias,
    int64_t q_heads,
    int64_t kv_heads,
    const c10::optional<torch::ScalarType>& out_dtype) {
  auto q = BitNetLinearForward(
      q_x,
      q_packed_weight,
      q_scale_values,
      q_layout_header,
      q_segment_offsets,
      q_bias,
      out_dtype,
      false);
  auto k = BitNetLinearForward(
      k_x,
      k_packed_weight,
      k_scale_values,
      k_layout_header,
      k_segment_offsets,
      k_bias,
      out_dtype,
      false);
  auto v = BitNetLinearForward(
      v_x,
      v_packed_weight,
      v_scale_values,
      v_layout_header,
      v_segment_offsets,
      v_bias,
      out_dtype,
      false);
  return {
      SplitHeadsForward(q, q_heads),
      SplitHeadsForward(k, kv_heads),
      SplitHeadsForward(v, kv_heads),
  };
}

std::vector<torch::Tensor> BitNetFusedQkvPackedHeadsProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& packed_bias,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads) {
  TORCH_CHECK(x.defined() && packed_weight.defined() && scale_values.defined() && layout_header.defined() &&
                  segment_offsets.defined(),
              "bitnet_fused_qkv_packed_heads_projection_forward: tensors must be defined");
  TORCH_CHECK(x.dim() == 3, "bitnet_fused_qkv_packed_heads_projection_forward: x must have shape (B, T, D)");
  TORCH_CHECK(q_size > 0 && k_size > 0 && v_size > 0,
              "bitnet_fused_qkv_packed_heads_projection_forward: q/k/v sizes must be positive");
  TORCH_CHECK(q_heads > 0 && kv_heads > 0,
              "bitnet_fused_qkv_packed_heads_projection_forward: q_heads and kv_heads must be positive");
  TORCH_CHECK(q_size + k_size + v_size == layout_header.to(torch::kInt32).contiguous()[3].item<int32_t>(),
              "bitnet_fused_qkv_packed_heads_projection_forward: fused output width mismatch");
  if (packed_bias.has_value() && packed_bias.value().defined()) {
    TORCH_CHECK(packed_bias.value().dim() == 1,
                "bitnet_fused_qkv_packed_heads_projection_forward: packed_bias must be rank-1");
    TORCH_CHECK(packed_bias.value().size(0) == q_size + k_size + v_size,
                "bitnet_fused_qkv_packed_heads_projection_forward: packed_bias size mismatch");
  }
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && packed_weight.is_cuda() && scale_values.is_cuda() && layout_header.is_cuda() &&
      segment_offsets.is_cuda() && t10::bitnet::HasCudaBitNetFusedQkvKernel()) {
    return t10::bitnet::CudaBitNetFusedQkvPackedHeadsProjectionForward(
        x,
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        packed_bias,
        q_size,
        k_size,
        v_size,
        q_heads,
        kv_heads);
  }
#endif
  auto fused = BitNetLinearForward(
      x,
      packed_weight,
      scale_values,
      layout_header,
      segment_offsets,
      packed_bias,
      c10::nullopt,
      false);
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

torch::Tensor EmbeddingForwardUnchecked(
    const torch::Tensor& weight,
    const torch::Tensor& indices,
    int64_t padding_idx) {
  TORCH_CHECK(weight.defined() && indices.defined(), "embedding_forward_unchecked: weight and indices must be defined");
  TORCH_CHECK(weight.dim() == 2, "embedding_forward_unchecked: weight must be rank-2");
  TORCH_CHECK(indices.scalar_type() == torch::kLong || indices.scalar_type() == torch::kInt,
              "embedding_forward_unchecked: indices must be int32 or int64");
#if MODEL_STACK_WITH_CUDA
  if (weight.is_cuda() && indices.is_cuda() && HasCudaEmbeddingKernel()) {
    return CudaEmbeddingForwardUnchecked(weight, indices);
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
  auto x_contig = x.contiguous();
  if (x_contig.size(1) == 1) {
    return x_contig.view({x.size(0), num_heads, 1, head_dim});
  }
  return x_contig.view({x.size(0), x.size(1), num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous();
}

torch::Tensor MergeHeadsForward(const torch::Tensor& x) {
  TORCH_CHECK(x.defined(), "merge_heads_forward: x must be defined");
  TORCH_CHECK(x.dim() == 4, "merge_heads_forward: x must have shape (B, H, T, Dh)");
  auto x_contig = x.contiguous();
  if (x_contig.size(2) == 1) {
    return x_contig.view({x.size(0), 1, x.size(1) * x.size(3)});
  }
  return x_contig.permute({0, 2, 1, 3}).contiguous().view({x.size(0), x.size(2), x.size(1) * x.size(3)});
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

std::string PyTypeName(const py::handle& obj) {
  return std::string(py::str(obj.get_type().attr("__name__")));
}

bool PyCallableAttr(const py::object& obj, const char* name) {
  if (!py::hasattr(obj, name)) {
    return false;
  }
  try {
    return PyCallable_Check(obj.attr(name).ptr()) != 0;
  } catch (...) {
    return false;
  }
}

py::object PyAttrOrNone(const py::object& obj, const char* name) {
  if (!py::hasattr(obj, name)) {
    return py::none();
  }
  return obj.attr(name);
}

bool PyBoolAttr(const py::object& obj, const char* name, bool default_value = false) {
  if (!py::hasattr(obj, name)) {
    return default_value;
  }
  try {
    return py::cast<bool>(obj.attr(name));
  } catch (...) {
    return default_value;
  }
}

double PyFloatAttr(const py::object& obj, const char* name, double default_value = 0.0) {
  if (!py::hasattr(obj, name)) {
    return default_value;
  }
  try {
    return py::cast<double>(obj.attr(name));
  } catch (...) {
    return default_value;
  }
}

c10::optional<double> PyFloatAttrOptional(const py::object& obj, const char* name) {
  auto value = PyAttrOrNone(obj, name);
  if (value.is_none()) {
    return c10::nullopt;
  }
  try {
    return py::cast<double>(value);
  } catch (...) {
    return c10::nullopt;
  }
}

int64_t PyIntAttr(const py::object& obj, const char* name, int64_t default_value = 0) {
  if (!py::hasattr(obj, name)) {
    return default_value;
  }
  try {
    return py::cast<int64_t>(obj.attr(name));
  } catch (...) {
    return default_value;
  }
}

c10::optional<int64_t> PyIntAttrOptional(const py::object& obj, const char* name) {
  auto value = PyAttrOrNone(obj, name);
  if (value.is_none()) {
    return c10::nullopt;
  }
  try {
    return py::cast<int64_t>(value);
  } catch (...) {
    return c10::nullopt;
  }
}

std::string PyStringAttr(const py::object& obj, const char* name, const std::string& default_value = "") {
  if (!py::hasattr(obj, name)) {
    return default_value;
  }
  try {
    return py::cast<std::string>(obj.attr(name));
  } catch (...) {
    return default_value;
  }
}

torch::Tensor TensorAttr(const py::object& obj, const char* name) {
  return py::cast<torch::Tensor>(obj.attr(name));
}

c10::optional<torch::Tensor> TensorAttrOptional(const py::object& obj, const char* name) {
  return OptionalTensorFromPyObject(PyAttrOrNone(obj, name));
}

py::module_ RuntimeOpsModule() {
  return py::module_::import("runtime.ops");
}

struct BitNetModuleState {
  torch::Tensor packed_weight;
  torch::Tensor scale_values;
  torch::Tensor layout_header;
  torch::Tensor segment_offsets;
  torch::Tensor compute_packed_words;
  torch::Tensor compute_row_scales;
  int64_t compute_tile_n = 0;
  torch::Tensor decode_nz_masks;
  torch::Tensor decode_sign_masks;
  torch::Tensor decode_row_scales;
  int64_t decode_tile_n = 0;
  torch::Tensor qweight;
  torch::Tensor inv_scale;
  c10::optional<torch::Tensor> bias = c10::nullopt;
  bool spin_enabled = false;
  torch::Tensor spin_signs;
  torch::Tensor pre_scale;
  std::string act_quant_mode = "none";
  std::string act_quant_method = "absmax";
  double act_quant_percentile = 0.999;
  int64_t act_quant_bits = 8;
  torch::Tensor act_scale;
};

bool IsSupportedLinearLikeModule(const py::object& module) {
  return py::hasattr(module, "weight") || PyCallableAttr(module, "runtime_linear");
}

bool ModuleSupportsPackedBackend(const py::object& module, const std::string& backend) {
  if (!PyCallableAttr(module, "runtime_supports_packed_backend")) {
    return false;
  }
  try {
    return py::cast<bool>(module.attr("runtime_supports_packed_backend")(backend));
  } catch (...) {
    return false;
  }
}

bool ModuleHasRuntimeLinear(const py::object& module) {
  return PyCallableAttr(module, "runtime_linear");
}

bool TryLoadBitNetModuleState(const py::object& module, BitNetModuleState* state) {
  if (state == nullptr) {
    return false;
  }
  if (!py::hasattr(module, "packed_weight") || !py::hasattr(module, "scale_values") ||
      !py::hasattr(module, "layout_header") || !py::hasattr(module, "segment_offsets")) {
    return false;
  }
  try {
    BitNetModuleState out;
    out.packed_weight = TensorAttr(module, "packed_weight");
    out.scale_values = TensorAttr(module, "scale_values");
    out.layout_header = TensorAttr(module, "layout_header");
    out.segment_offsets = TensorAttr(module, "segment_offsets");
    out.bias = TensorAttrOptional(module, "bias");
    if (PyCallableAttr(module, "_spin_enabled_runtime")) {
      out.spin_enabled = py::cast<bool>(module.attr("_spin_enabled_runtime")());
    } else {
      auto spin_enabled_flag = TensorAttrOptional(module, "spin_enabled_flag");
      if (spin_enabled_flag.has_value() && spin_enabled_flag.value().defined() && spin_enabled_flag.value().numel() > 0) {
        out.spin_enabled = spin_enabled_flag.value().item<int64_t>() != 0;
      }
    }
    if (out.spin_enabled && py::hasattr(module, "spin_signs")) {
      out.spin_signs = TensorAttr(module, "spin_signs");
    }
    bool pre_scale_active = true;
    if (PyCallableAttr(module, "_pre_scale_active_runtime")) {
      pre_scale_active = py::cast<bool>(module.attr("_pre_scale_active_runtime")());
    }
    if (pre_scale_active && py::hasattr(module, "pre_scale")) {
      out.pre_scale = TensorAttr(module, "pre_scale");
    }
    out.act_quant_mode = PyStringAttr(module, "act_quant_mode", "none");
    out.act_quant_method = PyStringAttr(module, "act_quant_method", "absmax");
    out.act_quant_percentile = PyFloatAttr(module, "act_quant_percentile", 0.999);
    out.act_quant_bits = PyIntAttr(module, "act_quant_bits", 8);
    if (py::hasattr(module, "act_scale")) {
      out.act_scale = TensorAttr(module, "act_scale");
    }
    const bool use_compute_packed =
        !out.spin_enabled && !pre_scale_active && NormalizeBackendName(out.act_quant_mode) == "none";
    if (use_compute_packed) {
      auto cached_compute_words = PyAttrOrNone(module, "_cached_compute_backend_words");
      auto cached_compute_row_scales = PyAttrOrNone(module, "_cached_compute_backend_row_scales");
      if (!cached_compute_words.is_none() && !cached_compute_row_scales.is_none()) {
        auto compute_words = py::cast<torch::Tensor>(cached_compute_words);
        auto compute_row_scales = py::cast<torch::Tensor>(cached_compute_row_scales);
        if (compute_words.defined() && compute_row_scales.defined() &&
            compute_words.device() == compute_row_scales.device()) {
          out.compute_packed_words = compute_words;
          out.compute_row_scales = compute_row_scales;
        }
      }
      if ((!out.compute_packed_words.defined() || !out.compute_row_scales.defined()) &&
          PyCallableAttr(module, "_compute_backend_weight")) {
        auto compute_backend = py::cast<py::tuple>(module.attr("_compute_backend_weight")("device"_a = out.packed_weight.device()));
        out.compute_packed_words = py::cast<torch::Tensor>(compute_backend[0]);
        out.compute_row_scales = py::cast<torch::Tensor>(compute_backend[1]);
      }
      if (out.compute_row_scales.defined() && out.compute_row_scales.dim() == 2) {
        out.compute_tile_n = out.compute_row_scales.size(1);
      }

      auto cached_decode_nz_masks = PyAttrOrNone(module, "_cached_decode_backend_nz_masks");
      auto cached_decode_sign_masks = PyAttrOrNone(module, "_cached_decode_backend_sign_masks");
      auto cached_decode_row_scales = PyAttrOrNone(module, "_cached_decode_backend_row_scales");
      if (!cached_decode_nz_masks.is_none() && !cached_decode_sign_masks.is_none() && !cached_decode_row_scales.is_none()) {
        auto decode_nz_masks = py::cast<torch::Tensor>(cached_decode_nz_masks);
        auto decode_sign_masks = py::cast<torch::Tensor>(cached_decode_sign_masks);
        auto decode_row_scales = py::cast<torch::Tensor>(cached_decode_row_scales);
        if (decode_nz_masks.defined() && decode_sign_masks.defined() && decode_row_scales.defined() &&
            decode_nz_masks.device() == decode_sign_masks.device() &&
            decode_nz_masks.device() == decode_row_scales.device()) {
          out.decode_nz_masks = decode_nz_masks;
          out.decode_sign_masks = decode_sign_masks;
          out.decode_row_scales = decode_row_scales;
        }
      }
      const auto backend_device =
          out.compute_packed_words.defined() ? out.compute_packed_words.device() : out.packed_weight.device();
      if ((!out.decode_nz_masks.defined() || !out.decode_sign_masks.defined() || !out.decode_row_scales.defined()) &&
          PyCallableAttr(module, "_decode_backend_weight")) {
        auto decode_backend = py::cast<py::tuple>(module.attr("_decode_backend_weight")("device"_a = backend_device));
        out.decode_nz_masks = py::cast<torch::Tensor>(decode_backend[0]);
        out.decode_sign_masks = py::cast<torch::Tensor>(decode_backend[1]);
        out.decode_row_scales = py::cast<torch::Tensor>(decode_backend[2]);
      }
      if (out.decode_row_scales.defined() && out.decode_row_scales.dim() == 2) {
        out.decode_tile_n = out.decode_row_scales.size(1);
      }
    }
    if (PyCallableAttr(module, "_uses_int8_packed_backend") &&
        py::cast<bool>(module.attr("_uses_int8_packed_backend")())) {
      auto cached_qweight = PyAttrOrNone(module, "_cached_int8_backend_qweight");
      auto cached_inv_scale = PyAttrOrNone(module, "_cached_int8_backend_inv_scale");
      if (!cached_qweight.is_none() && !cached_inv_scale.is_none()) {
        auto qweight = py::cast<torch::Tensor>(cached_qweight);
        auto inv_scale = py::cast<torch::Tensor>(cached_inv_scale);
        if (qweight.defined() && inv_scale.defined() &&
            qweight.device() == out.packed_weight.device() &&
            inv_scale.device() == out.packed_weight.device()) {
          out.qweight = qweight;
          out.inv_scale = inv_scale;
        }
      }
      if (!out.qweight.defined() || !out.inv_scale.defined()) {
        auto int8_backend = py::cast<py::tuple>(module.attr("_int8_backend_weight")("device"_a = out.packed_weight.device()));
        out.qweight = py::cast<torch::Tensor>(int8_backend[0]);
        out.inv_scale = py::cast<torch::Tensor>(int8_backend[1]);
      }
    }
    *state = std::move(out);
    return true;
  } catch (...) {
    return false;
  }
}

int64_t BitNetQuantMax(int64_t bits) {
  TORCH_CHECK(bits >= 2, "BitNet activation quantization bits must be >= 2");
  return (static_cast<int64_t>(1) << (bits - 1)) - 1;
}

std::vector<int64_t> BitNetPowerOfTwoSegments(int64_t size) {
  std::vector<int64_t> out;
  if (size <= 0) {
    return out;
  }
  int64_t remaining = size;
  while (remaining > 0) {
    int64_t seg = static_cast<int64_t>(1) << (63 - __builtin_clzll(static_cast<unsigned long long>(remaining)));
    out.push_back(seg);
    remaining -= seg;
  }
  return out;
}

torch::Tensor HadamardLastDimPower2(const torch::Tensor& x) {
  const auto width = x.size(-1);
  if (width <= 1) {
    return x;
  }
  TORCH_CHECK((width & (width - 1)) == 0, "Hadamard width must be a power of two");
  auto y = x.reshape({-1, width});
  int64_t block = 1;
  while (block < width) {
    auto view = y.view({-1, width / (block * 2), 2, block});
    auto a = view.select(2, 0);
    auto b = view.select(2, 1);
    y = torch::cat({a + b, a - b}, -1);
    block *= 2;
  }
  return (y.view(x.sizes()) / std::sqrt(static_cast<double>(width))).contiguous();
}

torch::Tensor ApplyBitNetSpinTransform(const torch::Tensor& x, const torch::Tensor& spin_signs) {
  TORCH_CHECK(x.size(-1) == spin_signs.numel(), "spin_signs must match the last dimension of x");
  auto work_dtype = x.scalar_type();
  if (!(work_dtype == torch::kFloat32 || work_dtype == torch::kFloat16 || work_dtype == torch::kBFloat16)) {
    work_dtype = torch::kFloat32;
  }
  auto signs = spin_signs.to(torch::TensorOptions().device(x.device()).dtype(work_dtype));
  auto x_local = x.to(torch::TensorOptions().device(x.device()).dtype(work_dtype)) * signs;
  auto segments = BitNetPowerOfTwoSegments(signs.numel());
  if (segments.empty()) {
    return x_local;
  }
  if (segments.size() == 1) {
    return HadamardLastDimPower2(x_local);
  }
  std::vector<torch::Tensor> parts;
  parts.reserve(segments.size());
  int64_t start = 0;
  for (auto seg : segments) {
    const auto stop = start + seg;
    auto part = x_local.slice(-1, start, stop);
    parts.push_back(seg > 1 ? HadamardLastDimPower2(part) : part);
    start = stop;
  }
  return torch::cat(parts, -1).contiguous();
}

bool BitNetPreScaleActive(const torch::Tensor& pre_scale) {
  if (!pre_scale.defined() || pre_scale.numel() == 0) {
    return false;
  }
  return (pre_scale.detach().to(torch::kFloat32) - 1.0).abs().amax().item<float>() > 1.0e-6f;
}

torch::Tensor ApplyBitNetPreScaleToInput(const torch::Tensor& x, const torch::Tensor& pre_scale) {
  std::vector<int64_t> view_shape(static_cast<size_t>(x.dim()), 1);
  view_shape.back() = pre_scale.numel();
  return x / pre_scale.to(torch::TensorOptions().device(x.device()).dtype(x.scalar_type())).view(view_shape);
}

bool BitNetActivationCalibrationMethodSupported(const std::string& method) {
  const auto method_name = NormalizeBackendName(method);
  return method_name.empty() || method_name == "absmax" || method_name == "percentile" || method_name == "mse";
}

torch::Tensor CalibrateBitNetActivationScale(
    const torch::Tensor& x,
    const std::string& method,
    int64_t bits,
    double percentile) {
  const auto method_name = NormalizeBackendName(method);
  TORCH_CHECK(BitNetActivationCalibrationMethodSupported(method_name),
              "Unsupported native BitNet activation calibration method: ",
              method);
  const auto qmax = static_cast<double>(BitNetQuantMax(bits));
  auto abs_x = x.detach().to(torch::kFloat32).abs();
  torch::Tensor clip;
  if (method_name == "percentile") {
    const auto q = std::max(0.0, std::min(1.0, percentile));
    auto flat = abs_x.reshape({-1});
    const auto k = std::max<int64_t>(static_cast<int64_t>(q * static_cast<double>(std::max<int64_t>(flat.numel() - 1, 0))), 0);
    clip = std::get<0>(torch::kthvalue(flat, k + 1));
  } else {
    // Python mse_scale() is currently an absmax fallback, so keep native behavior aligned.
    clip = abs_x.amax();
  }
  clip = clip.clamp_min(1.0e-8);
  return (clip / qmax).to(torch::kFloat32);
}

torch::Tensor CalibrateBitNetActivationRowScale(
    const torch::Tensor& x,
    const std::string& method,
    int64_t bits,
    double percentile) {
  const auto method_name = NormalizeBackendName(method);
  TORCH_CHECK(BitNetActivationCalibrationMethodSupported(method_name),
              "Unsupported native BitNet activation calibration method: ",
              method);
  const auto qmax = static_cast<double>(BitNetQuantMax(bits));
  const auto cols = x.size(-1);
  const auto rows = cols > 0 ? x.numel() / cols : 0;
  if (rows == 0 || cols == 0) {
    return torch::empty({rows}, torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
  }
  auto abs_x = x.detach().to(torch::kFloat32).abs().reshape({rows, cols});
  torch::Tensor clip;
  if (method_name == "percentile") {
    const auto q = std::max(0.0, std::min(1.0, percentile));
    const auto k = std::max<int64_t>(
        static_cast<int64_t>(q * static_cast<double>(std::max<int64_t>(cols - 1, 0))), 0);
    clip = std::get<0>(torch::kthvalue(abs_x, k + 1, -1));
  } else {
    // Python mse_scale() is currently an absmax fallback, so keep native behavior aligned.
    clip = std::get<0>(abs_x.max(-1));
  }
  return (clip.clamp_min(1.0e-8) / qmax).to(torch::kFloat32).contiguous();
}

torch::Tensor FakeQuantizeBitNetActivation(
    const torch::Tensor& x,
    const torch::Tensor& scale,
    int64_t bits) {
  const auto qmax = static_cast<double>(BitNetQuantMax(bits));
  auto scale_t = scale.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).clamp_min(1.0e-8);
  while (scale_t.dim() < x.dim()) {
    scale_t = scale_t.unsqueeze(0);
  }
  auto q = torch::round(x.to(torch::kFloat32) / scale_t).clamp_(-qmax, qmax);
  return (q * scale_t).to(x.scalar_type());
}

torch::Tensor FakeQuantizeBitNetActivationRowwise(
    const torch::Tensor& x,
    const torch::Tensor& row_scale,
    int64_t bits) {
  const auto qmax = static_cast<double>(BitNetQuantMax(bits));
  const auto cols = x.size(-1);
  const auto rows = cols > 0 ? x.numel() / cols : 0;
  auto scale_t = row_scale.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).reshape({-1});
  TORCH_CHECK(
      scale_t.numel() == 1 || scale_t.numel() == rows,
      "BitNet activation row_scale must have 1 or rows elements");
  if (scale_t.numel() == 1) {
    scale_t = scale_t.expand({rows});
  }
  scale_t = scale_t.clamp_min(1.0e-8).contiguous();
  auto flat = x.reshape({rows, cols}).to(torch::kFloat32);
  auto q = torch::round(flat / scale_t.unsqueeze(1)).clamp_(-qmax, qmax);
  return (q * scale_t.unsqueeze(1)).to(x.scalar_type()).view(x.sizes());
}

bool BitNetModuleDirectSupported(const BitNetModuleState& state) {
  const auto mode = NormalizeBackendName(state.act_quant_mode);
  if (mode.empty() || mode == "none" || mode == "off") {
    return true;
  }
  if ((mode == "dynamic_int8" || mode == "static_int8") && state.act_quant_bits >= 2) {
    return BitNetActivationCalibrationMethodSupported(state.act_quant_method);
  }
  return false;
}

bool BitNetModuleDirectSupported(const py::object& module, BitNetModuleState* state = nullptr) {
  BitNetModuleState local;
  if (!TryLoadBitNetModuleState(module, &local)) {
    return false;
  }
  if (!BitNetModuleDirectSupported(local)) {
    return false;
  }
  if (state != nullptr) {
    *state = std::move(local);
  }
  return true;
}

bool BitNetCudaTransformInputSupported(const BitNetModuleState& state) {
#if MODEL_STACK_WITH_CUDA
  if (!t10::bitnet::HasCudaBitNetInputFrontendKernel()) {
    return false;
  }
  if (state.spin_enabled) {
    return false;
  }
  const auto mode = NormalizeBackendName(state.act_quant_mode);
  if (mode.empty() || mode == "none" || mode == "off") {
    return true;
  }
  if (mode == "static_int8") {
    return state.act_quant_bits >= 2 && state.act_scale.defined();
  }
  if (mode == "dynamic_int8") {
    const auto method = NormalizeBackendName(state.act_quant_method);
    return state.act_quant_bits >= 2 && (method.empty() || method == "absmax" || method == "mse");
  }
#endif
  return false;
}

torch::Tensor ApplyBitNetModuleInputTransforms(const torch::Tensor& x, const BitNetModuleState& state) {
  auto target_dtype = x.scalar_type();
  if (!(target_dtype == torch::kFloat32 || target_dtype == torch::kFloat16 || target_dtype == torch::kBFloat16)) {
    target_dtype = torch::kFloat32;
  }
  auto x_local = x.to(torch::TensorOptions().device(x.device()).dtype(target_dtype));
  if (state.spin_enabled && state.spin_signs.defined() && state.spin_signs.numel() > 0) {
    x_local = ApplyBitNetSpinTransform(x_local, state.spin_signs);
  }
  if (state.pre_scale.defined() && BitNetPreScaleActive(state.pre_scale)) {
    x_local = ApplyBitNetPreScaleToInput(x_local, state.pre_scale);
  }
  const auto mode = NormalizeBackendName(state.act_quant_mode);
  if (mode.empty() || mode == "none" || mode == "off") {
    return x_local;
  }
  if (mode == "dynamic_int8") {
    auto row_scale = CalibrateBitNetActivationRowScale(
        x_local, state.act_quant_method, state.act_quant_bits, state.act_quant_percentile);
    return FakeQuantizeBitNetActivationRowwise(x_local, row_scale, state.act_quant_bits);
  }
  if (mode == "static_int8") {
    TORCH_CHECK(state.act_scale.defined(), "BitNet static_int8 activation quantization requires act_scale");
    return FakeQuantizeBitNetActivationRowwise(x_local, state.act_scale, state.act_quant_bits);
  }
  TORCH_CHECK(false, "Unsupported native BitNet activation quantization mode: ", state.act_quant_mode);
}

torch::Tensor BitNetTransformInputForward(
    const torch::Tensor& x,
    bool spin_enabled,
    const c10::optional<torch::Tensor>& spin_signs,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    double act_quant_percentile,
    const c10::optional<torch::Tensor>& act_scale) {
  TORCH_CHECK(x.defined(), "bitnet_transform_input_forward: x must be defined");
  TORCH_CHECK(x.dim() >= 1, "bitnet_transform_input_forward: x must have rank >= 1");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "bitnet_transform_input_forward: x must use float32, float16, or bfloat16");

  BitNetModuleState state;
  state.spin_enabled = spin_enabled;
  state.spin_signs = spin_signs.value_or(torch::Tensor());
  state.pre_scale = pre_scale.value_or(torch::Tensor());
  state.act_quant_mode = act_quant_mode;
  state.act_quant_method = act_quant_method;
  state.act_quant_bits = act_quant_bits;
  state.act_quant_percentile = act_quant_percentile;
  state.act_scale = act_scale.value_or(torch::Tensor());

  TORCH_CHECK(BitNetModuleDirectSupported(state),
              "bitnet_transform_input_forward: unsupported BitNet activation transform");
  if (state.spin_enabled) {
    TORCH_CHECK(state.spin_signs.defined() && state.spin_signs.numel() == x.size(-1),
                "bitnet_transform_input_forward: spin_signs must match x last dimension when spin is enabled");
  }
  if (state.pre_scale.defined() && state.pre_scale.numel() > 0) {
    TORCH_CHECK(state.pre_scale.numel() == x.size(-1),
                "bitnet_transform_input_forward: pre_scale must match x last dimension");
  }
  if (NormalizeBackendName(state.act_quant_mode) == "static_int8" && state.act_scale.defined()) {
    TORCH_CHECK(state.act_scale.scalar_type() == torch::kFloat32,
                "bitnet_transform_input_forward: act_scale must use float32 storage");
    TORCH_CHECK(state.act_scale.numel() == 1 || state.act_scale.numel() == (x.numel() / x.size(-1)),
                "bitnet_transform_input_forward: act_scale must have 1 or rows elements");
  }
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && BitNetCudaTransformInputSupported(state)) {
    return t10::bitnet::CudaBitNetTransformInputForward(
        x,
        state.pre_scale.defined() && state.pre_scale.numel() > 0 ? c10::optional<torch::Tensor>(state.pre_scale) : c10::nullopt,
        NormalizeBackendName(state.act_quant_mode),
        NormalizeBackendName(state.act_quant_method),
        state.act_quant_bits,
        state.act_scale.defined() ? c10::optional<torch::Tensor>(state.act_scale) : c10::nullopt);
  }
#endif
  return ApplyBitNetModuleInputTransforms(x, state);
}

std::pair<torch::Tensor, torch::Tensor> QuantizeBitNetActivationInt8Codes(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    double act_quant_percentile,
    const c10::optional<torch::Tensor>& act_scale) {
  TORCH_CHECK(x.defined(), "bitnet_int8_linear_from_float_forward: x must be defined");
  TORCH_CHECK(x.dim() >= 2, "bitnet_int8_linear_from_float_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "bitnet_int8_linear_from_float_forward: x must use float32, float16, or bfloat16");

  const auto mode = NormalizeBackendName(act_quant_mode);
  TORCH_CHECK(
      mode == "dynamic_int8" || mode == "static_int8",
      "bitnet_int8_linear_from_float_forward: act_quant_mode must be dynamic_int8 or static_int8, got ",
      act_quant_mode);
  TORCH_CHECK(
      act_quant_bits >= 2 && act_quant_bits <= 8,
      "bitnet_int8_linear_from_float_forward: act_quant_bits must be in [2, 8]");

  auto x_local = x.to(torch::TensorOptions().device(x.device()).dtype(x.scalar_type())).contiguous();
  if (pre_scale.has_value() && pre_scale.value().defined() && pre_scale.value().numel() > 0) {
    const auto scale = pre_scale.value().reshape({-1});
    TORCH_CHECK(
        scale.numel() == x.size(-1),
        "bitnet_int8_linear_from_float_forward: pre_scale must match x last dimension");
    if (BitNetPreScaleActive(scale)) {
      x_local = ApplyBitNetPreScaleToInput(x_local, scale);
    }
  }

  const auto rows = x_local.numel() / x_local.size(-1);
  auto flat = x_local.reshape({rows, x_local.size(-1)});
  torch::Tensor row_scale;
  if (mode == "dynamic_int8") {
    row_scale = CalibrateBitNetActivationRowScale(
        x_local,
        act_quant_method,
        act_quant_bits,
        act_quant_percentile);
  } else {
    TORCH_CHECK(
        act_scale.has_value() && act_scale.value().defined(),
        "bitnet_int8_linear_from_float_forward: static_int8 requires act_scale");
    auto scale = act_scale.value().to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).reshape({-1});
    TORCH_CHECK(
        scale.numel() == 1 || scale.numel() == rows,
        "bitnet_int8_linear_from_float_forward: act_scale must have 1 or rows elements");
    row_scale = (scale.numel() == 1 ? scale.expand({rows}).clone() : scale.contiguous()).clamp_min_(1.0e-8);
  }

  const auto qmax = static_cast<double>(BitNetQuantMax(act_quant_bits));
  auto qx = torch::round(flat.to(torch::kFloat32) / row_scale.unsqueeze(1))
                .clamp_(-qmax, qmax)
                .to(torch::kInt8)
                .view(x_local.sizes())
                .contiguous();
  return {qx, row_scale.to(torch::TensorOptions().device(x.device()).dtype(torch::kFloat32)).contiguous()};
}

torch::Tensor BitNetTransformInputFromState(const torch::Tensor& x, const BitNetModuleState& state) {
  return BitNetTransformInputForward(
      x,
      state.spin_enabled,
      state.spin_signs.defined() ? c10::optional<torch::Tensor>(state.spin_signs) : c10::nullopt,
      state.pre_scale.defined() ? c10::optional<torch::Tensor>(state.pre_scale) : c10::nullopt,
      state.act_quant_mode,
      state.act_quant_method,
      state.act_quant_bits,
      state.act_quant_percentile,
      state.act_scale.defined() ? c10::optional<torch::Tensor>(state.act_scale) : c10::nullopt);
}

torch::Tensor BitNetInt8LinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    double act_quant_percentile,
    const c10::optional<torch::Tensor>& act_scale,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(
      x.defined() && qweight.defined() && inv_scale.defined(),
      "bitnet_int8_linear_from_float_forward: x, qweight, and inv_scale must be defined");
  TORCH_CHECK(x.dim() >= 2, "bitnet_int8_linear_from_float_forward: x must have rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "bitnet_int8_linear_from_float_forward: x must use float32, float16, or bfloat16");
  TORCH_CHECK(qweight.dim() == 2, "bitnet_int8_linear_from_float_forward: qweight must be rank-2");
  TORCH_CHECK(
      qweight.scalar_type() == torch::kInt8,
      "bitnet_int8_linear_from_float_forward: qweight must use int8 storage");
  TORCH_CHECK(inv_scale.dim() == 1, "bitnet_int8_linear_from_float_forward: inv_scale must be rank-1");
  TORCH_CHECK(
      inv_scale.scalar_type() == torch::kFloat32,
      "bitnet_int8_linear_from_float_forward: inv_scale must use float32 storage");
  TORCH_CHECK(
      inv_scale.size(0) == qweight.size(0),
      "bitnet_int8_linear_from_float_forward: inv_scale size mismatch");
  TORCH_CHECK(
      qweight.size(1) == x.size(-1),
      "bitnet_int8_linear_from_float_forward: qweight column count mismatch");
  if (bias.has_value() && bias.value().defined()) {
    const auto& b = bias.value();
    TORCH_CHECK(b.dim() == 1, "bitnet_int8_linear_from_float_forward: bias must be rank-1");
    TORCH_CHECK(b.size(0) == qweight.size(0), "bitnet_int8_linear_from_float_forward: bias size mismatch");
  }

#if MODEL_STACK_WITH_CUDA
  const auto mode = NormalizeBackendName(act_quant_mode);
  const auto method = NormalizeBackendName(act_quant_method);
  const bool cuda_dynamic_supported =
      mode == "dynamic_int8" && (method.empty() || method == "absmax" || method == "mse");
  const bool cuda_static_supported = mode == "static_int8" && act_scale.has_value() && act_scale.value().defined();
  const bool has_pre_scale = pre_scale.has_value() && pre_scale.value().defined() && pre_scale.value().numel() > 0;
  const bool generic_int8_frontend_supported =
      !has_pre_scale && act_quant_bits == 8 &&
      ((mode == "static_int8" && act_scale.has_value() && act_scale.value().defined()) ||
       (mode == "dynamic_int8" && (method.empty() || method == "absmax")));
  if (x.is_cuda() && qweight.is_cuda() && inv_scale.is_cuda() && generic_int8_frontend_supported) {
    c10::optional<torch::Tensor> provided_scale = c10::nullopt;
    if (mode == "static_int8") {
      provided_scale = act_scale;
    }
    return Int8LinearFromFloatForward(
        x,
        qweight,
        inv_scale,
        bias,
        provided_scale,
        out_dtype);
  }
  const bool generic_int8_pre_scale_supported =
      has_pre_scale && act_quant_bits == 8 && mode == "dynamic_int8" && (method.empty() || method == "absmax");
  if (x.is_cuda() && qweight.is_cuda() && inv_scale.is_cuda() && generic_int8_pre_scale_supported) {
    return CudaInt8LinearFromFloatPreScaleForward(
        x,
        qweight,
        inv_scale,
        bias,
        pre_scale.value(),
        out_dtype);
  }
  // Intentionally stay on the split CUDA path here. The standalone BitNet row-1
  // from-float kernel is numerically correct, but on Ampere decode it loses to
  // "quantize once -> int8 backend" in the full model loop because the int8
  // backend can take the better small-GEMM route.
  if (x.is_cuda() && qweight.is_cuda() && inv_scale.is_cuda() && (cuda_dynamic_supported || cuda_static_supported)) {
    auto quantized = t10::bitnet::CudaBitNetQuantizeActivationInt8CodesForward(
        x,
        pre_scale,
        mode,
        method,
        act_quant_bits,
        act_scale);
    return Int8LinearForward(
        std::get<0>(quantized),
        std::get<1>(quantized),
        qweight,
        inv_scale,
        bias,
        out_dtype);
  }
#endif

  auto quantized = QuantizeBitNetActivationInt8Codes(
      x,
      pre_scale,
      act_quant_mode,
      act_quant_method,
      act_quant_bits,
      act_quant_percentile,
      act_scale);
  return Int8LinearForward(
      quantized.first,
      quantized.second,
      qweight,
      inv_scale,
      bias,
      out_dtype);
}

std::vector<torch::Tensor> BitNetInt8FusedQkvPackedHeadsProjectionForward(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    const c10::optional<torch::Tensor>& packed_bias,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    double act_quant_percentile,
    const c10::optional<torch::Tensor>& act_scale,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads,
    const c10::optional<torch::ScalarType>& out_dtype) {
  TORCH_CHECK(x.defined(), "bitnet_int8_fused_qkv_packed_heads_projection_forward: x must be defined");
  TORCH_CHECK(
      x.dim() == 3,
      "bitnet_int8_fused_qkv_packed_heads_projection_forward: x must have shape (B, T, D)");
  TORCH_CHECK(
      q_size > 0 && k_size > 0 && v_size > 0,
      "bitnet_int8_fused_qkv_packed_heads_projection_forward: q/k/v sizes must be positive");
  TORCH_CHECK(
      q_heads > 0 && kv_heads > 0,
      "bitnet_int8_fused_qkv_packed_heads_projection_forward: q_heads and kv_heads must be positive");
  TORCH_CHECK(
      qweight.defined() && inv_scale.defined(),
      "bitnet_int8_fused_qkv_packed_heads_projection_forward: qweight and inv_scale must be defined");
  TORCH_CHECK(
      qweight.dim() == 2 && inv_scale.dim() == 1,
      "bitnet_int8_fused_qkv_packed_heads_projection_forward: qweight must be rank-2 and inv_scale must be rank-1");
  TORCH_CHECK(
      qweight.size(0) == q_size + k_size + v_size,
      "bitnet_int8_fused_qkv_packed_heads_projection_forward: fused output width mismatch");
  if (packed_bias.has_value() && packed_bias.value().defined()) {
    TORCH_CHECK(
        packed_bias.value().dim() == 1,
        "bitnet_int8_fused_qkv_packed_heads_projection_forward: packed_bias must be rank-1");
    TORCH_CHECK(
        packed_bias.value().size(0) == qweight.size(0),
        "bitnet_int8_fused_qkv_packed_heads_projection_forward: packed_bias size mismatch");
  }
  auto fused = BitNetInt8LinearFromFloatForward(
      x,
      qweight,
      inv_scale,
      packed_bias,
      pre_scale,
      act_quant_mode,
      act_quant_method,
      act_quant_bits,
      act_quant_percentile,
      act_scale,
      out_dtype);
  return {
      SplitHeadsForward(fused.slice(-1, 0, q_size), q_heads),
      SplitHeadsForward(fused.slice(-1, q_size, q_size + k_size), kv_heads),
      SplitHeadsForward(fused.slice(-1, q_size + k_size, q_size + k_size + v_size), kv_heads),
  };
}

torch::Tensor BitNetLinearFromFloatForward(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale_values,
    const torch::Tensor& layout_header,
    const torch::Tensor& segment_offsets,
    const c10::optional<torch::Tensor>& bias,
    bool spin_enabled,
    const c10::optional<torch::Tensor>& spin_signs,
    const c10::optional<torch::Tensor>& pre_scale,
    const std::string& act_quant_mode,
    const std::string& act_quant_method,
    int64_t act_quant_bits,
    double act_quant_percentile,
    const c10::optional<torch::Tensor>& act_scale,
    const c10::optional<torch::ScalarType>& out_dtype) {
#if MODEL_STACK_WITH_CUDA
  const auto mode = NormalizeBackendName(act_quant_mode);
  const auto method = NormalizeBackendName(act_quant_method);
  const bool no_transform = !spin_enabled &&
      !(pre_scale.has_value() && pre_scale.value().defined() && pre_scale.value().numel() > 0) &&
      (mode.empty() || mode == "none" || mode == "off");
  const bool static_fused_supported = !spin_enabled && mode == "static_int8" && act_scale.has_value() &&
      act_scale.value().defined() && act_quant_bits >= 2 && act_quant_bits <= 8;
  const bool dynamic_fused_supported =
      !spin_enabled && mode == "dynamic_int8" && act_quant_bits >= 2 && act_quant_bits <= 8 &&
      (method.empty() || method == "absmax" || method == "mse");
  if (x.is_cuda() && packed_weight.is_cuda() && scale_values.is_cuda() && layout_header.is_cuda() &&
      segment_offsets.is_cuda() && t10::bitnet::HasCudaBitNetLinearKernel() &&
      (no_transform || static_fused_supported || dynamic_fused_supported ||
       ((!spin_enabled) && t10::bitnet::HasCudaBitNetInputFrontendKernel() &&
        (mode.empty() || mode == "none" || mode == "off" || mode == "static_int8" || mode == "dynamic_int8")))) {
    return t10::bitnet::CudaBitNetLinearFromFloatForward(
        x,
        packed_weight,
        scale_values,
        layout_header,
        segment_offsets,
        bias,
        pre_scale,
        mode,
        method,
        act_quant_bits,
        act_scale,
        out_dtype);
  }
#endif
  auto x_local = BitNetTransformInputForward(
      x,
      spin_enabled,
      spin_signs,
      pre_scale,
      act_quant_mode,
      act_quant_method,
      act_quant_bits,
      act_quant_percentile,
      act_scale);
  return BitNetLinearForward(
      x_local,
      packed_weight,
      scale_values,
      layout_header,
      segment_offsets,
      bias,
      out_dtype,
      false);
}

int64_t BitNetLayoutHeaderScalar(const torch::Tensor& layout_header, int64_t idx) {
  auto header_cpu = layout_header.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kLong)).contiguous();
  return header_cpu[idx].item<int64_t>();
}

bool BitNetOptionalTensorEqual(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  if (!lhs.defined() && !rhs.defined()) {
    return true;
  }
  if (lhs.defined() != rhs.defined()) {
    return false;
  }
  return torch::equal(lhs, rhs);
}

bool BitNetOptionalTensorEqual(
    const c10::optional<torch::Tensor>& lhs,
    const c10::optional<torch::Tensor>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return BitNetOptionalTensorEqual(lhs.value(), rhs.value());
}

bool BitNetModuleStatesShareInputTransforms(
    const BitNetModuleState& q_state,
    const BitNetModuleState& k_state,
    const BitNetModuleState& v_state) {
  auto normalized_mode = [](const BitNetModuleState& state) {
    auto mode = NormalizeBackendName(state.act_quant_mode);
    return mode.empty() || mode == "off" ? std::string("none") : mode;
  };
  auto activation_signature_matches = [&](const BitNetModuleState& lhs, const BitNetModuleState& rhs) {
    const auto lhs_mode = normalized_mode(lhs);
    const auto rhs_mode = normalized_mode(rhs);
    if (lhs_mode != rhs_mode) {
      return false;
    }
    if (lhs_mode == "none") {
      return true;
    }
    if (lhs_mode == "dynamic_int8") {
      return NormalizeBackendName(lhs.act_quant_method) == NormalizeBackendName(rhs.act_quant_method) &&
          lhs.act_quant_bits == rhs.act_quant_bits &&
          std::abs(lhs.act_quant_percentile - rhs.act_quant_percentile) <= 1.0e-9;
    }
    if (lhs_mode == "static_int8") {
      return lhs.act_quant_bits == rhs.act_quant_bits &&
          BitNetOptionalTensorEqual(lhs.act_scale, rhs.act_scale);
    }
    return NormalizeBackendName(lhs.act_quant_method) == NormalizeBackendName(rhs.act_quant_method) &&
        lhs.act_quant_bits == rhs.act_quant_bits &&
        std::abs(lhs.act_quant_percentile - rhs.act_quant_percentile) <= 1.0e-9 &&
        BitNetOptionalTensorEqual(lhs.act_scale, rhs.act_scale);
  };
  auto compatible_with_q = [&](const BitNetModuleState& other) {
    return q_state.spin_enabled == other.spin_enabled &&
        BitNetOptionalTensorEqual(q_state.spin_signs, other.spin_signs) &&
        BitNetOptionalTensorEqual(q_state.pre_scale, other.pre_scale) &&
        activation_signature_matches(q_state, other);
  };
  return compatible_with_q(k_state) && compatible_with_q(v_state);
}

bool BitNetStateUsesInt8PackedPath(const BitNetModuleState& state) {
  const auto mode = NormalizeBackendName(state.act_quant_mode);
  return !state.spin_enabled && (mode == "dynamic_int8" || mode == "static_int8");
}

std::uintptr_t BitNetTensorPtrKey(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0 ? reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) : 0;
}

std::uintptr_t BitNetTensorPtrKey(const c10::optional<torch::Tensor>& tensor) {
  return tensor.has_value() ? BitNetTensorPtrKey(tensor.value()) : 0;
}

int64_t BitNetTensorVersionKey(const torch::Tensor& tensor) {
  return tensor.defined() ? static_cast<int64_t>(tensor._version()) : -1;
}

struct DenseBitNetWeightCacheKey {
  int64_t device_index = -1;
  int64_t dtype = -1;
  int64_t rows = 0;
  int64_t cols = 0;
  std::uintptr_t qweight_ptr = 0;
  std::uintptr_t inv_scale_ptr = 0;
  int64_t qweight_version = -1;
  int64_t inv_scale_version = -1;

  bool operator==(const DenseBitNetWeightCacheKey& other) const {
    return device_index == other.device_index &&
        dtype == other.dtype &&
        rows == other.rows &&
        cols == other.cols &&
        qweight_ptr == other.qweight_ptr &&
        inv_scale_ptr == other.inv_scale_ptr &&
        qweight_version == other.qweight_version &&
        inv_scale_version == other.inv_scale_version;
  }
};

struct DenseBitNetWeightCacheKeyHash {
  size_t operator()(const DenseBitNetWeightCacheKey& key) const {
    size_t out = static_cast<size_t>(key.qweight_ptr);
    out ^= static_cast<size_t>(key.inv_scale_ptr + 0x9e3779b97f4a7c15ULL + (out << 6) + (out >> 2));
    out ^= static_cast<size_t>(key.device_index + 0x9e3779b97f4a7c15ULL + (out << 6) + (out >> 2));
    out ^= static_cast<size_t>(key.dtype + 0x9e3779b97f4a7c15ULL + (out << 6) + (out >> 2));
    out ^= static_cast<size_t>(key.rows + 0x9e3779b97f4a7c15ULL + (out << 6) + (out >> 2));
    out ^= static_cast<size_t>(key.cols + 0x9e3779b97f4a7c15ULL + (out << 6) + (out >> 2));
    out ^= static_cast<size_t>(key.qweight_version + 0x9e3779b97f4a7c15ULL + (out << 6) + (out >> 2));
    out ^= static_cast<size_t>(key.inv_scale_version + 0x9e3779b97f4a7c15ULL + (out << 6) + (out >> 2));
    return out;
  }
};

torch::Tensor DenseBitNetInt8WeightCached(
    const torch::Tensor& qweight,
    const torch::Tensor& inv_scale,
    torch::ScalarType dtype) {
  static thread_local std::unordered_map<
      DenseBitNetWeightCacheKey,
      torch::Tensor,
      DenseBitNetWeightCacheKeyHash>
      cache;
  DenseBitNetWeightCacheKey key;
  key.device_index = qweight.device().is_cuda() ? qweight.device().index() : -1;
  key.dtype = static_cast<int64_t>(dtype);
  key.rows = qweight.dim() == 2 ? qweight.size(0) : 0;
  key.cols = qweight.dim() == 2 ? qweight.size(1) : 0;
  key.qweight_ptr = BitNetTensorPtrKey(qweight);
  key.inv_scale_ptr = BitNetTensorPtrKey(inv_scale);
  key.qweight_version = BitNetTensorVersionKey(qweight);
  key.inv_scale_version = BitNetTensorVersionKey(inv_scale);
  auto found = cache.find(key);
  if (found != cache.end() && found->second.defined()) {
    return found->second;
  }
  auto scale = inv_scale.to(qweight.device(), dtype).unsqueeze(1);
  auto dense = (qweight.to(qweight.device(), dtype) * scale).contiguous();
  cache[key] = dense;
  return dense;
}

struct BitNetInt8QkvFuseCacheEntry {
  bool valid = false;
  std::uintptr_t q_qweight = 0;
  std::uintptr_t k_qweight = 0;
  std::uintptr_t v_qweight = 0;
  std::uintptr_t q_inv_scale = 0;
  std::uintptr_t k_inv_scale = 0;
  std::uintptr_t v_inv_scale = 0;
  std::uintptr_t q_bias = 0;
  std::uintptr_t k_bias = 0;
  std::uintptr_t v_bias = 0;
  std::uintptr_t q_pre_scale = 0;
  std::uintptr_t q_act_scale = 0;
  int64_t q_rows = 0;
  int64_t k_rows = 0;
  int64_t v_rows = 0;
  int64_t logical_in = 0;
  int64_t act_quant_bits = 0;
  double act_quant_percentile = 0.0;
  std::string act_quant_mode;
  std::string act_quant_method;
  BitNetModuleState fused_state;
  int64_t q_size = 0;
  int64_t k_size = 0;
  int64_t v_size = 0;
};

torch::Tensor ExpandBitNetRowScales(const BitNetModuleState& state) {
  const auto logical_out = BitNetLayoutHeaderScalar(state.layout_header, 3);
  const auto scale_granularity = BitNetLayoutHeaderScalar(state.layout_header, 7);
  const auto scale_group_size = BitNetLayoutHeaderScalar(state.layout_header, 8);
  auto out = torch::zeros({logical_out}, torch::TensorOptions().device(state.scale_values.device()).dtype(torch::kFloat32));
  auto values = state.scale_values.to(torch::TensorOptions().device(state.scale_values.device()).dtype(torch::kFloat32)).reshape({-1});
  if (scale_granularity == 0) {
    out.fill_(values[0].item<float>());
    return out;
  }
  if (scale_granularity == 1) {
    auto offsets = state.segment_offsets.to(torch::TensorOptions().device(state.scale_values.device()).dtype(torch::kInt64)).reshape({-1});
    for (int64_t idx = 0; idx + 1 < offsets.numel(); ++idx) {
      const auto start = offsets[idx].item<int64_t>();
      const auto end = offsets[idx + 1].item<int64_t>();
      out.slice(0, start, end).fill_(values[idx].item<float>());
    }
    return out;
  }
  if (scale_granularity == 2) {
    TORCH_CHECK(scale_group_size > 0, "BitNet per-output-group scaling requires a positive scale_group_size");
    for (int64_t idx = 0; idx < values.numel(); ++idx) {
      const auto start = idx * scale_group_size;
      const auto end = std::min<int64_t>(logical_out, start + scale_group_size);
      out.slice(0, start, end).fill_(values[idx].item<float>());
    }
    return out;
  }
  TORCH_CHECK(false, "Unsupported BitNet scale granularity for fused QKV state: ", scale_granularity);
}

torch::Tensor FlattenComputePackedBitNetRows(const torch::Tensor& compute_packed_words, int64_t logical_rows) {
  TORCH_CHECK(compute_packed_words.dim() == 3, "compute-packed BitNet rows must be rank-3");
  auto row_major = compute_packed_words.permute({0, 2, 1}).contiguous().view({-1, compute_packed_words.size(1)});
  return row_major.slice(0, 0, logical_rows).contiguous();
}

torch::Tensor FlattenComputePackedBitNetScales(const torch::Tensor& compute_row_scales, int64_t logical_rows) {
  TORCH_CHECK(compute_row_scales.dim() == 2, "compute-packed BitNet row scales must be rank-2");
  auto flat = compute_row_scales.contiguous().view({-1});
  return flat.slice(0, 0, logical_rows).contiguous();
}

torch::Tensor FlattenDecodePackedBitNetRows(const torch::Tensor& decode_masks, int64_t logical_rows) {
  TORCH_CHECK(decode_masks.dim() == 3, "decode-packed BitNet masks must be rank-3");
  auto row_major = decode_masks.permute({0, 2, 1}).contiguous().view({-1, decode_masks.size(1)});
  return row_major.slice(0, 0, logical_rows).contiguous();
}

std::vector<torch::Tensor> SplitProjectedBitNetQkv(
    const torch::Tensor& projected,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads) {
  if (projected.dim() == 3 && projected.size(1) == 1 && projected.is_contiguous()) {
    TORCH_CHECK(q_size % q_heads == 0, "split projected BitNet QKV: q size must divide q heads");
    TORCH_CHECK(k_size % kv_heads == 0, "split projected BitNet QKV: k size must divide kv heads");
    TORCH_CHECK(v_size % kv_heads == 0, "split projected BitNet QKV: v size must divide kv heads");
    const auto batch = projected.size(0);
    const auto total = projected.size(2);
    auto split_view = [&](int64_t offset, int64_t heads, int64_t head_dim) {
      return projected.as_strided(
          {batch, heads, 1, head_dim},
          {total, head_dim, total, 1},
          projected.storage_offset() + offset);
    };
    return {
        split_view(0, q_heads, q_size / q_heads),
        split_view(q_size, kv_heads, k_size / kv_heads),
        split_view(q_size + k_size, kv_heads, v_size / kv_heads),
    };
  }
  return {
      SplitHeadsForward(projected.slice(-1, 0, q_size), q_heads),
      SplitHeadsForward(projected.slice(-1, q_size, q_size + k_size), kv_heads),
      SplitHeadsForward(projected.slice(-1, q_size + k_size, q_size + k_size + v_size), kv_heads),
  };
}

bool TryFuseBitNetQkvStates(
    const BitNetModuleState& q_state,
    const BitNetModuleState& k_state,
    const BitNetModuleState& v_state,
    BitNetModuleState* fused_state,
    int64_t* q_size_out,
    int64_t* k_size_out,
    int64_t* v_size_out) {
  if (fused_state == nullptr || q_size_out == nullptr || k_size_out == nullptr || v_size_out == nullptr) {
    return false;
  }
  if (!BitNetModuleStatesShareInputTransforms(q_state, k_state, v_state)) {
    return false;
  }
  const bool int8_qkv =
      BitNetStateUsesInt8PackedPath(q_state) &&
      BitNetStateUsesInt8PackedPath(k_state) &&
      BitNetStateUsesInt8PackedPath(v_state) &&
      q_state.qweight.defined() && k_state.qweight.defined() && v_state.qweight.defined() &&
      q_state.inv_scale.defined() && k_state.inv_scale.defined() && v_state.inv_scale.defined();
  if (int8_qkv) {
    const auto q_size = q_state.qweight.size(0);
    const auto k_size = k_state.qweight.size(0);
    const auto v_size = v_state.qweight.size(0);
    const auto logical_in = q_state.qweight.size(1);
    if (q_state.qweight.dim() != 2 || k_state.qweight.dim() != 2 || v_state.qweight.dim() != 2 ||
        q_state.inv_scale.dim() != 1 || k_state.inv_scale.dim() != 1 || v_state.inv_scale.dim() != 1 ||
        k_state.qweight.size(1) != logical_in || v_state.qweight.size(1) != logical_in ||
        q_state.inv_scale.size(0) != q_size || k_state.inv_scale.size(0) != k_size ||
        v_state.inv_scale.size(0) != v_size) {
      return false;
    }
    static thread_local BitNetInt8QkvFuseCacheEntry cache;
    const bool cache_hit =
        cache.valid &&
        cache.q_qweight == BitNetTensorPtrKey(q_state.qweight) &&
        cache.k_qweight == BitNetTensorPtrKey(k_state.qweight) &&
        cache.v_qweight == BitNetTensorPtrKey(v_state.qweight) &&
        cache.q_inv_scale == BitNetTensorPtrKey(q_state.inv_scale) &&
        cache.k_inv_scale == BitNetTensorPtrKey(k_state.inv_scale) &&
        cache.v_inv_scale == BitNetTensorPtrKey(v_state.inv_scale) &&
        cache.q_bias == BitNetTensorPtrKey(q_state.bias) &&
        cache.k_bias == BitNetTensorPtrKey(k_state.bias) &&
        cache.v_bias == BitNetTensorPtrKey(v_state.bias) &&
        cache.q_pre_scale == BitNetTensorPtrKey(q_state.pre_scale) &&
        cache.q_act_scale == BitNetTensorPtrKey(q_state.act_scale) &&
        cache.q_rows == q_size &&
        cache.k_rows == k_size &&
        cache.v_rows == v_size &&
        cache.logical_in == logical_in &&
        cache.act_quant_bits == q_state.act_quant_bits &&
        cache.act_quant_percentile == q_state.act_quant_percentile &&
        cache.act_quant_mode == q_state.act_quant_mode &&
        cache.act_quant_method == q_state.act_quant_method;
    if (cache_hit) {
      *fused_state = cache.fused_state;
      *q_size_out = cache.q_size;
      *k_size_out = cache.k_size;
      *v_size_out = cache.v_size;
      return true;
    }
    c10::optional<torch::Tensor> fused_bias = c10::nullopt;
    if (q_state.bias.has_value() || k_state.bias.has_value() || v_state.bias.has_value()) {
      std::vector<torch::Tensor> parts;
      parts.reserve(3);
      auto append_bias = [&](const BitNetModuleState& state, int64_t logical_rows) {
        if (state.bias.has_value() && state.bias.value().defined()) {
          parts.push_back(state.bias.value().to(torch::TensorOptions().device(q_state.qweight.device()).dtype(torch::kFloat32)).reshape({-1}).slice(0, 0, logical_rows));
        } else {
          parts.push_back(torch::zeros({logical_rows}, torch::TensorOptions().device(q_state.qweight.device()).dtype(torch::kFloat32)));
        }
      };
      append_bias(q_state, q_size);
      append_bias(k_state, k_size);
      append_bias(v_state, v_size);
      fused_bias = torch::cat(parts).contiguous();
    }

    BitNetModuleState out;
    out.qweight = torch::cat({q_state.qweight, k_state.qweight, v_state.qweight}, 0).contiguous();
    out.inv_scale = torch::cat({q_state.inv_scale, k_state.inv_scale, v_state.inv_scale}, 0).contiguous();
    out.bias = fused_bias;
    out.spin_enabled = false;
    out.pre_scale = q_state.pre_scale;
    out.act_quant_mode = q_state.act_quant_mode;
    out.act_quant_method = q_state.act_quant_method;
    out.act_quant_percentile = q_state.act_quant_percentile;
    out.act_quant_bits = q_state.act_quant_bits;
    out.act_scale = q_state.act_scale;
    cache.valid = true;
    cache.q_qweight = BitNetTensorPtrKey(q_state.qweight);
    cache.k_qweight = BitNetTensorPtrKey(k_state.qweight);
    cache.v_qweight = BitNetTensorPtrKey(v_state.qweight);
    cache.q_inv_scale = BitNetTensorPtrKey(q_state.inv_scale);
    cache.k_inv_scale = BitNetTensorPtrKey(k_state.inv_scale);
    cache.v_inv_scale = BitNetTensorPtrKey(v_state.inv_scale);
    cache.q_bias = BitNetTensorPtrKey(q_state.bias);
    cache.k_bias = BitNetTensorPtrKey(k_state.bias);
    cache.v_bias = BitNetTensorPtrKey(v_state.bias);
    cache.q_pre_scale = BitNetTensorPtrKey(q_state.pre_scale);
    cache.q_act_scale = BitNetTensorPtrKey(q_state.act_scale);
    cache.q_rows = q_size;
    cache.k_rows = k_size;
    cache.v_rows = v_size;
    cache.logical_in = logical_in;
    cache.act_quant_bits = q_state.act_quant_bits;
    cache.act_quant_percentile = q_state.act_quant_percentile;
    cache.act_quant_mode = q_state.act_quant_mode;
    cache.act_quant_method = q_state.act_quant_method;
    cache.fused_state = out;
    cache.q_size = q_size;
    cache.k_size = k_size;
    cache.v_size = v_size;
    *fused_state = cache.fused_state;
    *q_size_out = cache.q_size;
    *k_size_out = cache.k_size;
    *v_size_out = cache.v_size;
    return true;
  }
  const auto q_size = BitNetLayoutHeaderScalar(q_state.layout_header, 3);
  const auto k_size = BitNetLayoutHeaderScalar(k_state.layout_header, 3);
  const auto v_size = BitNetLayoutHeaderScalar(v_state.layout_header, 3);
  const auto logical_in = BitNetLayoutHeaderScalar(q_state.layout_header, 4);
  const auto padded_in = BitNetLayoutHeaderScalar(q_state.layout_header, 6);
  const auto tile_n = BitNetLayoutHeaderScalar(q_state.layout_header, 1);
  const auto tile_k = BitNetLayoutHeaderScalar(q_state.layout_header, 2);
  const auto interleave_mode = BitNetLayoutHeaderScalar(q_state.layout_header, 9);
  if (BitNetLayoutHeaderScalar(k_state.layout_header, 4) != logical_in ||
      BitNetLayoutHeaderScalar(v_state.layout_header, 4) != logical_in ||
      BitNetLayoutHeaderScalar(k_state.layout_header, 6) != padded_in ||
      BitNetLayoutHeaderScalar(v_state.layout_header, 6) != padded_in ||
      BitNetLayoutHeaderScalar(k_state.layout_header, 1) != tile_n ||
      BitNetLayoutHeaderScalar(v_state.layout_header, 1) != tile_n ||
      BitNetLayoutHeaderScalar(k_state.layout_header, 2) != tile_k ||
      BitNetLayoutHeaderScalar(v_state.layout_header, 2) != tile_k ||
      BitNetLayoutHeaderScalar(k_state.layout_header, 9) != interleave_mode ||
      BitNetLayoutHeaderScalar(v_state.layout_header, 9) != interleave_mode) {
    return false;
  }
  const auto logical_out = q_size + k_size + v_size;
  const auto padded_out = ((logical_out + 15) / 16) * 16;
  const auto packed_cols = q_state.packed_weight.size(1);
  if (k_state.packed_weight.size(1) != packed_cols || v_state.packed_weight.size(1) != packed_cols) {
    return false;
  }

  auto packed_weight = torch::full(
      {padded_out, packed_cols},
      static_cast<uint8_t>(0x55),
      torch::TensorOptions().device(q_state.packed_weight.device()).dtype(torch::kUInt8));
  int64_t cursor = 0;
  packed_weight.slice(0, cursor, cursor + q_size).copy_(q_state.packed_weight.slice(0, 0, q_size));
  cursor += q_size;
  packed_weight.slice(0, cursor, cursor + k_size).copy_(k_state.packed_weight.slice(0, 0, k_size));
  cursor += k_size;
  packed_weight.slice(0, cursor, cursor + v_size).copy_(v_state.packed_weight.slice(0, 0, v_size));

  auto scale_values = torch::cat({
      ExpandBitNetRowScales(q_state),
      ExpandBitNetRowScales(k_state),
      ExpandBitNetRowScales(v_state),
  }).to(torch::TensorOptions().device(q_state.scale_values.device()).dtype(torch::kFloat32)).contiguous();
  auto layout_header = torch::tensor(
      {
          static_cast<int32_t>(1),
          static_cast<int32_t>(tile_n),
          static_cast<int32_t>(tile_k),
          static_cast<int32_t>(logical_out),
          static_cast<int32_t>(logical_in),
          static_cast<int32_t>(padded_out),
          static_cast<int32_t>(padded_in),
          static_cast<int32_t>(2),
          static_cast<int32_t>(1),
          static_cast<int32_t>(interleave_mode),
          static_cast<int32_t>(std::max({
              BitNetLayoutHeaderScalar(q_state.layout_header, 10),
              BitNetLayoutHeaderScalar(k_state.layout_header, 10),
              BitNetLayoutHeaderScalar(v_state.layout_header, 10),
          })),
          static_cast<int32_t>(1),
          static_cast<int32_t>(0),
      },
      torch::TensorOptions().device(q_state.packed_weight.device()).dtype(torch::kInt32));
  auto segment_offsets = torch::tensor(
      {static_cast<int32_t>(0), static_cast<int32_t>(logical_out)},
      torch::TensorOptions().device(q_state.packed_weight.device()).dtype(torch::kInt32));

  c10::optional<torch::Tensor> fused_bias = c10::nullopt;
  if (q_state.bias.has_value() || k_state.bias.has_value() || v_state.bias.has_value()) {
    std::vector<torch::Tensor> parts;
    parts.reserve(3);
    auto append_bias = [&](const BitNetModuleState& state, int64_t logical_rows) {
      if (state.bias.has_value() && state.bias.value().defined()) {
        parts.push_back(state.bias.value().to(torch::TensorOptions().device(packed_weight.device()).dtype(torch::kFloat32)).reshape({-1}).slice(0, 0, logical_rows));
      } else {
        parts.push_back(torch::zeros({logical_rows}, torch::TensorOptions().device(packed_weight.device()).dtype(torch::kFloat32)));
      }
    };
    append_bias(q_state, q_size);
    append_bias(k_state, k_size);
    append_bias(v_state, v_size);
    fused_bias = torch::cat(parts).contiguous();
  }

  BitNetModuleState out;
  out.packed_weight = packed_weight.contiguous();
  out.scale_values = scale_values;
  out.layout_header = layout_header.contiguous();
  out.segment_offsets = segment_offsets.contiguous();
  out.bias = fused_bias;
  out.spin_enabled = q_state.spin_enabled;
  out.spin_signs = q_state.spin_signs;
  out.pre_scale = q_state.pre_scale;
  out.act_quant_mode = q_state.act_quant_mode;
  out.act_quant_method = q_state.act_quant_method;
  out.act_quant_percentile = q_state.act_quant_percentile;
  out.act_quant_bits = q_state.act_quant_bits;
  out.act_scale = q_state.act_scale;
  if (q_state.qweight.defined() && k_state.qweight.defined() && v_state.qweight.defined()) {
    out.qweight = torch::cat({q_state.qweight, k_state.qweight, v_state.qweight}, 0).contiguous();
  }
  if (q_state.inv_scale.defined() && k_state.inv_scale.defined() && v_state.inv_scale.defined()) {
    out.inv_scale = torch::cat({q_state.inv_scale, k_state.inv_scale, v_state.inv_scale}, 0).contiguous();
  }
  if (q_state.compute_packed_words.defined() && k_state.compute_packed_words.defined() && v_state.compute_packed_words.defined() &&
      q_state.compute_row_scales.defined() && k_state.compute_row_scales.defined() && v_state.compute_row_scales.defined()) {
    if (q_state.compute_packed_words.dim() == 3 &&
        k_state.compute_packed_words.dim() == 3 &&
        v_state.compute_packed_words.dim() == 3 &&
        q_state.compute_row_scales.dim() == 2 &&
        k_state.compute_row_scales.dim() == 2 &&
        v_state.compute_row_scales.dim() == 2) {
      const auto compute_tile_n = q_state.compute_packed_words.size(2);
      const auto compute_word_cols = q_state.compute_packed_words.size(1);
      if (compute_tile_n <= 0) {
        // Ignore invalid cached compute-packed metadata and fall back to the public packed path.
      } else if (
        k_state.compute_packed_words.size(1) == compute_word_cols &&
        v_state.compute_packed_words.size(1) == compute_word_cols &&
        k_state.compute_packed_words.size(2) == compute_tile_n &&
        v_state.compute_packed_words.size(2) == compute_tile_n &&
        q_state.compute_row_scales.size(1) == compute_tile_n &&
        k_state.compute_row_scales.size(1) == compute_tile_n &&
        v_state.compute_row_scales.size(1) == compute_tile_n) {
      auto fused_rows = torch::cat({
          FlattenComputePackedBitNetRows(q_state.compute_packed_words, q_size),
          FlattenComputePackedBitNetRows(k_state.compute_packed_words, k_size),
          FlattenComputePackedBitNetRows(v_state.compute_packed_words, v_size),
      }, 0).contiguous();
      auto fused_row_scales = torch::cat({
          FlattenComputePackedBitNetScales(q_state.compute_row_scales, q_size),
          FlattenComputePackedBitNetScales(k_state.compute_row_scales, k_size),
          FlattenComputePackedBitNetScales(v_state.compute_row_scales, v_size),
      }, 0).contiguous();
      const auto padded_compute_rows = ((logical_out + compute_tile_n - 1) / compute_tile_n) * compute_tile_n;
      if (padded_compute_rows > logical_out) {
        fused_rows = torch::cat({
            fused_rows,
            torch::full(
                {padded_compute_rows - logical_out, compute_word_cols},
                static_cast<int32_t>(0x55555555u),
                torch::TensorOptions().device(fused_rows.device()).dtype(torch::kInt32)),
        }, 0).contiguous();
        fused_row_scales = torch::cat({
            fused_row_scales,
            torch::zeros(
                {padded_compute_rows - logical_out},
                torch::TensorOptions().device(fused_row_scales.device()).dtype(torch::kFloat32)),
        }, 0).contiguous();
      }
      out.compute_packed_words = fused_rows.view({-1, compute_tile_n, compute_word_cols}).permute({0, 2, 1}).contiguous();
      out.compute_row_scales = fused_row_scales.view({-1, compute_tile_n}).contiguous();
      out.compute_tile_n = compute_tile_n;
      }
    }
  }
  if (q_state.decode_nz_masks.defined() && k_state.decode_nz_masks.defined() && v_state.decode_nz_masks.defined() &&
      q_state.decode_sign_masks.defined() && k_state.decode_sign_masks.defined() && v_state.decode_sign_masks.defined() &&
      q_state.decode_row_scales.defined() && k_state.decode_row_scales.defined() && v_state.decode_row_scales.defined()) {
    if (q_state.decode_nz_masks.dim() == 3 &&
        k_state.decode_nz_masks.dim() == 3 &&
        v_state.decode_nz_masks.dim() == 3 &&
        q_state.decode_sign_masks.dim() == 3 &&
        k_state.decode_sign_masks.dim() == 3 &&
        v_state.decode_sign_masks.dim() == 3 &&
        q_state.decode_row_scales.dim() == 2 &&
        k_state.decode_row_scales.dim() == 2 &&
        v_state.decode_row_scales.dim() == 2) {
      const auto decode_tile_n = q_state.decode_nz_masks.size(2);
      const auto decode_chunk_cols = q_state.decode_nz_masks.size(1);
      if (decode_tile_n <= 0) {
        // Ignore invalid cached decode-packed metadata and fall back to the compute-packed path.
      } else if (
        k_state.decode_nz_masks.size(1) == decode_chunk_cols &&
        v_state.decode_nz_masks.size(1) == decode_chunk_cols &&
        k_state.decode_nz_masks.size(2) == decode_tile_n &&
        v_state.decode_nz_masks.size(2) == decode_tile_n &&
        q_state.decode_sign_masks.size(1) == decode_chunk_cols &&
        k_state.decode_sign_masks.size(1) == decode_chunk_cols &&
        v_state.decode_sign_masks.size(1) == decode_chunk_cols &&
        q_state.decode_sign_masks.size(2) == decode_tile_n &&
        k_state.decode_sign_masks.size(2) == decode_tile_n &&
        v_state.decode_sign_masks.size(2) == decode_tile_n &&
        q_state.decode_row_scales.size(1) == decode_tile_n &&
        k_state.decode_row_scales.size(1) == decode_tile_n &&
        v_state.decode_row_scales.size(1) == decode_tile_n) {
      auto fused_nz_masks = torch::cat({
          FlattenDecodePackedBitNetRows(q_state.decode_nz_masks, q_size),
          FlattenDecodePackedBitNetRows(k_state.decode_nz_masks, k_size),
          FlattenDecodePackedBitNetRows(v_state.decode_nz_masks, v_size),
      }, 0).contiguous();
      auto fused_sign_masks = torch::cat({
          FlattenDecodePackedBitNetRows(q_state.decode_sign_masks, q_size),
          FlattenDecodePackedBitNetRows(k_state.decode_sign_masks, k_size),
          FlattenDecodePackedBitNetRows(v_state.decode_sign_masks, v_size),
      }, 0).contiguous();
      auto fused_row_scales = torch::cat({
          FlattenComputePackedBitNetScales(q_state.decode_row_scales, q_size),
          FlattenComputePackedBitNetScales(k_state.decode_row_scales, k_size),
          FlattenComputePackedBitNetScales(v_state.decode_row_scales, v_size),
      }, 0).contiguous();
      const auto padded_decode_rows = ((logical_out + decode_tile_n - 1) / decode_tile_n) * decode_tile_n;
      if (padded_decode_rows > logical_out) {
        auto pad_rows = torch::zeros(
            {padded_decode_rows - logical_out, decode_chunk_cols},
            torch::TensorOptions().device(fused_nz_masks.device()).dtype(torch::kInt32));
        fused_nz_masks = torch::cat({fused_nz_masks, pad_rows}, 0).contiguous();
        fused_sign_masks = torch::cat({fused_sign_masks, pad_rows.clone()}, 0).contiguous();
        fused_row_scales = torch::cat({
            fused_row_scales,
            torch::zeros(
                {padded_decode_rows - logical_out},
                torch::TensorOptions().device(fused_row_scales.device()).dtype(torch::kFloat32)),
        }, 0).contiguous();
      }
      out.decode_nz_masks = fused_nz_masks.view({-1, decode_tile_n, decode_chunk_cols}).permute({0, 2, 1}).contiguous();
      out.decode_sign_masks = fused_sign_masks.view({-1, decode_tile_n, decode_chunk_cols}).permute({0, 2, 1}).contiguous();
      out.decode_row_scales = fused_row_scales.view({-1, decode_tile_n}).contiguous();
      out.decode_tile_n = decode_tile_n;
      }
    }
  }
  *fused_state = std::move(out);
  *q_size_out = q_size;
  *k_size_out = k_size;
  *v_size_out = v_size;
  return true;
}

torch::Tensor BitNetLinearStateForward(
    const torch::Tensor& x,
    const BitNetModuleState& state,
    const c10::optional<torch::ScalarType>& out_dtype = c10::nullopt);

torch::Tensor BitNetLinearStateForward(
    const torch::Tensor& x,
    const BitNetModuleState& state,
    const c10::optional<torch::ScalarType>& out_dtype) {
  const auto resolved_out_dtype =
      out_dtype.has_value() ? out_dtype : c10::optional<torch::ScalarType>(x.scalar_type());
  if (BitNetStateUsesInt8PackedPath(state) && state.qweight.defined() && state.inv_scale.defined()) {
    return BitNetInt8LinearFromFloatForward(
        x,
        state.qweight,
        state.inv_scale,
        state.bias,
        state.pre_scale.defined() ? c10::optional<torch::Tensor>(state.pre_scale) : c10::nullopt,
        state.act_quant_mode,
        state.act_quant_method,
        state.act_quant_bits,
        state.act_quant_percentile,
        state.act_scale.defined() ? c10::optional<torch::Tensor>(state.act_scale) : c10::nullopt,
        resolved_out_dtype);
  }
#if MODEL_STACK_WITH_CUDA
  if (x.is_cuda() && !state.spin_enabled && !state.pre_scale.defined() &&
      NormalizeBackendName(state.act_quant_mode) == "none" &&
      state.compute_packed_words.defined() && state.compute_row_scales.defined()) {
    return t10::bitnet::CudaBitNetLinearForwardComputePacked(
        x,
        state.packed_weight,
        state.scale_values,
        state.layout_header,
        state.segment_offsets,
        state.compute_packed_words,
        state.compute_row_scales,
        state.decode_nz_masks,
        state.decode_sign_masks,
        state.decode_row_scales,
        state.bias,
        resolved_out_dtype);
  }
#endif
  return BitNetLinearFromFloatForward(
      x,
      state.packed_weight,
      state.scale_values,
      state.layout_header,
      state.segment_offsets,
      state.bias,
      state.spin_enabled,
      state.spin_signs.defined() ? c10::optional<torch::Tensor>(state.spin_signs) : c10::nullopt,
      state.pre_scale.defined() ? c10::optional<torch::Tensor>(state.pre_scale) : c10::nullopt,
      state.act_quant_mode,
      state.act_quant_method,
      state.act_quant_bits,
      state.act_quant_percentile,
      state.act_scale.defined() ? c10::optional<torch::Tensor>(state.act_scale) : c10::nullopt,
      resolved_out_dtype);
}

c10::optional<torch::Tensor> TryBitNetGatedInt8LinearStateForward(
    const torch::Tensor& x,
    const std::string& activation,
    const BitNetModuleState& state,
    const c10::optional<torch::ScalarType>& out_dtype = c10::nullopt) {
#if MODEL_STACK_WITH_CUDA
  if (!x.is_cuda() || !BitNetStateUsesInt8PackedPath(state) || state.spin_enabled ||
      !state.qweight.defined() || !state.inv_scale.defined() || x.dim() < 2 || (x.size(-1) % 2) != 0 ||
      !t10::bitnet::HasCudaBitNetInputFrontendKernel()) {
    return c10::nullopt;
  }
  const auto mode = NormalizeBackendName(state.act_quant_mode);
  const auto method = NormalizeBackendName(state.act_quant_method);
  const auto rows = x.size(-1) > 0 ? x.numel() / x.size(-1) : 0;
  const bool static_supported = mode == "static_int8" && state.act_scale.defined();
  const bool dynamic_supported =
      mode == "dynamic_int8" && rows > 0 && rows <= 8 &&
      (method.empty() || method == "absmax" || method == "mse");
  const auto resolved_out_dtype =
      out_dtype.has_value() ? out_dtype : c10::optional<torch::ScalarType>(x.scalar_type());
  if (!static_supported && !dynamic_supported) {
    return c10::nullopt;
  }
  auto quantized = t10::bitnet::CudaBitNetQuantizeGatedActivationInt8CodesForward(
      x,
      activation,
      state.pre_scale.defined() ? c10::optional<torch::Tensor>(state.pre_scale) : c10::nullopt,
      state.act_quant_mode,
      state.act_quant_method,
      state.act_quant_bits,
      state.act_scale.defined() ? c10::optional<torch::Tensor>(state.act_scale) : c10::nullopt);
  return Int8LinearForward(
      std::get<0>(quantized),
      std::get<1>(quantized),
      state.qweight,
      state.inv_scale,
      state.bias,
      resolved_out_dtype);
#else
  (void)x;
  (void)activation;
  (void)state;
  (void)out_dtype;
  return c10::nullopt;
#endif
}

torch::Tensor PythonLinearModuleForward(
    const torch::Tensor& x,
    const py::object& module,
    const std::string& backend) {
  if (PyCallableAttr(module, "runtime_linear")) {
    return py::cast<torch::Tensor>(module.attr("runtime_linear")(x, "backend"_a = backend));
  }
  return LinearForward(x, TensorAttr(module, "weight"), TensorAttrOptional(module, "bias"), backend);
}

bool CanUseDenseBitNetDecodePolicy(
    const torch::Tensor& x,
    const BitNetModuleState& state);

bool CanUseCutlassDirectBitNetPrefillPolicy(
    const torch::Tensor& x,
    const BitNetModuleState& state);

bool CanUseModuleBitNetRuntimePolicy(
    const torch::Tensor& x,
    const BitNetModuleState& state);

bool PreferDirectBitNetRuntimeLinear(
    const torch::Tensor& x,
    const py::object& module,
    const BitNetModuleState& state) {
  if (!PyCallableAttr(module, "runtime_linear")) {
    return false;
  }
  // Match the Python/runtime policy: only validated dynamic-int8 routes may
  // escape to module runtime_linear() here. Sub-8-bit dynamic decode stays on
  // the native packed/int8 path until it has its own tuned policy.
  return CanUseModuleBitNetRuntimePolicy(x, state);
}

bool PreferDirectBitNetRuntimeLinearQkv(
    const torch::Tensor& x,
    const py::object& q_module,
    const BitNetModuleState& q_state,
    const py::object& k_module,
    const BitNetModuleState& k_state,
    const py::object& v_module,
    const BitNetModuleState& v_state) {
  return PreferDirectBitNetRuntimeLinear(x, q_module, q_state) &&
      PreferDirectBitNetRuntimeLinear(x, k_module, k_state) &&
      PreferDirectBitNetRuntimeLinear(x, v_module, v_state);
}

torch::Tensor LinearLikeModuleForward(
    const torch::Tensor& x,
    const py::object& module,
    const std::string& backend) {
  BitNetModuleState bitnet_state;
  if (BitNetModuleDirectSupported(module, &bitnet_state)) {
    if (CanUseCutlassDirectBitNetPrefillPolicy(x, bitnet_state)) {
      return BitNetLinearStateForward(x, bitnet_state);
    }
    if (PreferDirectBitNetRuntimeLinear(x, module, bitnet_state)) {
      return PythonLinearModuleForward(x, module, backend);
    }
    return BitNetLinearStateForward(x, bitnet_state);
  }
  if (ModuleHasRuntimeLinear(module)) {
    return PythonLinearModuleForward(x, module, backend);
  }
  return LinearForward(x, TensorAttr(module, "weight"), TensorAttrOptional(module, "bias"), backend);
}

bool AttentionQkvSupportsPackedBitNet(const py::object& attn) {
  for (const char* name : {"w_q", "w_k", "w_v"}) {
    if (!ModuleSupportsPackedBackend(attn.attr(name), "bitnet")) {
      return false;
    }
  }
  return true;
}

bool AttentionUsesBitNetInt8AttentionCore(const py::object& attn) {
  BitNetModuleState q_state;
  BitNetModuleState k_state;
  BitNetModuleState v_state;
  return BitNetModuleDirectSupported(attn.attr("w_q"), &q_state) &&
      BitNetModuleDirectSupported(attn.attr("w_k"), &k_state) &&
      BitNetModuleDirectSupported(attn.attr("w_v"), &v_state) &&
      BitNetStateUsesInt8PackedPath(q_state) &&
      BitNetStateUsesInt8PackedPath(k_state) &&
      BitNetStateUsesInt8PackedPath(v_state);
}

std::vector<torch::Tensor> AttentionPackedBitNetQkvForwardFromStates(
    const torch::Tensor& x,
    const BitNetModuleState& q_state,
    const BitNetModuleState& k_state,
    const BitNetModuleState& v_state,
    int64_t q_heads,
    int64_t kv_heads) {
  if (BitNetStateUsesInt8PackedPath(q_state) &&
      BitNetStateUsesInt8PackedPath(k_state) &&
      BitNetStateUsesInt8PackedPath(v_state)) {
    BitNetModuleState fused_state;
    int64_t q_size = 0;
    int64_t k_size = 0;
    int64_t v_size = 0;
    if (TryFuseBitNetQkvStates(q_state, k_state, v_state, &fused_state, &q_size, &k_size, &v_size) &&
        fused_state.qweight.defined() && fused_state.inv_scale.defined()) {
      return BitNetInt8FusedQkvPackedHeadsProjectionForward(
          x,
          fused_state.qweight,
          fused_state.inv_scale,
          fused_state.bias,
          fused_state.pre_scale.defined() ? c10::optional<torch::Tensor>(fused_state.pre_scale) : c10::nullopt,
          fused_state.act_quant_mode,
          fused_state.act_quant_method,
          fused_state.act_quant_bits,
          fused_state.act_quant_percentile,
          fused_state.act_scale.defined() ? c10::optional<torch::Tensor>(fused_state.act_scale) : c10::nullopt,
          q_size,
          k_size,
          v_size,
          q_heads,
          kv_heads,
          c10::optional<torch::ScalarType>(x.scalar_type()));
    }
    auto q = BitNetLinearStateForward(x, q_state);
    auto k = BitNetLinearStateForward(x, k_state);
    auto v = BitNetLinearStateForward(x, v_state);
    return {
        SplitHeadsForward(q, q_heads),
        SplitHeadsForward(k, kv_heads),
        SplitHeadsForward(v, kv_heads),
    };
  }
  BitNetModuleState fused_state;
  int64_t q_size = 0;
  int64_t k_size = 0;
  int64_t v_size = 0;
  if (TryFuseBitNetQkvStates(q_state, k_state, v_state, &fused_state, &q_size, &k_size, &v_size)) {
    if (fused_state.compute_packed_words.defined() && fused_state.compute_row_scales.defined()) {
      auto projected = BitNetLinearStateForward(x, fused_state);
      return SplitProjectedBitNetQkv(projected, q_size, k_size, v_size, q_heads, kv_heads);
    }
    auto x_local = BitNetTransformInputFromState(x, fused_state);
    return BitNetFusedQkvPackedHeadsProjectionForward(
        x_local,
        fused_state.packed_weight,
        fused_state.scale_values,
        fused_state.layout_header,
        fused_state.segment_offsets,
        fused_state.bias,
        q_size,
        k_size,
        v_size,
        q_heads,
        kv_heads);
  }
  auto q_x = BitNetTransformInputFromState(x, q_state);
  auto k_x = BitNetTransformInputFromState(x, k_state);
  auto v_x = BitNetTransformInputFromState(x, v_state);
  return BitNetQkvPackedHeadsProjectionForward(
      q_x,
      q_state.packed_weight,
      q_state.scale_values,
      q_state.layout_header,
      q_state.segment_offsets,
      q_state.bias,
      k_x,
      k_state.packed_weight,
      k_state.scale_values,
      k_state.layout_header,
      k_state.segment_offsets,
      k_state.bias,
      v_x,
      v_state.packed_weight,
      v_state.scale_values,
      v_state.layout_header,
      v_state.segment_offsets,
      v_state.bias,
      q_heads,
      kv_heads,
      c10::nullopt);
}

std::vector<torch::Tensor> AttentionPackedBitNetQkvForward(
    const torch::Tensor& x,
    const py::object& attn,
    int64_t q_heads,
    int64_t kv_heads) {
  BitNetModuleState q_state;
  BitNetModuleState k_state;
  BitNetModuleState v_state;
  if (BitNetModuleDirectSupported(attn.attr("w_q"), &q_state) &&
      BitNetModuleDirectSupported(attn.attr("w_k"), &k_state) &&
      BitNetModuleDirectSupported(attn.attr("w_v"), &v_state)) {
    return AttentionPackedBitNetQkvForwardFromStates(x, q_state, k_state, v_state, q_heads, kv_heads);
  }
  auto runtime_ops = RuntimeOpsModule();
  auto spec = runtime_ops.attr("resolve_packed_qkv_module_spec")(
      attn.attr("w_q"),
      attn.attr("w_k"),
      attn.attr("w_v"),
      "backend"_a = "bitnet",
      "reference"_a = x);
  auto out = py::cast<py::tuple>(runtime_ops.attr("qkv_packed_spec_heads_projection")(
      x,
      spec,
      "q_heads"_a = q_heads,
      "kv_heads"_a = kv_heads,
      "backend"_a = "bitnet"));
  return {
      py::cast<torch::Tensor>(out[0]),
      py::cast<torch::Tensor>(out[1]),
      py::cast<torch::Tensor>(out[2]),
  };
}

torch::Tensor AttentionPackedBitNetOutputForward(
    const torch::Tensor& x,
    const py::object& module) {
  auto merged = MergeHeadsForward(x);
  BitNetModuleState state;
  if (BitNetModuleDirectSupported(module, &state)) {
    if (PreferDirectBitNetRuntimeLinear(merged, module, state)) {
      return PythonLinearModuleForward(merged, module, "auto");
    }
    return BitNetLinearStateForward(merged, state);
  }
  auto runtime_ops = RuntimeOpsModule();
  auto spec = runtime_ops.attr("resolve_packed_linear_module_spec")(
      module,
      "backend"_a = "bitnet",
      "reference"_a = merged);
  return py::cast<torch::Tensor>(
      runtime_ops.attr("head_output_packed_projection")(x, spec, "backend"_a = "bitnet"));
}

bool IsSupportedNormModule(const py::object& norm) {
  const auto name = PyTypeName(norm);
  return name == "RMSNorm" || name == "LayerNorm";
}

torch::Tensor ApplyNormModuleForward(const torch::Tensor& x, const py::object& norm) {
  const auto name = PyTypeName(norm);
  if (name == "RMSNorm") {
    return RmsNormForward(x, TensorAttrOptional(norm, "weight"), PyFloatAttr(norm, "eps", 1e-6));
  }
  if (name == "LayerNorm") {
    return LayerNormForward(x, TensorAttrOptional(norm, "weight"), TensorAttrOptional(norm, "bias"), PyFloatAttr(norm, "eps", 1e-5));
  }
  throw std::runtime_error("unsupported norm module for native causal executor");
}

std::vector<torch::Tensor> AddNormModuleForward(const torch::Tensor& x, const torch::Tensor& update, const py::object& norm, double residual_scale) {
  const auto name = PyTypeName(norm);
  if (name == "RMSNorm") {
    return AddRmsNormForward(x, update, TensorAttrOptional(norm, "weight"), residual_scale, PyFloatAttr(norm, "eps", 1e-6));
  }
  if (name == "LayerNorm") {
    return AddLayerNormForward(
        x,
        update,
        TensorAttrOptional(norm, "weight"),
        TensorAttrOptional(norm, "bias"),
        residual_scale,
        PyFloatAttr(norm, "eps", 1e-5));
  }
  throw std::runtime_error("unsupported add+norm module for native causal executor");
}

bool HasNativeCacheSupport(const py::object& cache) {
  if (cache.is_none()) {
    return true;
  }
  auto native_cache = PyAttrOrNone(cache, "_native_cache");
  if (!native_cache.is_none()) {
    return true;
  }
  auto native_layers = PyAttrOrNone(cache, "_native_layers");
  return !native_layers.is_none();
}

bool IsSupportedAttentionMask(const py::object& attention_mask) {
  if (attention_mask.is_none()) {
    return true;
  }
  try {
    auto mask = py::cast<torch::Tensor>(attention_mask);
    return mask.defined() && mask.dim() >= 2 && mask.dim() <= 4;
  } catch (...) {
    return false;
  }
}

bool IsTokenAttentionMaskTensor(const torch::Tensor& attention_mask, int64_t batch_size, int64_t seq_len) {
  return attention_mask.defined() && attention_mask.dim() == 2 && attention_mask.size(0) == batch_size && attention_mask.size(1) == seq_len;
}

std::string ResolveAttentionMaskMode(
    const py::object& attention_mask,
    int64_t batch_size,
    int64_t seq_len) {
  if (attention_mask.is_none()) {
    return "none";
  }
  try {
    auto mask = py::cast<torch::Tensor>(attention_mask);
    return IsTokenAttentionMaskTensor(mask, batch_size, seq_len) ? std::string("token") : std::string("explicit");
  } catch (...) {
    return "python";
  }
}

torch::Tensor PrepareExplicitAttentionMaskForHeads(
    const torch::Tensor& mask,
    int64_t batch_size,
    int64_t num_heads,
    int64_t tgt_len,
    int64_t src_len) {
  TORCH_CHECK(mask.defined(), "prepare_explicit_attention_mask_for_heads: mask must be defined");
  TORCH_CHECK(mask.dim() >= 2 && mask.dim() <= 4,
              "prepare_explicit_attention_mask_for_heads: mask rank must be between 2 and 4");
  if (mask.dim() == 2) {
    if (mask.size(0) == batch_size && mask.size(1) == src_len) {
      auto padding_mask = mask.scalar_type() == torch::kBool ? mask : mask.eq(0);
      return padding_mask.view({batch_size, 1, 1, src_len}).expand({batch_size, num_heads, tgt_len, src_len});
    }
    TORCH_CHECK(mask.size(0) == tgt_len && mask.size(1) == src_len,
                "prepare_explicit_attention_mask_for_heads: 2D mask must have shape (B,S) or (T,S)");
    return mask.view({1, 1, tgt_len, src_len}).expand({batch_size, num_heads, tgt_len, src_len});
  }
  if (mask.dim() == 3) {
    TORCH_CHECK(mask.size(0) == batch_size && mask.size(1) == tgt_len && mask.size(2) == src_len,
                "prepare_explicit_attention_mask_for_heads: 3D mask must have shape (B,T,S)");
    return mask.unsqueeze(1).expand({batch_size, num_heads, tgt_len, src_len});
  }
  TORCH_CHECK(mask.size(-2) == tgt_len && mask.size(-1) == src_len,
              "prepare_explicit_attention_mask_for_heads: 4D mask must end in (T,S)");
  TORCH_CHECK(mask.size(0) == 1 || mask.size(0) == batch_size,
              "prepare_explicit_attention_mask_for_heads: 4D batch dim must be 1 or batch_size");
  TORCH_CHECK(mask.size(1) == 1 || mask.size(1) == num_heads,
              "prepare_explicit_attention_mask_for_heads: 4D head dim must be 1 or num_heads");
  return mask.expand({batch_size, num_heads, tgt_len, src_len});
}

torch::Tensor AppendExplicitDecodeAttentionMaskForward(
    const torch::Tensor& attention_mask,
    const c10::optional<torch::Tensor>& row_ids) {
  TORCH_CHECK(attention_mask.defined(), "append_explicit_decode_attention_mask_forward: attention_mask must be defined");
  TORCH_CHECK(attention_mask.dim() >= 2 && attention_mask.dim() <= 4,
              "append_explicit_decode_attention_mask_forward: attention_mask rank must be between 2 and 4");
  auto mask = attention_mask;
  if (row_ids.has_value() && row_ids.value().defined()) {
    auto ids = row_ids.value().to(torch::kLong).contiguous().view({-1});
    if (mask.dim() == 3 || mask.dim() == 4) {
      if (mask.size(0) == 1 && ids.numel() > 1) {
        std::vector<int64_t> expanded(mask.sizes().begin(), mask.sizes().end());
        expanded[0] = ids.numel();
        mask = mask.expand(expanded);
      } else if (mask.size(0) > 1) {
        mask = mask.index_select(0, ids);
      }
    }
  }
  const auto src_len = mask.size(-1) + 1;
  if (mask.dim() == 2) {
    return torch::zeros({1, src_len}, mask.options());
  }
  if (mask.dim() == 3) {
    return torch::zeros({mask.size(0), 1, src_len}, mask.options());
  }
  return torch::zeros({mask.size(0), mask.size(1), 1, src_len}, mask.options());
}

bool IsSupportedAttentionModule(const py::object& attn) {
  for (const char* name : {"w_q", "w_k", "w_v", "w_o"}) {
    if (!py::hasattr(attn, name)) {
      return false;
    }
  }
  return true;
}

bool IsSupportedMlpModule(const py::object& mlp) {
  return py::hasattr(mlp, "w_in") && py::hasattr(mlp, "w_out") && py::hasattr(mlp, "activation_name") && py::hasattr(mlp, "gated");
}

bool IsSupportedMoEMlpModule(const py::object& moe) {
  if (!py::hasattr(moe, "router") || !py::hasattr(moe, "experts") || !py::hasattr(moe, "num_experts") || !py::hasattr(moe, "k")) {
    return false;
  }
  auto router = moe.attr("router");
  if (!IsSupportedLinearLikeModule(router)) {
    return false;
  }
  const auto num_experts = PyIntAttr(moe, "num_experts", -1);
  const auto k = PyIntAttr(moe, "k", -1);
  if (num_experts <= 0 || k <= 0 || k > num_experts) {
    return false;
  }
  auto experts = moe.attr("experts");
  if (py::len(experts) != num_experts) {
    return false;
  }
  for (auto expert_handle : experts) {
    py::object expert = py::reinterpret_borrow<py::object>(expert_handle);
    if (!IsSupportedMlpModule(expert)) {
      return false;
    }
  }
  return true;
}

bool ModuleUsesBitNetInt8PackedPath(const py::object& module) {
  BitNetModuleState state;
  return BitNetModuleDirectSupported(module, &state) && BitNetStateUsesInt8PackedPath(state);
}

bool ModelUsesBitNetInt8PackedPath(const py::object& model) {
  if (py::hasattr(model, "lm_head") && ModuleUsesBitNetInt8PackedPath(model.attr("lm_head"))) {
    return true;
  }
  for (auto block_handle : model.attr("blocks")) {
    py::object block = py::reinterpret_borrow<py::object>(block_handle);
    if (py::hasattr(block, "attn")) {
      auto attn = block.attr("attn");
      for (const char* name : {"w_q", "w_k", "w_v", "w_o"}) {
        if (py::hasattr(attn, name) && ModuleUsesBitNetInt8PackedPath(attn.attr(name))) {
          return true;
        }
      }
    }
    if (py::hasattr(block, "mlp")) {
      auto mlp = block.attr("mlp");
      for (const char* name : {"w_in", "w_out"}) {
        if (py::hasattr(mlp, name) && ModuleUsesBitNetInt8PackedPath(mlp.attr(name))) {
          return true;
        }
      }
    }
    if (py::hasattr(block, "moe")) {
      auto moe = block.attr("moe");
      if (py::hasattr(moe, "router") && ModuleUsesBitNetInt8PackedPath(moe.attr("router"))) {
        return true;
      }
      if (py::hasattr(moe, "experts")) {
        for (auto expert_handle : moe.attr("experts")) {
          py::object expert = py::reinterpret_borrow<py::object>(expert_handle);
          for (const char* name : {"w_in", "w_out"}) {
            if (py::hasattr(expert, name) && ModuleUsesBitNetInt8PackedPath(expert.attr(name))) {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}

bool BlockUsesAttentionBiases(const py::object& block) {
  if (!py::hasattr(block, "bc")) {
    return false;
  }
  auto bc = block.attr("bc");
  if (PyBoolAttr(bc, "use_alibi", false)) {
    return true;
  }
  auto rpb_table = PyAttrOrNone(block, "rpb_table");
  return !rpb_table.is_none();
}

bool ModelUsesAttentionBiases(const py::object& model) {
  if (!py::hasattr(model, "blocks")) {
    return false;
  }
  for (auto block_handle : model.attr("blocks")) {
    py::object block = py::reinterpret_borrow<py::object>(block_handle);
    if (BlockUsesAttentionBiases(block)) {
      return true;
    }
  }
  return false;
}

struct PreparedLinearLikeModule {
  py::object module = py::none();
  bool direct_bitnet = false;
  BitNetModuleState bitnet_state;
  bool has_runtime_linear = false;
  bool supports_bitnet_packed = false;
  torch::Tensor dense_decode_weight;
  c10::optional<torch::Tensor> dense_decode_bias = c10::nullopt;
};

bool CanUseDenseBitNetDecodePolicy(
    const torch::Tensor& x,
    const BitNetModuleState& state) {
  if (!x.is_cuda() || x.dim() < 2) {
    return false;
  }
  if (state.spin_enabled || state.pre_scale.defined()) {
    return false;
  }
  if (NormalizeBackendName(state.act_quant_mode) != "dynamic_int8" || state.act_quant_bits != 8) {
    return false;
  }
  const auto rows = x.size(-1) > 0 ? x.numel() / x.size(-1) : 0;
  if (rows <= 0 || rows > 8) {
    return false;
  }
  const auto out_features = state.qweight.defined() && state.qweight.dim() == 2 ? state.qweight.size(0) : 0;
  return x.size(-1) >= 2048 || out_features >= 2048;
}

bool CanUseHopperDynamicInt8BitNetPrefillPolicy(
    const torch::Tensor& x,
    const BitNetModuleState& state) {
  if (!x.is_cuda() || x.dim() < 2) {
    return false;
  }
  if (state.spin_enabled || state.pre_scale.defined()) {
    return false;
  }
  if (!state.qweight.defined() || state.qweight.dim() != 2) {
    return false;
  }
  if (NormalizeBackendName(state.act_quant_mode) != "dynamic_int8" ||
      state.act_quant_bits < 2 || state.act_quant_bits > 8) {
    return false;
  }
  const auto rows = x.size(-1) > 0 ? x.numel() / x.size(-1) : 0;
  if (rows <= 8) {
    return false;
  }
#if MODEL_STACK_WITH_CUDA
  return t10::cuda::DeviceIsSm90OrLater(x);
#else
  return false;
#endif
}

bool CanUseCutlassDirectBitNetPrefillPolicy(
    const torch::Tensor& x,
    const BitNetModuleState& state) {
  if (!EnvFlagEnabled("MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED")) {
    return false;
  }
  if (!CanUseHopperDynamicInt8BitNetPrefillPolicy(x, state)) {
    return false;
  }
  const auto in_features = state.qweight.size(1);
  const auto out_features = state.qweight.size(0);
  return (in_features == 1024 && out_features == 3072) ||
      (in_features == 3072 && out_features == 1024);
}

bool CanUseDenseBitNetPrefillPolicy(
    const torch::Tensor& x,
    const BitNetModuleState& state) {
  if (!CanUseHopperDynamicInt8BitNetPrefillPolicy(x, state)) {
    return false;
  }
  if (state.act_quant_bits != 8) {
    return false;
  }
  const auto out_features = state.qweight.size(0);
  if (out_features >= 32768) {
    return false;
  }
  return !CanUseCutlassDirectBitNetPrefillPolicy(x, state);
}

bool CanUseModuleBitNetRuntimePolicy(
    const torch::Tensor& x,
    const BitNetModuleState& state) {
  return CanUseDenseBitNetDecodePolicy(x, state) ||
      CanUseDenseBitNetPrefillPolicy(x, state) ||
      CanUseCutlassDirectBitNetPrefillPolicy(x, state);
}

bool CanUsePreparedDenseBitNetDecodeWeight(
    const torch::Tensor& x,
    const PreparedLinearLikeModule& prepared) {
  if (!prepared.direct_bitnet || !prepared.dense_decode_weight.defined() || !x.is_cuda() || x.dim() < 2) {
    return false;
  }
  const auto rows = x.size(-1) > 0 ? x.numel() / x.size(-1) : 0;
  if (rows > 0 && rows <= 8) {
    return true;
  }
  return CanUseDenseBitNetPrefillPolicy(x, prepared.bitnet_state);
}

void MaybePrepareDenseBitNetLinearDecodeCache(PreparedLinearLikeModule* prepared) {
  if (prepared == nullptr || !prepared->direct_bitnet) {
    return;
  }
  if (!HopperDenseBitNetDecodeLinearCacheEnabled()) {
    return;
  }
  const auto& state = prepared->bitnet_state;
  if (!state.qweight.defined() || !state.inv_scale.defined() ||
      state.spin_enabled || state.pre_scale.defined() ||
      NormalizeBackendName(state.act_quant_mode) != "dynamic_int8" ||
      state.act_quant_bits != 8 ||
      !state.qweight.is_cuda() ||
      !t10::cuda::DeviceIsSm90OrLater(state.qweight)) {
    return;
  }
  const auto out_features = state.qweight.dim() == 2 ? state.qweight.size(0) : 0;
  if (out_features >= 32768) {
    return;
  }
  prepared->dense_decode_weight =
      DenseBitNetInt8WeightCached(state.qweight, state.inv_scale, torch::kBFloat16);
  if (state.bias.has_value() && state.bias.value().defined()) {
    prepared->dense_decode_bias = state.bias.value().to(state.qweight.device(), torch::kBFloat16).contiguous();
  }
}

bool PrepareLinearLikeModule(const py::object& module, PreparedLinearLikeModule* out) {
  if (out == nullptr) {
    return false;
  }
  PreparedLinearLikeModule prepared;
  prepared.module = module;
  prepared.has_runtime_linear = ModuleHasRuntimeLinear(module);
  prepared.supports_bitnet_packed = ModuleSupportsPackedBackend(module, "bitnet");
  if (BitNetModuleDirectSupported(module, &prepared.bitnet_state)) {
    prepared.direct_bitnet = true;
    MaybePrepareDenseBitNetLinearDecodeCache(&prepared);
    *out = std::move(prepared);
    return true;
  }
  if (prepared.has_runtime_linear || py::hasattr(module, "weight")) {
    *out = std::move(prepared);
    return true;
  }
  return false;
}

torch::Tensor PreparedLinearLikeForward(
    const torch::Tensor& x,
    const PreparedLinearLikeModule& prepared,
    const std::string& backend) {
  if (prepared.direct_bitnet &&
      CanUseCutlassDirectBitNetPrefillPolicy(x, prepared.bitnet_state)) {
    return BitNetLinearStateForward(x, prepared.bitnet_state);
  }
  if (CanUsePreparedDenseBitNetDecodeWeight(x, prepared)) {
    return LinearForward(
        x,
        prepared.dense_decode_weight,
        prepared.dense_decode_bias,
        backend);
  }
  if (prepared.direct_bitnet &&
      PreferDirectBitNetRuntimeLinear(x, prepared.module, prepared.bitnet_state)) {
    return PythonLinearModuleForward(x, prepared.module, backend);
  }
  if (prepared.direct_bitnet) {
    return BitNetLinearStateForward(x, prepared.bitnet_state);
  }
  if (prepared.has_runtime_linear) {
    return PythonLinearModuleForward(x, prepared.module, backend);
  }
  return LinearForward(
      x,
      TensorAttr(prepared.module, "weight"),
      TensorAttrOptional(prepared.module, "bias"),
      backend);
}

bool CanUseFusedRmsNormBitNetLinearStateForward(
    const torch::Tensor& x,
    const py::object& norm,
    const BitNetModuleState& state) {
#if MODEL_STACK_WITH_CUDA
  const auto rows = x.size(-1) > 0 ? x.numel() / x.size(-1) : 0;
  const bool row1_decode_layout =
      rows == 1 &&
      state.decode_nz_masks.defined() &&
      state.decode_sign_masks.defined() &&
      state.decode_row_scales.defined();
  const bool compute_decode_layout =
      rows > 1 &&
      BitNetDecodeFusedNormRowsEnabled(x) &&
      state.compute_packed_words.defined() &&
      state.compute_row_scales.defined();
  return x.is_cuda() &&
      rows > 0 &&
      rows <= 8 &&
      PyTypeName(norm) == "RMSNorm" &&
      !state.spin_enabled &&
      !state.pre_scale.defined() &&
      NormalizeBackendName(state.act_quant_mode) == "none" &&
      (row1_decode_layout || compute_decode_layout);
#else
  (void)x;
  (void)norm;
  (void)state;
  return false;
#endif
}

torch::Tensor FusedRmsNormBitNetLinearStateForward(
    const torch::Tensor& x,
    const py::object& norm,
    const BitNetModuleState& state,
    const c10::optional<torch::ScalarType>& out_dtype = c10::nullopt) {
  const auto resolved_out_dtype =
      out_dtype.has_value() ? out_dtype : c10::optional<torch::ScalarType>(x.scalar_type());
#if MODEL_STACK_WITH_CUDA
  if (CanUseFusedRmsNormBitNetLinearStateForward(x, norm, state)) {
    return t10::bitnet::CudaBitNetRmsNormLinearForwardDecodeRows(
        x,
        TensorAttrOptional(norm, "weight"),
        PyFloatAttr(norm, "eps", 1e-6),
        state.layout_header,
        state.compute_packed_words,
        state.compute_row_scales,
        state.decode_nz_masks,
        state.decode_sign_masks,
        state.decode_row_scales,
        state.bias,
        resolved_out_dtype);
  }
#endif
  return BitNetLinearStateForward(ApplyNormModuleForward(x, norm), state, resolved_out_dtype);
}

bool CanUseFusedAddRmsNormBitNetLinearStateForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const py::object& norm,
    const BitNetModuleState& state) {
#if MODEL_STACK_WITH_CUDA
  return update.defined() &&
      update.is_cuda() &&
      update.sizes() == x.sizes() &&
      update.scalar_type() == x.scalar_type() &&
      CanUseFusedRmsNormBitNetLinearStateForward(x, norm, state);
#else
  (void)x;
  (void)update;
  (void)norm;
  (void)state;
  return false;
#endif
}

std::vector<torch::Tensor> FusedAddRmsNormBitNetLinearStateForward(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const py::object& norm,
    double residual_scale,
    const BitNetModuleState& state,
    const c10::optional<torch::ScalarType>& out_dtype = c10::nullopt) {
  const auto resolved_out_dtype =
      out_dtype.has_value() ? out_dtype : c10::optional<torch::ScalarType>(x.scalar_type());
#if MODEL_STACK_WITH_CUDA
  if (CanUseFusedAddRmsNormBitNetLinearStateForward(x, update, norm, state)) {
    return t10::bitnet::CudaBitNetAddRmsNormLinearForwardDecodeRows(
        x,
        update,
        TensorAttrOptional(norm, "weight"),
        residual_scale,
        PyFloatAttr(norm, "eps", 1e-6),
        state.layout_header,
        state.compute_packed_words,
        state.compute_row_scales,
        state.decode_nz_masks,
        state.decode_sign_masks,
        state.decode_row_scales,
        state.bias,
        resolved_out_dtype);
  }
#endif
  auto add_norm = AddNormModuleForward(x, update, norm, residual_scale);
  return {add_norm[0], BitNetLinearStateForward(add_norm[1], state, resolved_out_dtype)};
}

struct PreparedAttentionModule {
  py::object attn = py::none();
  PreparedLinearLikeModule w_q;
  PreparedLinearLikeModule w_k;
  PreparedLinearLikeModule w_v;
  PreparedLinearLikeModule w_o;
  int64_t q_heads = 0;
  int64_t kv_heads = 0;
  int64_t head_dim = 0;
  double attention_scale = 1.0;
  bool use_rope = false;
  double rope_theta = 1e6;
  double rope_attention_scaling = 1.0;
  std::string rope_scaling_type;
  bool has_rope_scaling_factor = false;
  double rope_scaling_factor = 1.0;
  bool has_max_position_embeddings = false;
  int64_t max_position_embeddings = 0;
  bool has_rope_scaling_original_max_position_embeddings = false;
  int64_t rope_scaling_original_max_position_embeddings = 0;
  bool has_rope_scaling_low_freq_factor = false;
  double rope_scaling_low_freq_factor = 0.0;
  bool has_rope_scaling_high_freq_factor = false;
  double rope_scaling_high_freq_factor = 0.0;
  bool direct_bitnet_qkv = false;
  bool direct_fused_bitnet_qkv = false;
  BitNetModuleState fused_bitnet_qkv_state;
  int64_t fused_q_size = 0;
  int64_t fused_k_size = 0;
  int64_t fused_v_size = 0;
  torch::Tensor fused_dense_qkv_weight;
  c10::optional<torch::Tensor> fused_dense_qkv_bias = c10::nullopt;
  bool use_bitnet_int8_attention_core = false;
  bool use_packed_bitnet_qkv = false;
};

std::pair<double, double> ResolvePreparedAttentionRopeParameters(
    const PreparedAttentionModule& prepared,
    int64_t seq_len) {
  c10::optional<int64_t> original_max_position_embeddings = c10::nullopt;
  if (prepared.has_rope_scaling_original_max_position_embeddings) {
    original_max_position_embeddings = prepared.rope_scaling_original_max_position_embeddings;
  } else if (prepared.has_max_position_embeddings) {
    original_max_position_embeddings = prepared.max_position_embeddings;
  }
  c10::optional<double> low_freq_factor = c10::nullopt;
  if (prepared.has_rope_scaling_low_freq_factor) {
    low_freq_factor = prepared.rope_scaling_low_freq_factor;
  }
  c10::optional<double> high_freq_factor = c10::nullopt;
  if (prepared.has_rope_scaling_high_freq_factor) {
    high_freq_factor = prepared.rope_scaling_high_freq_factor;
  }
  return ResolveNativeRopeParameters(
      seq_len,
      prepared.head_dim,
      prepared.rope_theta,
      prepared.rope_attention_scaling,
      prepared.rope_scaling_type,
      prepared.has_rope_scaling_factor,
      prepared.rope_scaling_factor,
      original_max_position_embeddings,
      low_freq_factor,
      high_freq_factor);
}

bool PrepareAttentionModule(const py::object& attn, PreparedAttentionModule* out) {
  if (out == nullptr) {
    return false;
  }
  PreparedAttentionModule prepared;
  prepared.attn = attn;
  prepared.q_heads = PyIntAttr(attn, "n_heads", 0);
  prepared.kv_heads = PyIntAttr(attn, "n_kv_heads", prepared.q_heads);
  prepared.head_dim = PyIntAttr(attn, "head_dim", 0);
  if (prepared.q_heads <= 0 || prepared.kv_heads <= 0 || prepared.head_dim <= 0) {
    return false;
  }
  prepared.attention_scale = PyFloatAttr(attn, "scaling", std::pow(static_cast<double>(prepared.head_dim), -0.5));
  prepared.use_rope = PyBoolAttr(attn, "use_rope", false);
  prepared.rope_theta = PyFloatAttr(attn, "rope_theta", 1e6);
  prepared.rope_attention_scaling = PyFloatAttr(attn, "rope_attention_scaling", 1.0);
  prepared.rope_scaling_type = PyStringAttr(attn, "rope_scaling_type", "");
  auto rope_scaling_factor_obj = PyAttrOrNone(attn, "rope_scaling_factor");
  if (!rope_scaling_factor_obj.is_none()) {
    prepared.has_rope_scaling_factor = true;
    prepared.rope_scaling_factor = py::cast<double>(rope_scaling_factor_obj);
  }
  auto max_position_embeddings_obj = PyAttrOrNone(attn, "max_position_embeddings");
  if (!max_position_embeddings_obj.is_none()) {
    try {
      prepared.has_max_position_embeddings = true;
      prepared.max_position_embeddings = py::cast<int64_t>(max_position_embeddings_obj);
    } catch (...) {
      prepared.has_max_position_embeddings = false;
      prepared.max_position_embeddings = 0;
    }
  }
  auto rope_scaling_original_max_position_embeddings_obj =
      PyAttrOrNone(attn, "rope_scaling_original_max_position_embeddings");
  if (!rope_scaling_original_max_position_embeddings_obj.is_none()) {
    try {
      prepared.has_rope_scaling_original_max_position_embeddings = true;
      prepared.rope_scaling_original_max_position_embeddings =
          py::cast<int64_t>(rope_scaling_original_max_position_embeddings_obj);
    } catch (...) {
      prepared.has_rope_scaling_original_max_position_embeddings = false;
      prepared.rope_scaling_original_max_position_embeddings = 0;
    }
  }
  auto rope_scaling_low_freq_factor_obj = PyAttrOrNone(attn, "rope_scaling_low_freq_factor");
  if (!rope_scaling_low_freq_factor_obj.is_none()) {
    try {
      prepared.has_rope_scaling_low_freq_factor = true;
      prepared.rope_scaling_low_freq_factor = py::cast<double>(rope_scaling_low_freq_factor_obj);
    } catch (...) {
      prepared.has_rope_scaling_low_freq_factor = false;
      prepared.rope_scaling_low_freq_factor = 0.0;
    }
  }
  auto rope_scaling_high_freq_factor_obj = PyAttrOrNone(attn, "rope_scaling_high_freq_factor");
  if (!rope_scaling_high_freq_factor_obj.is_none()) {
    try {
      prepared.has_rope_scaling_high_freq_factor = true;
      prepared.rope_scaling_high_freq_factor = py::cast<double>(rope_scaling_high_freq_factor_obj);
    } catch (...) {
      prepared.has_rope_scaling_high_freq_factor = false;
      prepared.rope_scaling_high_freq_factor = 0.0;
    }
  }
  if (!PrepareLinearLikeModule(attn.attr("w_q"), &prepared.w_q) ||
      !PrepareLinearLikeModule(attn.attr("w_k"), &prepared.w_k) ||
      !PrepareLinearLikeModule(attn.attr("w_v"), &prepared.w_v) ||
      !PrepareLinearLikeModule(attn.attr("w_o"), &prepared.w_o)) {
    return false;
  }
  prepared.direct_bitnet_qkv =
      prepared.w_q.direct_bitnet &&
      prepared.w_k.direct_bitnet &&
      prepared.w_v.direct_bitnet;
  if (prepared.direct_bitnet_qkv) {
    prepared.direct_fused_bitnet_qkv = TryFuseBitNetQkvStates(
        prepared.w_q.bitnet_state,
        prepared.w_k.bitnet_state,
        prepared.w_v.bitnet_state,
        &prepared.fused_bitnet_qkv_state,
        &prepared.fused_q_size,
        &prepared.fused_k_size,
        &prepared.fused_v_size);
    if (prepared.direct_fused_bitnet_qkv) {
      const auto& state = prepared.fused_bitnet_qkv_state;
      if (HopperDenseBitNetDecodeQkvCacheEnabled() &&
          state.qweight.defined() && state.inv_scale.defined() &&
          !state.spin_enabled && !state.pre_scale.defined() &&
          NormalizeBackendName(state.act_quant_mode) == "dynamic_int8" &&
          state.act_quant_bits == 8 &&
          state.qweight.is_cuda() &&
          t10::cuda::DeviceIsSm90OrLater(state.qweight)) {
        prepared.fused_dense_qkv_weight =
            DenseBitNetInt8WeightCached(state.qweight, state.inv_scale, torch::kBFloat16);
        if (state.bias.has_value() && state.bias.value().defined()) {
          prepared.fused_dense_qkv_bias =
              state.bias.value().to(state.qweight.device(), torch::kBFloat16).contiguous();
        }
      }
    }
  }
  prepared.use_bitnet_int8_attention_core =
      prepared.direct_bitnet_qkv &&
      BitNetStateUsesInt8PackedPath(prepared.w_q.bitnet_state) &&
      BitNetStateUsesInt8PackedPath(prepared.w_k.bitnet_state) &&
      BitNetStateUsesInt8PackedPath(prepared.w_v.bitnet_state);
  prepared.use_packed_bitnet_qkv =
      prepared.w_q.supports_bitnet_packed &&
      prepared.w_k.supports_bitnet_packed &&
      prepared.w_v.supports_bitnet_packed;
  *out = std::move(prepared);
  return true;
}

bool CanUsePreparedFusedDenseBitNetQkvDecode(
    const PreparedAttentionModule& prepared,
    const torch::Tensor& x) {
  if (!prepared.direct_fused_bitnet_qkv || !prepared.fused_dense_qkv_weight.defined()) {
    return false;
  }
  if (!x.is_cuda() || !t10::cuda::DeviceIsSm90OrLater(x) || x.scalar_type() != torch::kBFloat16) {
    return false;
  }
  const auto rows = x.size(-1) > 0 ? x.numel() / x.size(-1) : 0;
  return rows > 0 && rows <= 8;
}

c10::optional<std::vector<torch::Tensor>> TryFusedRmsNormPreparedAttentionQkv(
    const PreparedAttentionModule& prepared,
    const torch::Tensor& x,
    const py::object& norm) {
  if (!prepared.use_packed_bitnet_qkv || !prepared.direct_bitnet_qkv || !prepared.direct_fused_bitnet_qkv) {
    return c10::nullopt;
  }
  const auto& state = prepared.fused_bitnet_qkv_state;
  if (state.qweight.defined() || NormalizeBackendName(state.act_quant_mode) != "none") {
    return c10::nullopt;
  }
  if (!CanUseFusedRmsNormBitNetLinearStateForward(x, norm, state)) {
    return c10::nullopt;
  }
  auto projected = FusedRmsNormBitNetLinearStateForward(
      x,
      norm,
      state,
      c10::optional<torch::ScalarType>(x.scalar_type()));
  return SplitProjectedBitNetQkv(
      projected,
      prepared.fused_q_size,
      prepared.fused_k_size,
      prepared.fused_v_size,
      prepared.q_heads,
      prepared.kv_heads);
}

struct PreparedMlpModule {
  py::object mlp = py::none();
  PreparedLinearLikeModule w_in;
  PreparedLinearLikeModule w_out;
  std::string activation = "gelu";
  bool gated = false;
};

bool PrepareMlpModule(const py::object& mlp, PreparedMlpModule* out) {
  if (out == nullptr) {
    return false;
  }
  PreparedMlpModule prepared;
  prepared.mlp = mlp;
  prepared.activation = PyStringAttr(mlp, "activation_name", "gelu");
  prepared.gated = PyBoolAttr(mlp, "gated", false);
  if (!PrepareLinearLikeModule(mlp.attr("w_in"), &prepared.w_in) ||
      !PrepareLinearLikeModule(mlp.attr("w_out"), &prepared.w_out)) {
    return false;
  }
  *out = std::move(prepared);
  return true;
}

enum class PreparedBlockKind {
  kStandard,
  kParallel,
};

struct PreparedNativeBlock {
  PreparedBlockKind kind = PreparedBlockKind::kStandard;
  py::object block = py::none();
  py::object bc = py::none();
  py::object n = py::none();
  py::object n1 = py::none();
  py::object n2 = py::none();
  PreparedAttentionModule attn;
  PreparedMlpModule mlp;
  double residual_scale = 1.0;
  std::string norm_policy = "prenorm";
};

bool PrepareNativeBlock(const py::object& block, PreparedNativeBlock* out) {
  if (out == nullptr) {
    return false;
  }
  const auto block_type = PyTypeName(block);
  const bool standard_block =
      block_type == "TransformerBlock" || block_type == "LlamaBlock" || block_type == "GPTBlock";
  const bool parallel_block = block_type == "ParallelTransformerBlock";
  if (!standard_block && !parallel_block) {
    return false;
  }
  PreparedNativeBlock prepared;
  prepared.kind = parallel_block ? PreparedBlockKind::kParallel : PreparedBlockKind::kStandard;
  prepared.block = block;
  prepared.bc = block.attr("bc");
  prepared.residual_scale = PyFloatAttr(prepared.bc, "residual_scale", 1.0);
  prepared.norm_policy = PyStringAttr(prepared.bc, "norm_policy", "prenorm");
  if (!PrepareAttentionModule(block.attr("attn"), &prepared.attn) ||
      !PrepareMlpModule(block.attr("mlp"), &prepared.mlp)) {
    return false;
  }
  if (parallel_block) {
    prepared.n = block.attr("n");
  } else {
    prepared.n1 = block.attr("n1");
    prepared.n2 = block.attr("n2");
  }
  *out = std::move(prepared);
  return true;
}

struct PreparedNativeCausalModel {
  py::object model = py::none();
  torch::Tensor embed_weight;
  int64_t padding_idx = -1;
  py::object norm = py::none();
  PreparedLinearLikeModule lm_head;
  std::vector<PreparedNativeBlock> blocks;
};

std::shared_ptr<PreparedNativeCausalModel> TryPrepareNativeCausalModel(const py::object& model) {
  try {
    auto prepared = std::make_shared<PreparedNativeCausalModel>();
    prepared->model = model;
    auto embed = model.attr("embed");
    prepared->embed_weight = TensorAttr(embed, "weight");
    auto padding_idx_obj = PyAttrOrNone(embed, "padding_idx");
    if (!padding_idx_obj.is_none()) {
      prepared->padding_idx = py::cast<int64_t>(padding_idx_obj);
    }
    prepared->norm = model.attr("norm");
    if (!PrepareLinearLikeModule(model.attr("lm_head"), &prepared->lm_head)) {
      return nullptr;
    }
    prepared->blocks.reserve(static_cast<size_t>(py::len(model.attr("blocks"))));
    for (auto block_handle : model.attr("blocks")) {
      PreparedNativeBlock block;
      if (!PrepareNativeBlock(py::reinterpret_borrow<py::object>(block_handle), &block)) {
        return nullptr;
      }
      prepared->blocks.push_back(std::move(block));
    }
    return prepared;
  } catch (...) {
    return nullptr;
  }
}

bool SupportsNativeCausalRuntimeInputs(const py::object& attention_mask, const py::object& cache) {
  return IsSupportedAttentionMask(attention_mask) && HasNativeCacheSupport(cache);
}

bool IsSupportedNativeCausalModel(const py::object& model, const py::object& attention_mask, const py::object& cache) {
  if (PyBoolAttr(model, "training", false)) {
    return false;
  }
  if (!SupportsNativeCausalRuntimeInputs(attention_mask, cache)) {
    return false;
  }
  for (const char* name : {"embed", "blocks", "norm", "lm_head"}) {
    if (!py::hasattr(model, name)) {
      return false;
    }
  }
  if (!IsSupportedNormModule(model.attr("norm"))) {
    return false;
  }
  for (auto block_handle : model.attr("blocks")) {
    py::object block = py::reinterpret_borrow<py::object>(block_handle);
    const auto block_type = PyTypeName(block);
    const bool standard_block = block_type == "TransformerBlock" || block_type == "LlamaBlock" || block_type == "GPTBlock";
    const bool parallel_block = block_type == "ParallelTransformerBlock";
    const bool moe_block = block_type == "MoEBlock";
    if (!standard_block && !parallel_block && !moe_block) {
      return false;
    }
    if (!py::hasattr(block, "attn") || !py::hasattr(block, "bc")) {
      return false;
    }
    if (moe_block) {
      if (!py::hasattr(block, "moe")) {
        return false;
      }
    } else if (!py::hasattr(block, "mlp")) {
      return false;
    }
    if (standard_block || moe_block) {
      if (!py::hasattr(block, "n1") || !py::hasattr(block, "n2")) {
        return false;
      }
      if (!IsSupportedNormModule(block.attr("n1")) || !IsSupportedNormModule(block.attr("n2"))) {
        return false;
      }
    } else {
      if (!py::hasattr(block, "n") || !IsSupportedNormModule(block.attr("n"))) {
        return false;
      }
    }
    if (!IsSupportedAttentionModule(block.attr("attn"))) {
      return false;
    }
    if (moe_block) {
      if (!IsSupportedMoEMlpModule(block.attr("moe"))) {
        return false;
      }
    } else if (!IsSupportedMlpModule(block.attr("mlp"))) {
      return false;
    }
    auto bc = block.attr("bc");
    const auto norm_policy = PyStringAttr(bc, "norm_policy", "prenorm");
    if (standard_block || moe_block) {
      if (norm_policy != "prenorm" && norm_policy != "postnorm") {
        return false;
      }
    }
  }
  return true;
}

std::string DetectNativeExecutorKind(const py::object& model) {
  return IsSupportedNativeCausalModel(model, py::none(), py::none()) ? std::string("causal_lm") : std::string("python");
}

std::shared_ptr<PagedKvCacheState> TryCastNativePagedKvCacheState(const py::handle& obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  try {
    return py::cast<std::shared_ptr<PagedKvCacheState>>(obj);
  } catch (...) {
    return nullptr;
  }
}

std::shared_ptr<PagedKvLayerState> TryCastNativePagedKvLayerState(const py::handle& obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  try {
    return py::cast<std::shared_ptr<PagedKvLayerState>>(obj);
  } catch (...) {
    return nullptr;
  }
}

struct ResolvedNativeCache {
  py::object native_cache = py::none();
  py::object native_layers = py::none();
  std::shared_ptr<PagedKvCacheState> native_cache_state;

  bool HasState() const {
    return native_cache_state != nullptr || !native_layers.is_none();
  }

  std::shared_ptr<PagedKvLayerState> LayerState(int64_t layer_idx) const {
    if (native_cache_state != nullptr) {
      return nullptr;
    }
    if (native_layers.is_none()) {
      return nullptr;
    }
    try {
      return TryCastNativePagedKvLayerState(native_layers[py::int_(layer_idx)]);
    } catch (...) {
      return nullptr;
    }
  }
};

ResolvedNativeCache ResolveNativeCache(const py::object& cache) {
  ResolvedNativeCache resolved;
  if (cache.is_none()) {
    return resolved;
  }
  resolved.native_cache = PyAttrOrNone(cache, "_native_cache");
  if (!resolved.native_cache.is_none()) {
    resolved.native_cache_state = TryCastNativePagedKvCacheState(resolved.native_cache);
    return resolved;
  }
  resolved.native_layers = PyAttrOrNone(cache, "_native_layers");
  return resolved;
}

int64_t NativeCacheLayerLength(const ResolvedNativeCache& cache, int64_t layer_idx) {
  if (cache.native_cache_state != nullptr) {
    return cache.native_cache_state->MaxLength(layer_idx);
  }
  if (!cache.native_cache.is_none()) {
    return py::cast<int64_t>(cache.native_cache.attr("max_length")(layer_idx));
  }
  if (auto native_layer_state = cache.LayerState(layer_idx)) {
    return native_layer_state->MaxLength();
  }
  if (!cache.native_layers.is_none()) {
    py::object layer = cache.native_layers[py::int_(layer_idx)];
    return py::cast<int64_t>(layer.attr("max_length")());
  }
  return 0;
}

int64_t NativeCacheLayerLength(const py::object& cache, int64_t layer_idx) {
  return NativeCacheLayerLength(ResolveNativeCache(cache), layer_idx);
}

void NativeCacheAppendLayer(
    const ResolvedNativeCache& cache,
    int64_t layer_idx,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  if (cache.native_cache_state != nullptr) {
    cache.native_cache_state->Append(layer_idx, k, v, c10::nullopt);
    return;
  }
  if (!cache.native_cache.is_none()) {
    cache.native_cache.attr("append")(layer_idx, k, v);
    return;
  }
  if (auto native_layer_state = cache.LayerState(layer_idx)) {
    native_layer_state->Append(k, v, c10::nullopt);
    return;
  }
  if (!cache.native_layers.is_none()) {
    py::object layer = cache.native_layers[py::int_(layer_idx)];
    layer.attr("append")(k, v);
  }
}

void NativeCacheAppendLayer(const py::object& cache, int64_t layer_idx, const torch::Tensor& k, const torch::Tensor& v) {
  NativeCacheAppendLayer(ResolveNativeCache(cache), layer_idx, k, v);
}

c10::optional<torch::Tensor> NativeCacheAppendProjectedQkvRotaryLayer(
    const ResolvedNativeCache& cache,
    int64_t layer_idx,
    const torch::Tensor& projected,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t q_heads,
    int64_t kv_heads) {
  if (cache.native_cache_state != nullptr) {
    return cache.native_cache_state->AppendProjectedQkvRotary(
        layer_idx,
        projected,
        cos,
        sin,
        q_size,
        k_size,
        v_size,
        q_heads,
        kv_heads);
  }
  if (auto native_layer_state = cache.LayerState(layer_idx)) {
    return native_layer_state->AppendProjectedQkvRotary(
        projected,
        cos,
        sin,
        q_size,
        k_size,
        v_size,
        q_heads,
        kv_heads);
  }
  return c10::nullopt;
}

std::pair<torch::Tensor, torch::Tensor> NativeCacheReadLayerAll(
    const ResolvedNativeCache& cache,
    int64_t layer_idx) {
  const auto length = NativeCacheLayerLength(cache, layer_idx);
  if (cache.native_cache_state != nullptr) {
    auto out = cache.native_cache_state->ReadRange(layer_idx, 0, length, c10::nullopt);
    return {out[0], out[1]};
  }
  if (!cache.native_cache.is_none()) {
    auto out = py::cast<py::tuple>(cache.native_cache.attr("read_range")(layer_idx, 0, length));
    return {py::cast<torch::Tensor>(out[0]), py::cast<torch::Tensor>(out[1])};
  }
  if (auto native_layer_state = cache.LayerState(layer_idx)) {
    auto out = native_layer_state->ReadRange(0, length, c10::nullopt);
    return {out[0], out[1]};
  }
  if (!cache.native_layers.is_none()) {
    py::object layer = cache.native_layers[py::int_(layer_idx)];
    auto out = py::cast<py::tuple>(layer.attr("read_range")(0, length));
    return {py::cast<torch::Tensor>(out[0]), py::cast<torch::Tensor>(out[1])};
  }
  throw std::runtime_error("native cache state unavailable for supported causal executor");
}

std::pair<torch::Tensor, torch::Tensor> NativeCacheReadLayerAll(const py::object& cache, int64_t layer_idx) {
  return NativeCacheReadLayerAll(ResolveNativeCache(cache), layer_idx);
}

c10::optional<torch::Tensor> NativeCachePagedAttentionDecodeLayer(
    const ResolvedNativeCache& cache,
    int64_t layer_idx,
    const torch::Tensor& q,
    const c10::optional<torch::Tensor>& attn_mask,
    const c10::optional<double>& scale) {
  auto normalize_mask = [&](const c10::optional<torch::Tensor>& mask_opt, int64_t max_len) {
    if (!mask_opt.has_value() || !mask_opt.value().defined()) {
      return mask_opt;
    }
    auto mask = mask_opt.value();
    if (mask.dim() != 4) {
      return mask_opt;
    }
    if (mask.size(0) == 1 && q.size(0) > 1) {
      mask = mask.expand({q.size(0), mask.size(1), mask.size(2), mask.size(3)});
    }
    if (mask.size(1) == 1 && q.size(1) > 1) {
      mask = mask.expand({mask.size(0), q.size(1), mask.size(2), mask.size(3)});
    }
    if (mask.size(2) > q.size(2)) {
      mask = mask.slice(2, mask.size(2) - q.size(2), mask.size(2));
    }
    if (max_len > 0 && mask.size(3) > max_len) {
      mask = mask.slice(3, mask.size(3) - max_len, mask.size(3));
    }
    return c10::optional<torch::Tensor>(mask.contiguous());
  };
  auto cuda_paged_decode = [&](const torch::Tensor& k_pages,
                               const torch::Tensor& v_pages,
                               const torch::Tensor& block_table,
                               const torch::Tensor& lengths,
                               const c10::optional<torch::Tensor>& normalized_mask,
                               int64_t known_mask_seq) -> c10::optional<torch::Tensor> {
#if MODEL_STACK_WITH_CUDA
    auto block_table_safe = block_table.to(torch::kLong).contiguous();
    auto lengths_long = lengths.to(torch::kLong).contiguous().view(-1);
    if (auto bridged = TryPagedAttentionDecodeSdpaBridgeForward(
            q, k_pages, v_pages, block_table_safe, lengths_long, known_mask_seq, normalized_mask, scale)) {
      return bridged;
    }
    if (q.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda() && block_table.is_cuda() && lengths.is_cuda() &&
        (!normalized_mask.has_value() || !normalized_mask.value().defined() || normalized_mask.value().is_cuda()) &&
        HasCudaPagedAttentionDecodeKernel()) {
      block_table_safe.clamp_min_(0);
      return CudaPagedAttentionDecodeForward(
          q,
          k_pages,
          v_pages,
          block_table_safe,
          lengths_long,
          normalized_mask,
          scale,
          known_mask_seq);
    }
#endif
    return c10::nullopt;
  };
  if (cache.native_cache_state != nullptr) {
    const auto page_count = cache.native_cache_state->PageCount(layer_idx);
    const auto max_len = cache.native_cache_state->MaxLength(layer_idx);
    auto k_pages = cache.native_cache_state->KPages(layer_idx).slice(0, 0, page_count);
    auto v_pages = cache.native_cache_state->VPages(layer_idx).slice(0, 0, page_count);
    auto block_table = cache.native_cache_state->BlockTable(layer_idx);
    auto lengths = cache.native_cache_state->Lengths(layer_idx);
    auto normalized_mask = normalize_mask(attn_mask, max_len);
    if (auto out = cuda_paged_decode(k_pages, v_pages, block_table, lengths, normalized_mask, max_len)) {
      return out;
    }
    return PagedAttentionDecodeForward(q, k_pages, v_pages, block_table, lengths, normalized_mask, scale);
  }
  if (!cache.native_cache.is_none()) {
    const auto page_count = py::cast<int64_t>(cache.native_cache.attr("page_count")(layer_idx));
    const auto max_len = py::cast<int64_t>(cache.native_cache.attr("max_length")(layer_idx));
    auto k_pages = py::cast<torch::Tensor>(cache.native_cache.attr("k_pages")(layer_idx)).slice(0, 0, page_count);
    auto v_pages = py::cast<torch::Tensor>(cache.native_cache.attr("v_pages")(layer_idx)).slice(0, 0, page_count);
    auto block_table = py::cast<torch::Tensor>(cache.native_cache.attr("block_table")(layer_idx));
    auto lengths = py::cast<torch::Tensor>(cache.native_cache.attr("lengths")(layer_idx));
    auto normalized_mask = normalize_mask(attn_mask, max_len);
    if (auto out = cuda_paged_decode(k_pages, v_pages, block_table, lengths, normalized_mask, max_len)) {
      return out;
    }
    return PagedAttentionDecodeForward(q, k_pages, v_pages, block_table, lengths, normalized_mask, scale);
  }
  if (auto native_layer_state = cache.LayerState(layer_idx)) {
    const auto page_count = native_layer_state->PageCount();
    const auto max_len = native_layer_state->MaxLength();
    auto k_pages = native_layer_state->KPages().slice(0, 0, page_count);
    auto v_pages = native_layer_state->VPages().slice(0, 0, page_count);
    auto block_table = native_layer_state->BlockTable();
    auto lengths = native_layer_state->Lengths();
    auto normalized_mask = normalize_mask(attn_mask, max_len);
    if (auto out = cuda_paged_decode(k_pages, v_pages, block_table, lengths, normalized_mask, max_len)) {
      return out;
    }
    return PagedAttentionDecodeForward(q, k_pages, v_pages, block_table, lengths, normalized_mask, scale);
  }
  if (!cache.native_layers.is_none()) {
    py::object layer = cache.native_layers[py::int_(layer_idx)];
    const auto page_count = py::cast<int64_t>(layer.attr("page_count")());
    const auto max_len = py::cast<int64_t>(layer.attr("max_length")());
    auto k_pages = py::cast<torch::Tensor>(layer.attr("k_pages")()).slice(0, 0, page_count);
    auto v_pages = py::cast<torch::Tensor>(layer.attr("v_pages")()).slice(0, 0, page_count);
    auto block_table = py::cast<torch::Tensor>(layer.attr("block_table")());
    auto lengths = py::cast<torch::Tensor>(layer.attr("lengths")());
    auto normalized_mask = normalize_mask(attn_mask, max_len);
    if (auto out = cuda_paged_decode(k_pages, v_pages, block_table, lengths, normalized_mask, max_len)) {
      return out;
    }
    return PagedAttentionDecodeForward(q, k_pages, v_pages, block_table, lengths, normalized_mask, scale);
  }
  return c10::nullopt;
}

c10::optional<torch::Tensor> NativeCachePagedAttentionDecodeLayer(
    const py::object& cache,
    int64_t layer_idx,
    const torch::Tensor& q,
    const c10::optional<torch::Tensor>& attn_mask,
    const c10::optional<double>& scale) {
  return NativeCachePagedAttentionDecodeLayer(ResolveNativeCache(cache), layer_idx, q, attn_mask, scale);
}

torch::Tensor ToAdditiveMask(const torch::Tensor& mask) {
  if (mask.scalar_type() == torch::kBool) {
    auto masked = torch::full(
        mask.sizes(),
        -std::numeric_limits<float>::infinity(),
        torch::TensorOptions().dtype(torch::kFloat32).device(mask.device()));
    auto zeros = torch::zeros(
        mask.sizes(),
        torch::TensorOptions().dtype(torch::kFloat32).device(mask.device()));
    return torch::where(mask, masked, zeros);
  }
  return mask.to(torch::kFloat32);
}

torch::Tensor MergeAttentionBias(
    const c10::optional<torch::Tensor>& base_mask,
    const torch::Tensor& bias) {
  if (!base_mask.has_value() || !base_mask.value().defined()) {
    return bias;
  }
  return ToAdditiveMask(base_mask.value()) + bias.to(torch::kFloat32);
}

torch::Tensor BuildAlibiBiasForPositions(
    int64_t batch_size,
    int64_t num_heads,
    int64_t tgt_len,
    int64_t src_len,
    const torch::Device& device,
    const c10::optional<torch::Tensor>& position_ids) {
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  int64_t m = 1;
  while ((m << 1) <= num_heads) {
    m <<= 1;
  }
  auto slopes = torch::pow(
      torch::full({m}, 2.0f, float_opts),
      -torch::arange(0, m, float_opts) / static_cast<double>(m));
  if (m < num_heads) {
    auto extra = torch::pow(
        torch::full({num_heads - m}, 2.0f, float_opts),
        -torch::arange(1, 2 * (num_heads - m) + 1, 2, float_opts) / static_cast<double>(m));
    slopes = torch::cat({slopes, extra}, 0);
  }
  slopes = slopes.slice(0, 0, num_heads).view({1, num_heads, 1, 1});

  torch::Tensor qpos;
  if (position_ids.has_value() && position_ids.value().defined()) {
    qpos = position_ids.value().to(float_opts);
    if (qpos.dim() == 1) {
      qpos = qpos.view({1, -1});
    }
    if (qpos.size(0) == 1 && batch_size > 1) {
      qpos = qpos.expand({batch_size, qpos.size(1)});
    }
  } else {
    auto start = std::max<int64_t>(src_len - tgt_len, 0);
    qpos = torch::arange(start, start + tgt_len, float_opts).view({1, tgt_len}).expand({batch_size, tgt_len});
  }
  auto kpos = torch::arange(src_len, float_opts).view({1, 1, 1, src_len});
  return -slopes * (qpos.view({batch_size, 1, tgt_len, 1}) - kpos);
}

torch::Tensor BuildRelativePositionBiasForPositions(
    const torch::Tensor& table,
    int64_t batch_size,
    int64_t tgt_len,
    int64_t src_len,
    int64_t max_distance,
    const torch::Device& device,
    const c10::optional<torch::Tensor>& position_ids) {
  auto table_f = table.to(torch::TensorOptions().dtype(torch::kFloat32).device(device)).contiguous();
  auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(device);
  torch::Tensor qpos;
  if (position_ids.has_value() && position_ids.value().defined()) {
    qpos = position_ids.value().to(long_opts);
    if (qpos.dim() == 1) {
      qpos = qpos.view({1, -1});
    }
    if (qpos.size(0) == 1 && batch_size > 1) {
      qpos = qpos.expand({batch_size, qpos.size(1)});
    }
  } else {
    auto start = std::max<int64_t>(src_len - tgt_len, 0);
    qpos = torch::arange(start, start + tgt_len, long_opts).view({1, tgt_len}).expand({batch_size, tgt_len});
  }
  auto kpos = torch::arange(src_len, long_opts).view({1, 1, src_len});
  auto rel = (kpos - qpos.view({batch_size, tgt_len, 1}))
      .clamp(-(max_distance - 1), max_distance - 1)
      + (max_distance - 1);
  auto flat = rel.contiguous().view({-1});
  auto gathered = table_f.index_select(1, flat);
  return gathered.view({table_f.size(0), batch_size, tgt_len, src_len}).permute({1, 0, 2, 3}).contiguous();
}

c10::optional<torch::Tensor> ApplySupportedAttentionBiases(
    const py::object& block,
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& base_mask,
    const c10::optional<torch::Tensor>& position_ids) {
  c10::optional<torch::Tensor> mask = base_mask;
  auto bc = block.attr("bc");
  const auto num_heads = PyIntAttr(block.attr("attn"), "n_heads", 0);
  const auto tgt_len = x.size(1);
  const auto src_len = mask.has_value() && mask.value().defined() ? mask.value().size(-1) : tgt_len;
  if (PyBoolAttr(bc, "use_alibi", false)) {
    mask = MergeAttentionBias(
        mask,
        BuildAlibiBiasForPositions(x.size(0), num_heads, tgt_len, src_len, x.device(), position_ids));
  }
  auto rpb_table = PyAttrOrNone(block, "rpb_table");
  if (!rpb_table.is_none()) {
    const auto max_distance = PyIntAttr(bc, "rpb_max_distance", 0);
    TORCH_CHECK(max_distance > 0, "supported causal executor requires positive rpb_max_distance when rpb_table is present");
    mask = MergeAttentionBias(
        mask,
        BuildRelativePositionBiasForPositions(
            py::cast<torch::Tensor>(rpb_table),
            x.size(0),
            tgt_len,
            src_len,
            max_distance,
            x.device(),
            position_ids));
  }
  return mask;
}

std::vector<torch::Tensor> ResolveAttentionRotaryEmbedding(
    const py::object& attn,
    const torch::Tensor& x,
    int64_t head_dim,
    double base_theta,
    double attention_scaling,
    torch::ScalarType target_dtype,
    const c10::optional<torch::Tensor>& position_ids,
    const c10::optional<int64_t>& known_single_position) {
  const auto seq_len = x.size(1);
  int64_t needed = ResolveRopeSequenceLength(seq_len, position_ids, known_single_position);
  torch::Tensor gather_pos;
  bool single_position = false;
  int64_t single_position_idx = 0;
  if (known_single_position.has_value()) {
    single_position = true;
    single_position_idx = known_single_position.value();
    needed = single_position_idx + 1;
  } else if (position_ids.has_value() && position_ids.value().defined()) {
    gather_pos = position_ids.value().to(torch::TensorOptions().dtype(torch::kLong).device(x.device()));
    if (gather_pos.dim() == 2) {
      gather_pos = gather_pos[0];
    }
    if (gather_pos.numel() == 1) {
      single_position = true;
      single_position_idx = gather_pos.reshape({-1})[0].item<int64_t>();
      needed = single_position_idx + 1;
    } else if (gather_pos.numel() > 0) {
      needed = gather_pos.max().item<int64_t>() + 1;
    }
  }

  auto rope_cos_obj = PyAttrOrNone(attn, "_rope_cos");
  auto rope_sin_obj = PyAttrOrNone(attn, "_rope_sin");
  torch::Tensor cache_cos;
  torch::Tensor cache_sin;
  if (!rope_cos_obj.is_none() && !rope_sin_obj.is_none()) {
    try {
      cache_cos = py::cast<torch::Tensor>(rope_cos_obj);
      cache_sin = py::cast<torch::Tensor>(rope_sin_obj);
    } catch (...) {
      cache_cos = torch::Tensor();
      cache_sin = torch::Tensor();
    }
  }

  const bool cache_compatible = cache_cos.defined() && cache_sin.defined() &&
      cache_cos.dim() == 2 && cache_sin.dim() == 2 &&
      cache_cos.size(1) == head_dim && cache_sin.size(1) == head_dim &&
      cache_cos.device() == x.device() && cache_sin.device() == x.device() &&
      cache_cos.scalar_type() == target_dtype && cache_sin.scalar_type() == target_dtype;
  const bool cache_usable = cache_compatible &&
      cache_cos.size(0) >= needed && cache_sin.size(0) >= needed;

  if (!cache_usable) {
    auto reference = torch::empty({1, 1, 1}, x.options().dtype(target_dtype));
    if (cache_compatible && cache_cos.size(0) < needed) {
      auto missing = ResolveRotaryEmbeddingRangeForward(
          reference,
          head_dim,
          base_theta,
          attention_scaling,
          cache_cos.size(0),
          needed);
      cache_cos = torch::cat({cache_cos, missing[0]}, 0);
      cache_sin = torch::cat({cache_sin, missing[1]}, 0);
    } else {
      auto full_reference = torch::empty({1, needed, 1}, x.options().dtype(target_dtype));
      auto full = ResolveRotaryEmbeddingForward(
          full_reference,
          head_dim,
          base_theta,
          attention_scaling,
          c10::nullopt);
      cache_cos = full[0];
      cache_sin = full[1];
    }
    try {
      py::setattr(attn, "_rope_cos", py::cast(cache_cos));
      py::setattr(attn, "_rope_sin", py::cast(cache_sin));
    } catch (...) {
    }
  }

  if (single_position) {
    return {
        cache_cos.slice(0, single_position_idx, single_position_idx + 1),
        cache_sin.slice(0, single_position_idx, single_position_idx + 1),
    };
  }
  if (gather_pos.defined()) {
    return {
        at::index_select(cache_cos, 0, gather_pos.reshape({-1})).view({seq_len, head_dim}),
        at::index_select(cache_sin, 0, gather_pos.reshape({-1})).view({seq_len, head_dim}),
    };
  }
  return {
      cache_cos.slice(0, 0, seq_len),
      cache_sin.slice(0, 0, seq_len),
  };
}

torch::Tensor ProjectPreparedAttentionOutput(
    const PreparedAttentionModule& prepared,
    const torch::Tensor& heads_out) {
  if (prepared.w_o.direct_bitnet &&
      prepared.w_o.dense_decode_weight.defined()) {
    return PreparedLinearLikeForward(MergeHeadsForward(heads_out), prepared.w_o, "auto");
  }
  if (prepared.w_o.supports_bitnet_packed) {
    return AttentionPackedBitNetOutputForward(heads_out, prepared.w_o.module);
  }
  return PreparedLinearLikeForward(MergeHeadsForward(heads_out), prepared.w_o, "auto");
}

c10::optional<torch::Tensor> TryProjectedQkvRotaryCacheAttention(
    const PreparedAttentionModule& prepared,
    const torch::Tensor& x,
    const torch::Tensor& projected,
    const c10::optional<torch::Tensor>& mask,
    const py::object& cache,
    const ResolvedNativeCache* resolved_cache,
    int64_t layer_idx,
    const c10::optional<torch::Tensor>& position_ids,
    const c10::optional<int64_t>& known_single_position) {
  if (!BitNetProjectedQkvDecodeFusedAppendEnabled() ||
      !prepared.use_rope || cache.is_none() || x.size(0) < 4 || x.size(1) != 1 || !projected.is_cuda()) {
    return c10::nullopt;
  }
  auto native_cache = resolved_cache != nullptr ? *resolved_cache : ResolveNativeCache(cache);
  if (NativeCacheLayerLength(native_cache, layer_idx) > HopperDenseBitNetDecodeFusedAppendMaxCacheLength()) {
    return c10::nullopt;
  }

  const auto rope_seq_len = ResolveRopeSequenceLength(x.size(1), position_ids, known_single_position);
  auto rope_parameters = ResolvePreparedAttentionRopeParameters(prepared, rope_seq_len);
  auto cos_sin = ResolveAttentionRotaryEmbedding(
      prepared.attn,
      x,
      prepared.head_dim,
      rope_parameters.first,
      rope_parameters.second,
      projected.scalar_type(),
      position_ids,
      known_single_position);
  if (!cos_sin[0].defined() || !cos_sin[1].defined() ||
      cos_sin[0].dim() != 2 || cos_sin[1].dim() != 2 ||
      cos_sin[0].size(0) != 1 || cos_sin[1].size(0) != 1) {
    return c10::nullopt;
  }

  auto fused_q = NativeCacheAppendProjectedQkvRotaryLayer(
      native_cache,
      layer_idx,
      projected,
      cos_sin[0],
      cos_sin[1],
      prepared.fused_q_size,
      prepared.fused_k_size,
      prepared.fused_v_size,
      prepared.q_heads,
      prepared.kv_heads);
  if (!fused_q.has_value()) {
    return c10::nullopt;
  }

  const auto attention_scale = c10::optional<double>(prepared.attention_scale);
  auto paged_out = NativeCachePagedAttentionDecodeLayer(
      native_cache,
      layer_idx,
      fused_q.value(),
      mask,
      attention_scale);
  if (paged_out.has_value()) {
    return ProjectPreparedAttentionOutput(prepared, paged_out.value());
  }
  auto kv = NativeCacheReadLayerAll(native_cache, layer_idx);
  auto out = NativeAttentionForward(
      fused_q.value(),
      kv.first,
      kv.second,
      mask,
      !mask.has_value(),
      attention_scale);
  return ProjectPreparedAttentionOutput(prepared, out);
}

torch::Tensor ExecutePreparedAttentionProjected(
    const PreparedAttentionModule& prepared,
    const torch::Tensor& x,
    std::vector<torch::Tensor> qkv,
    const c10::optional<torch::Tensor>& mask,
    const py::object& cache,
    const ResolvedNativeCache* resolved_cache,
    int64_t layer_idx,
    const c10::optional<torch::Tensor>& position_ids,
    const c10::optional<int64_t>& known_single_position) {
  auto qh = qkv[0];
  auto kh_new = qkv[1];
  auto vh_new = qkv[2];
  const auto attention_scale = c10::optional<double>(prepared.attention_scale);

  if (prepared.use_rope) {
    const auto rope_seq_len = ResolveRopeSequenceLength(x.size(1), position_ids, known_single_position);
    auto rope_parameters = ResolvePreparedAttentionRopeParameters(prepared, rope_seq_len);
    auto cos_sin = ResolveAttentionRotaryEmbedding(
        prepared.attn,
        x,
        prepared.head_dim,
        rope_parameters.first,
        rope_parameters.second,
        qh.scalar_type(),
        position_ids,
        known_single_position);
    auto rotated = ApplyRotaryForward(qh, kh_new, cos_sin[0], cos_sin[1]);
    qh = rotated[0];
    kh_new = rotated[1];
  }

  torch::Tensor kh_all = kh_new;
  torch::Tensor vh_all = vh_new;
  if (!cache.is_none()) {
    const auto& native_cache = resolved_cache != nullptr ? *resolved_cache : ResolveNativeCache(cache);
    NativeCacheAppendLayer(native_cache, layer_idx, kh_new, vh_new);
    if (qh.size(2) == 1) {
      auto paged_out = NativeCachePagedAttentionDecodeLayer(
          native_cache,
          layer_idx,
          qh,
          mask,
          attention_scale);
      if (paged_out.has_value()) {
        return ProjectPreparedAttentionOutput(prepared, paged_out.value());
      }
    }
    auto kv = NativeCacheReadLayerAll(native_cache, layer_idx);
    kh_all = kv.first;
    vh_all = kv.second;
  }

  const bool use_bitnet_int8_attention_core =
      prepared.use_bitnet_int8_attention_core &&
      !(qh.is_cuda() &&
        t10::cuda::DeviceIsSm90OrLater(qh) &&
        ((qh.size(0) * qh.size(2)) >= 8));
  auto out = use_bitnet_int8_attention_core
      ? Int8AttentionFromFloatForward(
            qh,
            kh_all,
            vh_all,
            mask,
            !mask.has_value(),
            attention_scale,
            c10::optional<torch::ScalarType>(qh.scalar_type()),
            c10::nullopt,
            c10::nullopt,
            c10::nullopt)
      : NativeAttentionForward(
            qh,
            kh_all,
            vh_all,
            mask,
            !mask.has_value(),
            attention_scale);
  return ProjectPreparedAttentionOutput(prepared, out);
}

torch::Tensor ExecutePreparedAttention(
    const PreparedAttentionModule& prepared,
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& mask,
    const py::object& cache,
    const ResolvedNativeCache* resolved_cache,
    int64_t layer_idx,
    const c10::optional<torch::Tensor>& position_ids,
    const c10::optional<int64_t>& known_single_position) {
  std::vector<torch::Tensor> qkv;
  if (CanUsePreparedFusedDenseBitNetQkvDecode(prepared, x)) {
    auto projected = LinearForward(
        x,
        prepared.fused_dense_qkv_weight,
        prepared.fused_dense_qkv_bias,
        "auto");
    if (auto out = TryProjectedQkvRotaryCacheAttention(
            prepared,
            x,
            projected,
            mask,
            cache,
            resolved_cache,
            layer_idx,
            position_ids,
            known_single_position)) {
      return out.value();
    }
    qkv = SplitProjectedBitNetQkv(
        projected,
        prepared.fused_q_size,
        prepared.fused_k_size,
        prepared.fused_v_size,
        prepared.q_heads,
        prepared.kv_heads);
  } else if (prepared.direct_bitnet_qkv &&
      PreferDirectBitNetRuntimeLinearQkv(
          x,
          prepared.w_q.module,
          prepared.w_q.bitnet_state,
          prepared.w_k.module,
          prepared.w_k.bitnet_state,
          prepared.w_v.module,
          prepared.w_v.bitnet_state)) {
    auto q = PreparedLinearLikeForward(x, prepared.w_q, "auto");
    auto k = PreparedLinearLikeForward(x, prepared.w_k, "auto");
    auto v = PreparedLinearLikeForward(x, prepared.w_v, "auto");
    qkv = {
        SplitHeadsForward(q, prepared.q_heads),
        SplitHeadsForward(k, prepared.kv_heads),
        SplitHeadsForward(v, prepared.kv_heads),
    };
  } else if (prepared.use_packed_bitnet_qkv) {
    if (prepared.direct_bitnet_qkv) {
      if (prepared.direct_fused_bitnet_qkv) {
        const bool projected_decode_fast_path =
            prepared.use_rope && !cache.is_none() && x.size(0) >= 4 && x.size(1) == 1 &&
            x.is_cuda() && BitNetProjectedQkvDecodeFusedAppendEnabled();
        if (projected_decode_fast_path) {
          auto projected = BitNetLinearStateForward(
              x,
              prepared.fused_bitnet_qkv_state,
              c10::optional<torch::ScalarType>(x.scalar_type()));
          if (auto out = TryProjectedQkvRotaryCacheAttention(
                  prepared,
                  x,
                  projected,
                  mask,
                  cache,
                  resolved_cache,
                  layer_idx,
                  position_ids,
                  known_single_position)) {
            return out.value();
          }
          qkv = SplitProjectedBitNetQkv(
              projected,
              prepared.fused_q_size,
              prepared.fused_k_size,
              prepared.fused_v_size,
              prepared.q_heads,
              prepared.kv_heads);
        } else if (prepared.fused_bitnet_qkv_state.qweight.defined() && prepared.fused_bitnet_qkv_state.inv_scale.defined()) {
          qkv = BitNetInt8FusedQkvPackedHeadsProjectionForward(
              x,
              prepared.fused_bitnet_qkv_state.qweight,
              prepared.fused_bitnet_qkv_state.inv_scale,
              prepared.fused_bitnet_qkv_state.bias,
              prepared.fused_bitnet_qkv_state.pre_scale.defined()
                  ? c10::optional<torch::Tensor>(prepared.fused_bitnet_qkv_state.pre_scale)
                  : c10::nullopt,
              prepared.fused_bitnet_qkv_state.act_quant_mode,
              prepared.fused_bitnet_qkv_state.act_quant_method,
              prepared.fused_bitnet_qkv_state.act_quant_bits,
              prepared.fused_bitnet_qkv_state.act_quant_percentile,
              prepared.fused_bitnet_qkv_state.act_scale.defined()
                  ? c10::optional<torch::Tensor>(prepared.fused_bitnet_qkv_state.act_scale)
                  : c10::nullopt,
              prepared.fused_q_size,
              prepared.fused_k_size,
              prepared.fused_v_size,
              prepared.q_heads,
              prepared.kv_heads,
              c10::optional<torch::ScalarType>(x.scalar_type()));
        } else {
          if (prepared.fused_bitnet_qkv_state.compute_packed_words.defined() &&
              prepared.fused_bitnet_qkv_state.compute_row_scales.defined()) {
            auto projected = BitNetLinearStateForward(x, prepared.fused_bitnet_qkv_state);
            qkv = SplitProjectedBitNetQkv(
                projected,
                prepared.fused_q_size,
                prepared.fused_k_size,
                prepared.fused_v_size,
                prepared.q_heads,
                prepared.kv_heads);
          } else {
            qkv = BitNetFusedQkvPackedHeadsProjectionForward(
                x,
                prepared.fused_bitnet_qkv_state.packed_weight,
                prepared.fused_bitnet_qkv_state.scale_values,
                prepared.fused_bitnet_qkv_state.layout_header,
                prepared.fused_bitnet_qkv_state.segment_offsets,
                prepared.fused_bitnet_qkv_state.bias,
                prepared.fused_q_size,
                prepared.fused_k_size,
                prepared.fused_v_size,
                prepared.q_heads,
                prepared.kv_heads);
          }
        }
      } else {
        qkv = AttentionPackedBitNetQkvForwardFromStates(
            x,
            prepared.w_q.bitnet_state,
            prepared.w_k.bitnet_state,
            prepared.w_v.bitnet_state,
            prepared.q_heads,
            prepared.kv_heads);
      }
    } else {
      qkv = AttentionPackedBitNetQkvForward(x, prepared.attn, prepared.q_heads, prepared.kv_heads);
    }
  } else {
    auto q = PreparedLinearLikeForward(x, prepared.w_q, "auto");
    auto k = PreparedLinearLikeForward(x, prepared.w_k, "auto");
    auto v = PreparedLinearLikeForward(x, prepared.w_v, "auto");
    qkv = {
        SplitHeadsForward(q, prepared.q_heads),
        SplitHeadsForward(k, prepared.kv_heads),
        SplitHeadsForward(v, prepared.kv_heads),
    };
  }

  return ExecutePreparedAttentionProjected(
      prepared,
      x,
      std::move(qkv),
      mask,
      cache,
      resolved_cache,
      layer_idx,
      position_ids,
      known_single_position);
}

torch::Tensor ExecutePreparedMlpHidden(
    const PreparedMlpModule& prepared,
    torch::Tensor hidden);

torch::Tensor ExecutePreparedMlp(
    const PreparedMlpModule& prepared,
    const torch::Tensor& x) {
  return ExecutePreparedMlpHidden(
      prepared,
      PreparedLinearLikeForward(x, prepared.w_in, "auto"));
}

torch::Tensor ExecutePreparedMlpHidden(
    const PreparedMlpModule& prepared,
    torch::Tensor hidden) {
  if (prepared.gated) {
    if (prepared.w_out.direct_bitnet) {
      torch::Tensor gated_hidden;
      const auto mode = NormalizeBackendName(prepared.w_out.bitnet_state.act_quant_mode);
      if (mode == "dynamic_int8" && prepared.w_out.bitnet_state.act_quant_bits == 8) {
        gated_hidden = ApplyGatedActivation(hidden, prepared.activation);
        if (PreferDirectBitNetRuntimeLinear(gated_hidden, prepared.w_out.module, prepared.w_out.bitnet_state)) {
          return PreparedLinearLikeForward(gated_hidden, prepared.w_out, "auto");
        }
      }
      if (auto fused = TryBitNetGatedInt8LinearStateForward(hidden, prepared.activation, prepared.w_out.bitnet_state)) {
        return fused.value();
      }
      if (gated_hidden.defined()) {
        hidden = std::move(gated_hidden);
      } else {
        hidden = ApplyGatedActivation(hidden, prepared.activation);
      }
    } else {
      hidden = ApplyGatedActivation(hidden, prepared.activation);
    }
  } else {
    hidden = ApplyActivation(hidden, prepared.activation);
  }
  return PreparedLinearLikeForward(hidden, prepared.w_out, "auto");
}

c10::optional<torch::Tensor> TryPreparedNativeCausalModelForward(
    const std::shared_ptr<PreparedNativeCausalModel>& prepared,
    const torch::Tensor& input_ids,
    const py::object& attention_mask,
    const py::object& cache,
    const py::object& position_ids_obj,
    const py::object& cache_position_obj,
    bool model_uses_attention_biases,
    int64_t known_decode_position) {
  if (prepared == nullptr) {
    return c10::nullopt;
  }
  if (!SupportsNativeCausalRuntimeInputs(attention_mask, cache)) {
    return c10::nullopt;
  }

  auto provided_attention_mask = OptionalTensorFromPyObject(attention_mask);
  c10::optional<torch::Tensor> token_attention_mask = c10::nullopt;
  c10::optional<torch::Tensor> explicit_attention_mask = c10::nullopt;
  if (provided_attention_mask.has_value() && provided_attention_mask.value().defined()) {
    if (IsTokenAttentionMaskTensor(provided_attention_mask.value(), input_ids.size(0), input_ids.size(1))) {
      token_attention_mask = provided_attention_mask;
    } else {
      explicit_attention_mask = provided_attention_mask;
    }
  }
  auto cache_position = OptionalTensorFromPyObject(cache_position_obj);
  auto position_ids = OptionalTensorFromPyObject(position_ids_obj);
  auto native_cache = ResolveNativeCache(cache);
  c10::optional<int64_t> known_single_position = c10::nullopt;
  if (!position_ids.has_value() &&
      known_decode_position >= 0 &&
      !cache.is_none() &&
      input_ids.size(1) == 1 &&
      !token_attention_mask.has_value() &&
      !explicit_attention_mask.has_value() &&
      !model_uses_attention_biases) {
    known_single_position = known_decode_position;
  }

  auto x = known_single_position.has_value()
      ? EmbeddingForwardUnchecked(prepared->embed_weight, input_ids, prepared->padding_idx)
      : EmbeddingForward(prepared->embed_weight, input_ids, prepared->padding_idx);

  if (!position_ids.has_value() && !known_single_position.has_value()) {
    int64_t past_length = 0;
    if (!cache.is_none() && !cache_position.has_value()) {
      past_length = NativeCacheLayerLength(native_cache, 0);
    }
    position_ids = ResolvePositionIdsForward(
        x.size(0),
        x.size(1),
        x,
        token_attention_mask,
        cache_position,
        past_length);
  }

  c10::optional<torch::Tensor> mask = c10::nullopt;
  if (explicit_attention_mask.has_value()) {
    TORCH_CHECK(!prepared->blocks.empty(), "supported causal executor requires at least one block");
    mask = PrepareExplicitAttentionMaskForHeads(
        explicit_attention_mask.value(),
        x.size(0),
        prepared->blocks.front().attn.q_heads,
        x.size(1),
        explicit_attention_mask.value().size(-1));
  } else if (
      cache.is_none() || x.size(1) != 1 || token_attention_mask.has_value() ||
      model_uses_attention_biases) {
    mask = CreateCausalMaskForward(x, token_attention_mask, cache_position, position_ids);
  }

  int64_t layer_idx = 0;
  for (const auto& block : prepared->blocks) {
    const auto block_mask = model_uses_attention_biases
        ? ApplySupportedAttentionBiases(block.block, x, mask, position_ids)
        : mask;
    if (block.kind == PreparedBlockKind::kParallel) {
      auto y = ApplyNormModuleForward(x, block.n);
      auto attn_out = ExecutePreparedAttention(block.attn, y, mask, cache, &native_cache, layer_idx, position_ids, known_single_position);
      auto mlp_out = ExecutePreparedMlp(block.mlp, y);
      x = ResidualAddForward(x, attn_out, block.residual_scale);
      x = ResidualAddForward(x, mlp_out, block.residual_scale);
      ++layer_idx;
      continue;
    }
    if (block.norm_policy == "prenorm") {
      torch::Tensor attn_out;
      if (auto fused_qkv = TryFusedRmsNormPreparedAttentionQkv(block.attn, x, block.n1)) {
        attn_out = ExecutePreparedAttentionProjected(
            block.attn,
            x,
            std::move(fused_qkv.value()),
            block_mask,
            cache,
            &native_cache,
            layer_idx,
            position_ids,
            known_single_position);
      } else {
        auto attn_in = ApplyNormModuleForward(x, block.n1);
        attn_out = ExecutePreparedAttention(
            block.attn,
            attn_in,
            block_mask,
            cache,
            &native_cache,
            layer_idx,
            position_ids,
            known_single_position);
      }
      torch::Tensor mlp_out;
      if (block.mlp.w_in.direct_bitnet &&
          CanUseFusedAddRmsNormBitNetLinearStateForward(x, attn_out, block.n2, block.mlp.w_in.bitnet_state)) {
        auto fused_mlp_in = FusedAddRmsNormBitNetLinearStateForward(
            x,
            attn_out,
            block.n2,
            block.residual_scale,
            block.mlp.w_in.bitnet_state,
            c10::optional<torch::ScalarType>(x.scalar_type()));
        x = fused_mlp_in[0];
        mlp_out = ExecutePreparedMlpHidden(block.mlp, fused_mlp_in[1]);
      } else {
        auto add_norm = AddNormModuleForward(x, attn_out, block.n2, block.residual_scale);
        x = add_norm[0];
        mlp_out = ExecutePreparedMlp(block.mlp, add_norm[1]);
      }
      x = ResidualAddForward(x, mlp_out, block.residual_scale);
    } else {
      auto attn_out = ExecutePreparedAttention(
          block.attn,
          x,
          block_mask,
          cache,
          &native_cache,
          layer_idx,
          position_ids,
          known_single_position);
      x = AddNormModuleForward(x, attn_out, block.n1, block.residual_scale)[1];
      auto mlp_out = ExecutePreparedMlp(block.mlp, x);
      x = AddNormModuleForward(x, mlp_out, block.n2, block.residual_scale)[1];
    }
    ++layer_idx;
  }

  if (prepared->lm_head.direct_bitnet &&
      CanUseFusedRmsNormBitNetLinearStateForward(x, prepared->norm, prepared->lm_head.bitnet_state)) {
    return FusedRmsNormBitNetLinearStateForward(x, prepared->norm, prepared->lm_head.bitnet_state);
  }
  x = ApplyNormModuleForward(x, prepared->norm);
  auto logits = PreparedLinearLikeForward(x, prepared->lm_head, "auto");
  if (known_single_position.has_value() &&
      prepared->lm_head.direct_bitnet &&
      logits.dim() == 3 &&
      logits.size(1) == 1) {
    return logits.view({logits.size(0), logits.size(2)});
  }
  return logits;
}

torch::Tensor ExecuteSupportedAttention(
    const py::object& attn,
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& mask,
    const py::object& cache,
    int64_t layer_idx,
    const c10::optional<torch::Tensor>& position_ids) {
  auto w_q = attn.attr("w_q");
  auto w_k = attn.attr("w_k");
  auto w_v = attn.attr("w_v");
  auto w_o = attn.attr("w_o");
  const auto q_heads = PyIntAttr(attn, "n_heads", 0);
  const auto kv_heads = PyIntAttr(attn, "n_kv_heads", q_heads);
  const auto head_dim = PyIntAttr(attn, "head_dim", x.size(-1) / std::max<int64_t>(q_heads, 1));
  BitNetModuleState q_state;
  BitNetModuleState k_state;
  BitNetModuleState v_state;
  const bool direct_bitnet_qkv =
      BitNetModuleDirectSupported(w_q, &q_state) &&
      BitNetModuleDirectSupported(w_k, &k_state) &&
      BitNetModuleDirectSupported(w_v, &v_state);
  const bool use_bitnet_int8_attention_core =
      direct_bitnet_qkv &&
      BitNetStateUsesInt8PackedPath(q_state) &&
      BitNetStateUsesInt8PackedPath(k_state) &&
      BitNetStateUsesInt8PackedPath(v_state);
  const bool use_packed_bitnet_qkv =
      ModuleSupportsPackedBackend(w_q, "bitnet") &&
      ModuleSupportsPackedBackend(w_k, "bitnet") &&
      ModuleSupportsPackedBackend(w_v, "bitnet");
  BitNetModuleState w_o_state;
  const bool direct_bitnet_output = BitNetModuleDirectSupported(w_o, &w_o_state);
  std::vector<torch::Tensor> qkv;
  if (direct_bitnet_qkv &&
      PreferDirectBitNetRuntimeLinearQkv(x, w_q, q_state, w_k, k_state, w_v, v_state)) {
    auto q = LinearLikeModuleForward(x, w_q, "auto");
    auto k = LinearLikeModuleForward(x, w_k, "auto");
    auto v = LinearLikeModuleForward(x, w_v, "auto");
    qkv = {
        SplitHeadsForward(q, q_heads),
        SplitHeadsForward(k, kv_heads),
        SplitHeadsForward(v, kv_heads),
    };
  } else if (use_packed_bitnet_qkv) {
    qkv = direct_bitnet_qkv
        ? AttentionPackedBitNetQkvForwardFromStates(x, q_state, k_state, v_state, q_heads, kv_heads)
        : AttentionPackedBitNetQkvForward(x, attn, q_heads, kv_heads);
  } else if (ModuleHasRuntimeLinear(w_q) || ModuleHasRuntimeLinear(w_k) || ModuleHasRuntimeLinear(w_v)) {
    auto q = LinearLikeModuleForward(x, w_q, "auto");
    auto k = LinearLikeModuleForward(x, w_k, "auto");
    auto v = LinearLikeModuleForward(x, w_v, "auto");
    qkv = {
        SplitHeadsForward(q, q_heads),
        SplitHeadsForward(k, kv_heads),
        SplitHeadsForward(v, kv_heads),
    };
  } else {
    qkv = QkvHeadsProjectionForward(
        x,
        TensorAttr(w_q, "weight"),
        TensorAttrOptional(w_q, "bias"),
        TensorAttr(w_k, "weight"),
        TensorAttrOptional(w_k, "bias"),
        TensorAttr(w_v, "weight"),
        TensorAttrOptional(w_v, "bias"),
        q_heads,
        kv_heads,
        "auto");
  }
  auto qh = qkv[0];
  auto kh_new = qkv[1];
  auto vh_new = qkv[2];
  const auto attention_scale =
      c10::optional<double>(PyFloatAttr(attn, "scaling", std::pow(static_cast<double>(head_dim), -0.5)));

  if (PyBoolAttr(attn, "use_rope", false)) {
    const auto rope_seq_len = ResolveRopeSequenceLength(x.size(1), position_ids);
    const auto scaling_factor = PyFloatAttrOptional(attn, "rope_scaling_factor");
    auto original_max_position_embeddings =
        PyIntAttrOptional(attn, "rope_scaling_original_max_position_embeddings");
    if (!original_max_position_embeddings.has_value()) {
      original_max_position_embeddings = PyIntAttrOptional(attn, "max_position_embeddings");
    }
    auto rope_parameters = ResolveNativeRopeParameters(
        rope_seq_len,
        head_dim,
        PyFloatAttr(attn, "rope_theta", 1e6),
        PyFloatAttr(attn, "rope_attention_scaling", 1.0),
        PyStringAttr(attn, "rope_scaling_type", ""),
        scaling_factor.has_value(),
        scaling_factor.value_or(1.0),
        original_max_position_embeddings,
        PyFloatAttrOptional(attn, "rope_scaling_low_freq_factor"),
        PyFloatAttrOptional(attn, "rope_scaling_high_freq_factor"));
    auto cos_sin = ResolveAttentionRotaryEmbedding(
        attn,
        x,
        head_dim,
        rope_parameters.first,
        rope_parameters.second,
        qh.scalar_type(),
        position_ids,
        c10::nullopt);
    auto rotated = ApplyRotaryForward(qh, kh_new, cos_sin[0], cos_sin[1]);
    qh = rotated[0];
    kh_new = rotated[1];
  }

  torch::Tensor kh_all = kh_new;
  torch::Tensor vh_all = vh_new;
  if (!cache.is_none()) {
    NativeCacheAppendLayer(cache, layer_idx, kh_new, vh_new);
    if (qh.size(2) == 1) {
      auto paged_out = NativeCachePagedAttentionDecodeLayer(
          cache,
          layer_idx,
          qh,
          mask,
          attention_scale);
      if (paged_out.has_value()) {
        if (direct_bitnet_output) {
          auto merged = MergeHeadsForward(paged_out.value());
          if (PreferDirectBitNetRuntimeLinear(merged, w_o, w_o_state)) {
            return PythonLinearModuleForward(merged, w_o, "auto");
          }
          return BitNetLinearStateForward(merged, w_o_state);
        }
        if (ModuleSupportsPackedBackend(w_o, "bitnet")) {
          return AttentionPackedBitNetOutputForward(paged_out.value(), w_o);
        }
        if (ModuleHasRuntimeLinear(w_o)) {
          return PythonLinearModuleForward(MergeHeadsForward(paged_out.value()), w_o, "auto");
        }
        return HeadOutputProjectionForward(
            paged_out.value(),
            TensorAttr(w_o, "weight"),
            TensorAttrOptional(w_o, "bias"),
            "auto");
      }
    }
    auto kv = NativeCacheReadLayerAll(cache, layer_idx);
    kh_all = kv.first;
    vh_all = kv.second;
  }

  const bool use_int8_attention_core =
      use_bitnet_int8_attention_core &&
      !(qh.is_cuda() &&
        t10::cuda::DeviceIsSm90OrLater(qh) &&
        ((qh.size(0) * qh.size(2)) >= 8));
  auto out = use_int8_attention_core
      ? Int8AttentionFromFloatForward(
            qh,
            kh_all,
            vh_all,
            mask,
            !mask.has_value(),
            attention_scale,
            c10::optional<torch::ScalarType>(qh.scalar_type()),
            c10::nullopt,
            c10::nullopt,
            c10::nullopt)
      : NativeAttentionForward(
            qh,
            kh_all,
            vh_all,
            mask,
            !mask.has_value(),
            attention_scale);
  if (direct_bitnet_output) {
    auto merged = MergeHeadsForward(out);
    if (PreferDirectBitNetRuntimeLinear(merged, w_o, w_o_state)) {
      return PythonLinearModuleForward(merged, w_o, "auto");
    }
    return BitNetLinearStateForward(merged, w_o_state);
  }
  if (ModuleSupportsPackedBackend(w_o, "bitnet")) {
    return AttentionPackedBitNetOutputForward(out, w_o);
  }
  if (ModuleHasRuntimeLinear(w_o)) {
    return PythonLinearModuleForward(MergeHeadsForward(out), w_o, "auto");
  }
  return HeadOutputProjectionForward(
      out,
      TensorAttr(w_o, "weight"),
      TensorAttrOptional(w_o, "bias"),
      "auto");
}

torch::Tensor ExecuteSupportedMlp(const py::object& mlp, const torch::Tensor& x) {
  const auto activation = PyStringAttr(mlp, "activation_name", "gelu");
  const bool gated = PyBoolAttr(mlp, "gated", false);
  auto w_in = mlp.attr("w_in");
  auto w_out = mlp.attr("w_out");
  const bool module_aware = ModuleHasRuntimeLinear(w_in) || ModuleHasRuntimeLinear(w_out);
  if (!module_aware) {
    return MlpForward(
        x,
        TensorAttr(w_in, "weight"),
        TensorAttrOptional(w_in, "bias"),
        TensorAttr(w_out, "weight"),
        TensorAttrOptional(w_out, "bias"),
        activation,
        gated,
        "auto");
  }

  auto hidden = LinearLikeModuleForward(x, w_in, "auto");
  if (gated) {
    hidden = ApplyGatedActivation(hidden, activation);
  } else {
    hidden = ApplyActivation(hidden, activation);
  }
  return LinearLikeModuleForward(hidden, w_out, "auto");
}

torch::Tensor ExecuteSupportedMoE(const py::object& moe, const torch::Tensor& x) {
  const auto num_experts = PyIntAttr(moe, "num_experts", 0);
  const auto k = PyIntAttr(moe, "k", 1);
  TORCH_CHECK(num_experts > 0, "supported native MoE requires positive num_experts");
  TORCH_CHECK(k > 0 && k <= num_experts, "supported native MoE requires 0 < k <= num_experts");

  auto logits = LinearLikeModuleForward(x, moe.attr("router"), "auto");
  auto probs = torch::softmax(logits.to(torch::kFloat32), -1);
  auto topk = torch::topk(probs, k, -1, true, true);
  auto assignments = std::get<1>(topk).to(torch::kLong);
  auto combine_weights = std::get<0>(topk);

  std::vector<torch::Tensor> expert_outputs;
  expert_outputs.reserve(static_cast<size_t>(num_experts));
  for (auto expert_handle : moe.attr("experts")) {
    py::object expert = py::reinterpret_borrow<py::object>(expert_handle);
    expert_outputs.push_back(ExecuteSupportedMlp(expert, x));
  }
  auto stacked = torch::stack(expert_outputs, 2);
  auto gather_index = assignments.unsqueeze(-1).expand({stacked.size(0), stacked.size(1), assignments.size(-1), stacked.size(-1)});
  auto gathered = stacked.gather(2, gather_index);
  return (gathered * combine_weights.unsqueeze(-1).to(gathered.scalar_type())).sum(2);
}

c10::optional<torch::Tensor> TryNativeCausalModelForward(
    const py::object& model,
    const torch::Tensor& input_ids,
    const py::object& attention_mask,
    const py::object& cache,
    const py::object& position_ids_obj,
    const py::object& cache_position_obj,
    bool skip_model_support_check,
    bool model_uses_attention_biases) {
  if (skip_model_support_check) {
    if (!SupportsNativeCausalRuntimeInputs(attention_mask, cache)) {
      return c10::nullopt;
    }
  } else if (!IsSupportedNativeCausalModel(model, attention_mask, cache)) {
    return c10::nullopt;
  }
  auto embed = model.attr("embed");
  auto norm = model.attr("norm");
  auto lm_head = model.attr("lm_head");
  int64_t padding_idx = -1;
  auto padding_idx_obj = PyAttrOrNone(embed, "padding_idx");
  if (!padding_idx_obj.is_none()) {
    padding_idx = py::cast<int64_t>(padding_idx_obj);
  }
  auto x = EmbeddingForward(TensorAttr(embed, "weight"), input_ids, padding_idx);

  auto provided_attention_mask = OptionalTensorFromPyObject(attention_mask);
  c10::optional<torch::Tensor> token_attention_mask = c10::nullopt;
  c10::optional<torch::Tensor> explicit_attention_mask = c10::nullopt;
  if (provided_attention_mask.has_value() && provided_attention_mask.value().defined()) {
    if (IsTokenAttentionMaskTensor(provided_attention_mask.value(), x.size(0), x.size(1))) {
      token_attention_mask = provided_attention_mask;
    } else {
      explicit_attention_mask = provided_attention_mask;
    }
  }
  auto cache_position = OptionalTensorFromPyObject(cache_position_obj);
  auto position_ids = OptionalTensorFromPyObject(position_ids_obj);
  if (!position_ids.has_value()) {
    int64_t past_length = 0;
    if (!cache.is_none() && !cache_position.has_value()) {
      past_length = NativeCacheLayerLength(cache, 0);
    }
    position_ids = ResolvePositionIdsForward(
        x.size(0),
        x.size(1),
        x,
        token_attention_mask,
        cache_position,
        past_length);
  }
  auto blocks = model.attr("blocks");
  c10::optional<torch::Tensor> mask = c10::nullopt;
  if (explicit_attention_mask.has_value()) {
    TORCH_CHECK(py::len(blocks) > 0, "supported causal executor requires at least one block");
    auto first_block = py::reinterpret_borrow<py::object>(blocks[py::int_(0)]);
    auto first_attn = first_block.attr("attn");
    const auto num_heads = PyIntAttr(first_attn, "n_heads", 0);
    TORCH_CHECK(num_heads > 0, "supported causal executor requires attention heads");
    mask = PrepareExplicitAttentionMaskForHeads(
        explicit_attention_mask.value(),
        x.size(0),
        num_heads,
        x.size(1),
        explicit_attention_mask.value().size(-1));
  } else if (
      cache.is_none() || x.size(1) != 1 || token_attention_mask.has_value() ||
      model_uses_attention_biases) {
    mask = CreateCausalMaskForward(x, token_attention_mask, cache_position, position_ids);
  }
  int64_t layer_idx = 0;
  for (auto block_handle : blocks) {
    py::object block = py::reinterpret_borrow<py::object>(block_handle);
    const auto block_type = PyTypeName(block);
    auto attn = block.attr("attn");
    auto bc = block.attr("bc");
    const auto residual_scale = PyFloatAttr(bc, "residual_scale", 1.0);
    if (block_type == "ParallelTransformerBlock") {
      auto n = block.attr("n");
      auto mlp = block.attr("mlp");
      auto y = ApplyNormModuleForward(x, n);
      auto attn_out = ExecuteSupportedAttention(attn, y, mask, cache, layer_idx, position_ids);
      auto mlp_out = ExecuteSupportedMlp(mlp, y);
      x = ResidualAddForward(x, attn_out, residual_scale);
      x = ResidualAddForward(x, mlp_out, residual_scale);
      ++layer_idx;
      continue;
    }
    const auto block_mask = ApplySupportedAttentionBiases(block, x, mask, position_ids);
    const auto norm_policy = PyStringAttr(bc, "norm_policy", "prenorm");
    const bool moe_block = block_type == "MoEBlock";
    auto n1 = block.attr("n1");
    auto n2 = block.attr("n2");
    py::object mlp = moe_block ? py::object(py::none()) : block.attr("mlp");
    py::object moe = moe_block ? block.attr("moe") : py::object(py::none());
    if (norm_policy == "prenorm") {
      auto attn_in = ApplyNormModuleForward(x, n1);
      auto attn_out = ExecuteSupportedAttention(attn, attn_in, block_mask, cache, layer_idx, position_ids);
      auto add_norm = AddNormModuleForward(x, attn_out, n2, residual_scale);
      x = add_norm[0];
      auto mlp_out = moe_block ? ExecuteSupportedMoE(moe, add_norm[1]) : ExecuteSupportedMlp(mlp, add_norm[1]);
      x = ResidualAddForward(x, mlp_out, residual_scale);
    } else {
      auto attn_out = ExecuteSupportedAttention(attn, x, block_mask, cache, layer_idx, position_ids);
      x = AddNormModuleForward(x, attn_out, n1, residual_scale)[1];
      auto mlp_out = moe_block ? ExecuteSupportedMoE(moe, x) : ExecuteSupportedMlp(mlp, x);
      x = AddNormModuleForward(x, mlp_out, n2, residual_scale)[1];
    }
    ++layer_idx;
  }

  BitNetModuleState lm_head_state;
  if (BitNetModuleDirectSupported(lm_head, &lm_head_state) &&
      CanUseFusedRmsNormBitNetLinearStateForward(x, norm, lm_head_state)) {
    return FusedRmsNormBitNetLinearStateForward(x, norm, lm_head_state);
  }
  x = ApplyNormModuleForward(x, norm);
  return LinearLikeModuleForward(x, lm_head, "auto");
}

std::string ResolveLinearBackendPy(const std::string& requested) {
  return ResolveLinearBackend(requested);
}

}  // namespace

PYBIND11_MODULE(_model_stack_native, m) {
  m.doc() = "Model-stack native C++/CUDA extension boundary.";
  py::class_<NativeModelSession>(m, "NativeModelSession")
      .def(py::init<py::object, const torch::Tensor&, py::object, py::object, bool>(),
           py::arg("model"),
           py::arg("seq"),
           py::arg("attention_mask") = py::none(),
           py::arg("cache") = py::none(),
           py::arg("trace") = false)
      .def_property("seq", &NativeModelSession::GetSeq, &NativeModelSession::SetSeq)
      .def_property("attention_mask", &NativeModelSession::GetAttentionMask, &NativeModelSession::SetAttentionMask)
      .def_property("cache", &NativeModelSession::GetCache, &NativeModelSession::SetCache)
      .def_property_readonly("batch_size", &NativeModelSession::BatchSize)
      .def_property_readonly("seq_len", &NativeModelSession::SeqLen)
      .def_property_readonly("native_executor_kind", &NativeModelSession::NativeExecutorKind)
      .def("disable_cache", &NativeModelSession::DisableCache)
      .def("prefill_next_logits", &NativeModelSession::PrefillNextLogits)
      .def("full_next_logits", &NativeModelSession::FullNextLogits)
      .def("append", &NativeModelSession::Append, py::arg("next_id"))
      .def("decode_positions", &NativeModelSession::DecodePositions)
      .def("decode_graph_eligible", &NativeModelSession::DecodeGraphEligible)
      .def("set_decode_graph_enabled", &NativeModelSession::SetDecodeGraphEnabled, py::arg("enabled") = true)
      .def("decode_next_logits", &NativeModelSession::DecodeNextLogits)
      .def("reorder_cache_", &NativeModelSession::ReorderCache, py::arg("row_ids"), py::arg("source_cache") = py::none())
      .def("append_attention_mask_", &NativeModelSession::AppendAttentionMask, py::arg("row_ids") = py::none(), py::arg("source_attention_mask") = py::none())
      .def("evict_if_needed", &NativeModelSession::EvictIfNeeded, py::arg("max_tokens"), py::arg("policy") = "sliding-window")
      .def("advance_beam_decode", &NativeModelSession::AdvanceBeamDecode,
           py::arg("next_beams"),
           py::arg("cache_row_ids"),
           py::arg("mask_row_ids") = py::none(),
           py::arg("source_attention_mask") = py::none(),
           py::arg("source_cache") = py::none(),
           py::arg("max_tokens") = -1,
           py::arg("policy") = "sliding-window");
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
      .def("clone_rows", &PagedKvLayerState::CloneRows, py::arg("row_ids"))
      .def("reorder_rows_", &PagedKvLayerState::ReorderRowsInPlace, py::arg("row_ids"))
      .def("k_pages", &PagedKvLayerState::KPages)
      .def("v_pages", &PagedKvLayerState::VPages)
      .def("block_table", &PagedKvLayerState::BlockTable)
      .def("lengths", &PagedKvLayerState::Lengths)
      .def("page_count", &PagedKvLayerState::PageCount)
      .def("max_length", &PagedKvLayerState::MaxLength);
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
      .def("clone_rows", &PagedKvCacheState::CloneRows, py::arg("row_ids"))
      .def("reorder_rows_", &PagedKvCacheState::ReorderRowsInPlace, py::arg("row_ids"))
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
  m.def("int3_kv_pack_forward", &Int3KvPackForward, py::arg("x"));
  m.def("int3_kv_dequantize_forward", &Int3KvDequantizeForward, py::arg("packed"), py::arg("scale"),
        py::arg("original_last_dim"), py::arg("out_dtype") = py::none());
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
  m.def("paged_attention_decode_forward", &PagedAttentionDecodeForward, py::arg("q"), py::arg("k_pages"),
        py::arg("v_pages"), py::arg("block_table"), py::arg("lengths"), py::arg("attn_mask") = py::none(),
        py::arg("scale") = py::none());
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
  m.def("attention_partitioned_reference_forward", &ReferenceAttentionPartitionedForward, py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("attn_mask") = py::none(), py::arg("is_causal") = false,
        py::arg("scale") = py::none(), py::arg("kv_chunk_size"));
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
  m.def("speculative_accept_forward", &SpeculativeAcceptForward, py::arg("target_probs"), py::arg("draft_probs"),
        py::arg("draft_token_ids"), py::arg("bonus_probs") = py::none(), py::arg("bonus_enabled") = py::none(),
        py::arg("method") = "rejection_sampler", py::arg("posterior_threshold") = 0.09,
        py::arg("posterior_alpha") = 0.3);
  m.def("beam_search_step_forward", &BeamSearchStepForward, py::arg("beams"), py::arg("logits"),
        py::arg("raw_scores"), py::arg("finished"), py::arg("lengths"), py::arg("beam_size"),
        py::arg("eos_id"), py::arg("pad_id"));
  m.def("incremental_beam_search_forward", &IncrementalBeamSearchForward, py::arg("initial_beams"),
        py::arg("initial_logits"), py::arg("beam_size"), py::arg("max_new_tokens"), py::arg("prompt_length"),
        py::arg("eos_id"), py::arg("pad_id"), py::arg("advance_fn"));
  m.def("repetition_penalty_forward", &RepetitionPenaltyForward, py::arg("logits"), py::arg("token_ids"),
        py::arg("penalty"));
  m.def("token_counts_forward", &TokenCountsForward, py::arg("token_ids"), py::arg("vocab_size"),
        py::arg("counts_dtype"));
  m.def("append_tokens_forward", &AppendTokensForward, py::arg("seq"), py::arg("next_id"),
        py::arg("attention_mask") = py::none(), py::arg("row_ids") = py::none());
  m.def("decode_positions_forward", &DecodePositionsForward, py::arg("batch_size"), py::arg("seq_len"),
        py::arg("reference"));
  m.def("activation_forward", &ApplyActivation, py::arg("x"), py::arg("activation") = "gelu");
  m.def("gated_activation_forward", &ApplyGatedActivation, py::arg("x"), py::arg("activation") = "swiglu");
  m.def("linear_forward", &LinearForward, py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
        py::arg("backend") = "auto");
  m.def("linear_module_forward", &LinearLikeModuleForward, py::arg("x"), py::arg("module"),
        py::arg("backend") = "auto");
  m.def("nf4_linear_forward", &Nf4LinearForward, py::arg("x"), py::arg("packed_weight"),
        py::arg("weight_scale"), py::arg("bias") = py::none());
  m.def("fp8_linear_forward", &Fp8LinearForward, py::arg("x"), py::arg("weight_fp8"),
        py::arg("weight_scale"), py::arg("bias") = py::none(), py::arg("out_dtype") = py::none());
  m.def("int8_quantize_activation_forward", &Int8QuantizeActivationForward, py::arg("x"),
        py::arg("provided_scale") = py::none());
  m.def("int8_quantize_activation_transpose_forward", &Int8QuantizeActivationTransposeForward, py::arg("x"),
        py::arg("provided_scale") = py::none());
  m.def("int8_quantize_activation_columnwise_forward", &CudaInt8QuantizeActivationColumnwiseForward, py::arg("x"),
        py::arg("provided_scale") = py::none());
  m.def("int8_quantize_relu2_activation_forward", &Int8QuantizeRelu2ActivationForward, py::arg("x"),
        py::arg("act_quant_bits") = 8);
  m.def("int8_quantize_leaky_relu_half2_activation_forward", &Int8QuantizeLeakyReluHalf2ActivationForward, py::arg("x"),
        py::arg("act_quant_bits") = 8);
  m.def(
      "int8_linear_forward",
      [](const torch::Tensor& qx,
         const torch::Tensor& x_scale,
         const torch::Tensor& qweight,
         const torch::Tensor& inv_scale,
         const c10::optional<torch::Tensor>& bias,
         const c10::optional<torch::ScalarType>& out_dtype) {
        return Int8LinearForward(qx, x_scale, qweight, inv_scale, bias, out_dtype);
      },
      py::arg("qx"),
      py::arg("x_scale"),
      py::arg("qweight"),
      py::arg("inv_scale"),
      py::arg("bias") = py::none(),
      py::arg("out_dtype") = py::none());
  m.def("int8_linear_from_float_forward", &Int8LinearFromFloatForward, py::arg("x"), py::arg("qweight"),
        py::arg("inv_scale"), py::arg("bias") = py::none(), py::arg("provided_scale") = py::none(),
        py::arg("out_dtype") = py::none());
  m.def("int8_linear_grad_weight_from_float_forward", &Int8LinearGradWeightFromFloatForward, py::arg("grad_out"),
        py::arg("x"), py::arg("out_dtype") = py::none());
  m.def("int8_linear_accum_forward", &CublasLtInt8LinearAccumForward, py::arg("qx"), py::arg("qweight"));
  m.def("int8_attention_forward", &Int8AttentionForward, py::arg("q"), py::arg("q_scale"), py::arg("k"),
        py::arg("k_scale"), py::arg("v"), py::arg("v_scale"), py::arg("attn_mask") = py::none(),
        py::arg("is_causal") = false, py::arg("scale") = py::none(), py::arg("out_dtype") = py::none());
  m.def("int8_attention_from_float_forward", &Int8AttentionFromFloatForward, py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("attn_mask") = py::none(), py::arg("is_causal") = false,
        py::arg("scale") = py::none(), py::arg("out_dtype") = py::none(),
        py::arg("q_provided_scale") = py::none(), py::arg("k_provided_scale") = py::none(),
        py::arg("v_provided_scale") = py::none());
  m.def("bitnet_transform_input_forward", &BitNetTransformInputForward, py::arg("x"),
        py::arg("spin_enabled") = false, py::arg("spin_signs") = py::none(), py::arg("pre_scale") = py::none(),
        py::arg("act_quant_mode") = "none", py::arg("act_quant_method") = "absmax",
        py::arg("act_quant_bits") = 8, py::arg("act_quant_percentile") = 0.999, py::arg("act_scale") = py::none());
  m.def("bitnet_linear_from_float_forward", &BitNetLinearFromFloatForward, py::arg("x"),
        py::arg("packed_weight"), py::arg("scale_values"), py::arg("layout_header"), py::arg("segment_offsets"),
        py::arg("bias") = py::none(), py::arg("spin_enabled") = false, py::arg("spin_signs") = py::none(),
        py::arg("pre_scale") = py::none(), py::arg("act_quant_mode") = "none",
        py::arg("act_quant_method") = "absmax", py::arg("act_quant_bits") = 8,
        py::arg("act_quant_percentile") = 0.999, py::arg("act_scale") = py::none(),
        py::arg("out_dtype") = py::none());
  m.def(
      "bitnet_int8_linear_from_float_forward",
      [](const torch::Tensor& x,
         const torch::Tensor& qweight,
         const torch::Tensor& inv_scale,
         const c10::optional<torch::Tensor>& bias,
         const c10::optional<torch::Tensor>& pre_scale,
         const std::string& act_quant_mode,
         const std::string& act_quant_method,
         int64_t act_quant_bits,
         double act_quant_percentile,
         const c10::optional<torch::Tensor>& act_scale,
         const c10::optional<torch::ScalarType>& out_dtype) {
        return BitNetInt8LinearFromFloatForward(
            x,
            qweight,
            inv_scale,
            bias,
            pre_scale,
            act_quant_mode,
            act_quant_method,
            act_quant_bits,
            act_quant_percentile,
            act_scale,
            out_dtype);
      },
      py::arg("x"),
      py::arg("qweight"),
      py::arg("inv_scale"),
      py::arg("bias") = py::none(),
      py::arg("pre_scale") = py::none(),
      py::arg("act_quant_mode") = "dynamic_int8",
      py::arg("act_quant_method") = "absmax",
      py::arg("act_quant_bits") = 8,
      py::arg("act_quant_percentile") = 0.999,
      py::arg("act_scale") = py::none(),
      py::arg("out_dtype") = py::none());
  m.def(
      "bitnet_int8_fused_qkv_packed_heads_projection_forward",
      &BitNetInt8FusedQkvPackedHeadsProjectionForward,
      py::arg("x"),
      py::arg("qweight"),
      py::arg("inv_scale"),
      py::arg("packed_bias") = py::none(),
      py::arg("pre_scale") = py::none(),
      py::arg("act_quant_mode") = "dynamic_int8",
      py::arg("act_quant_method") = "absmax",
      py::arg("act_quant_bits") = 8,
      py::arg("act_quant_percentile") = 0.999,
      py::arg("act_scale") = py::none(),
      py::arg("q_size"),
      py::arg("k_size"),
      py::arg("v_size"),
      py::arg("q_heads"),
      py::arg("kv_heads"),
      py::arg("out_dtype") = py::none());
  m.def("bitnet_linear_forward", &BitNetLinearForward, py::arg("x"), py::arg("packed_weight"),
        py::arg("scale_values"), py::arg("layout_header"), py::arg("segment_offsets"),
        py::arg("bias") = py::none(), py::arg("out_dtype") = py::none(),
        py::arg("debug_dense_fallback") = false);
#if MODEL_STACK_WITH_CUDA
  m.def(
      "bitnet_linear_compute_packed_forward",
      [](const torch::Tensor& x,
         const torch::Tensor& packed_weight,
         const torch::Tensor& scale_values,
         const torch::Tensor& layout_header,
         const torch::Tensor& segment_offsets,
         const torch::Tensor& compute_packed_words,
         const torch::Tensor& compute_row_scales,
         const torch::Tensor& decode_nz_masks,
         const torch::Tensor& decode_sign_masks,
         const torch::Tensor& decode_row_scales,
         const c10::optional<torch::Tensor>& bias,
         const c10::optional<torch::ScalarType>& out_dtype) {
        return t10::bitnet::CudaBitNetLinearForwardComputePacked(
            x,
            packed_weight,
            scale_values,
            layout_header,
            segment_offsets,
            compute_packed_words,
            compute_row_scales,
            decode_nz_masks,
            decode_sign_masks,
            decode_row_scales,
            bias,
            out_dtype);
      },
      py::arg("x"),
      py::arg("packed_weight"),
      py::arg("scale_values"),
      py::arg("layout_header"),
      py::arg("segment_offsets"),
      py::arg("compute_packed_words"),
      py::arg("compute_row_scales"),
      py::arg("decode_nz_masks"),
      py::arg("decode_sign_masks"),
      py::arg("decode_row_scales"),
      py::arg("bias") = py::none(),
      py::arg("out_dtype") = py::none());
#endif
  m.def("int4_linear_forward", &Int4LinearForward, py::arg("x"), py::arg("packed_weight"),
        py::arg("inv_scale"), py::arg("bias") = py::none());
  m.def("int4_linear_grad_input_forward", &Int4LinearGradInputForward, py::arg("grad_out"),
        py::arg("packed_weight"), py::arg("inv_scale"), py::arg("in_features"));
  m.def("cutlass_int4_bf16_linear_forward", &CutlassInt4Bf16LinearForwardChecked, py::arg("x"),
        py::arg("packed_weight_rowmajor"), py::arg("scale"), py::arg("bias") = py::none(),
        py::arg("packed_weight_is_shuffled") = false);
  m.def("cutlass_int4_pack_shuffled_forward", &CutlassInt4PackShuffledForwardChecked, py::arg("qweight"));
  m.def("pack_bitnet_weight_forward", &PackBitNetWeightForward, py::arg("weight"),
        py::arg("scale_values") = py::none(), py::arg("layout_header") = py::none(),
        py::arg("segment_offsets") = py::none());
  m.def(
      "bitnet_runtime_row_quantize_forward",
      &BitNetRuntimeRowQuantizeForward,
      py::arg("weight"),
      py::arg("eps") = 1.0e-8);
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
  m.def("bitnet_qkv_packed_heads_projection_forward", &BitNetQkvPackedHeadsProjectionForward,
        py::arg("q_x"), py::arg("q_packed_weight"), py::arg("q_scale_values"), py::arg("q_layout_header"),
        py::arg("q_segment_offsets"), py::arg("q_bias") = py::none(), py::arg("k_x"),
        py::arg("k_packed_weight"), py::arg("k_scale_values"), py::arg("k_layout_header"),
        py::arg("k_segment_offsets"), py::arg("k_bias") = py::none(), py::arg("v_x"),
        py::arg("v_packed_weight"), py::arg("v_scale_values"), py::arg("v_layout_header"),
        py::arg("v_segment_offsets"), py::arg("v_bias") = py::none(), py::arg("q_heads"),
        py::arg("kv_heads"), py::arg("out_dtype") = py::none());
  m.def("bitnet_fused_qkv_packed_heads_projection_forward", &BitNetFusedQkvPackedHeadsProjectionForward,
        py::arg("x"), py::arg("packed_weight"), py::arg("scale_values"), py::arg("layout_header"),
        py::arg("segment_offsets"), py::arg("packed_bias") = py::none(), py::arg("q_size"), py::arg("k_size"),
        py::arg("v_size"), py::arg("q_heads"), py::arg("kv_heads"));
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
