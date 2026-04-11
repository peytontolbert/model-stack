#include <torch/extension.h>
#include <ATen/TensorIndexing.h>
#include <ATen/ops/_unique2.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/silu.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

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
torch::Tensor CublasLtLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias);
bool HasCublasLtLinearBackend();
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
bool HasCudaLayerNormKernel() {
  return false;
}
bool HasCudaAddLayerNormKernel() {
  return false;
}
bool HasCublasLtLinearBackend() {
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

std::map<std::string, bool> NativeOpMap() {
  return {
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
      {"rms_norm", true},
      {"add_rms_norm", true},
      {"layer_norm", true},
      {"add_layer_norm", true},
      {"rope", true},
      {"kv_cache_append", true},
      {"kv_cache_write", true},
      {"attention_decode", true},
      {"attention_prefill", true},
      {"sampling", true},
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

torch::Tensor ApplyActivation(const torch::Tensor& x, const std::string& activation) {
  const auto act = NormalizeBackendName(activation);
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
      "embedding", "linear", "pack_linear_weight", "mlp", "qkv_projection", "pack_qkv_weights", "qkv_packed_heads_projection", "qkv_heads_projection", "split_heads", "merge_heads", "head_output_projection", "prepare_attention_mask", "rms_norm", "add_rms_norm", "layer_norm", "add_layer_norm", "rope", "kv_cache_append", "kv_cache_write", "attention_decode", "attention_prefill", "sampling"};
  info["planned_ops"] = std::vector<std::string>{
      "embedding", "linear", "pack_linear_weight", "mlp", "qkv_projection", "pack_qkv_weights", "qkv_packed_heads_projection", "qkv_heads_projection", "split_heads", "merge_heads", "head_output_projection", "prepare_attention_mask", "rms_norm", "add_rms_norm", "layer_norm", "add_layer_norm", "rope", "kv_cache_append", "kv_cache_write", "attention_decode",
      "attention_prefill", "sampling"};
  info["linear_backend_default"] = HasCublasLtLinearBackend() ? "cublaslt" : "aten";
  info["linear_backends_supported"] = SupportedLinearBackends();
  info["linear_backends_planned"] = PlannedLinearBackends();
  info["cublaslt_linear_dtypes"] = HasCublasLtLinearBackend()
      ? std::vector<std::string>{"float32", "float16"}
      : std::vector<std::string>{};
  std::vector<std::string> cuda_backend_ops;
  if (HasCudaRmsNormKernel()) {
    cuda_backend_ops.push_back("rms_norm");
  }
  if (HasCudaAddRmsNormKernel()) {
    cuda_backend_ops.push_back("add_rms_norm");
  }
  if (HasCudaLayerNormKernel()) {
    cuda_backend_ops.push_back("layer_norm");
  }
  if (HasCudaAddLayerNormKernel()) {
    cuda_backend_ops.push_back("add_layer_norm");
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
      q.scalar_type() != torch::kBFloat16 && HasCudaRopeKernel()) {
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

  auto k_all = k;
  auto v_all = v;
  if (q.size(1) != k.size(1)) {
    const auto repeat = q.size(1) / k.size(1);
    auto head_index = torch::arange(k.size(1), torch::TensorOptions().dtype(torch::kLong).device(k.device()))
                          .repeat_interleave(repeat);
    k_all = at::index_select(k, 1, head_index);
    v_all = at::index_select(v, 1, head_index);
  }

  const double scale_value = scale.has_value() ? scale.value() : (1.0 / std::sqrt(static_cast<double>(q.size(3))));
  auto scores = torch::matmul(q, k_all.transpose(-2, -1)) * scale_value;

  if (attn_mask.has_value() && attn_mask.value().defined()) {
    auto mask = attn_mask.value();
    if (mask.scalar_type() == torch::kBool) {
      scores = scores.masked_fill(mask, -std::numeric_limits<float>::infinity());
    } else {
      scores = scores + mask.to(scores.scalar_type());
    }
  }

  if (is_causal) {
    auto tgt = q.size(2);
    auto src = k_all.size(2);
    auto causal = torch::ones({tgt, src}, torch::TensorOptions().dtype(torch::kBool).device(q.device())).triu(1);
    scores = scores.masked_fill(causal.view({1, 1, tgt, src}), -std::numeric_limits<float>::infinity());
  }

  auto probs = torch::softmax(scores.to(torch::kFloat32), -1).to(q.scalar_type());
  return torch::matmul(probs, v_all);
}

torch::Tensor TemperatureForward(const torch::Tensor& logits, double tau) {
  TORCH_CHECK(logits.defined(), "temperature_forward: logits must be defined");
  TORCH_CHECK(std::isfinite(tau), "temperature_forward: tau must be finite");
  const auto denom = std::max(tau, 1e-8);
  return logits / denom;
}

torch::Tensor TopkMaskForward(const torch::Tensor& logits, int64_t k) {
  TORCH_CHECK(logits.defined(), "topk_mask_forward: logits must be defined");
  TORCH_CHECK(logits.dim() >= 1, "topk_mask_forward: logits must have rank >= 1");
  const auto vocab = logits.size(-1);
  TORCH_CHECK(vocab > 0, "topk_mask_forward: logits last dimension must be non-empty");
  TORCH_CHECK(k > 0, "topk_mask_forward: k must be positive");
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
  auto penalty =
      alpha_presence * counts.gt(0).to(logits.scalar_type()) +
      alpha_frequency * counts.to(logits.scalar_type());
  return logits - penalty;
}

torch::Tensor SampleNextTokenForward(const torch::Tensor& logits, bool do_sample) {
  TORCH_CHECK(logits.defined(), "sample_next_token_forward: logits must be defined");
  TORCH_CHECK(logits.dim() == 2, "sample_next_token_forward: logits must be rank-2 (B, V)");
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

torch::Tensor LinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const std::string& backend) {
  auto AtenLinearForward = [&]() {
    return at::linear(x, weight, bias);
  };
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
  return AtenLinearForward();
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
  if (auto_backend && gated && x.is_cuda() && x.scalar_type() != torch::kFloat32) {
    auto hidden = at::linear(x, w_in_weight, w_in_bias);
    torch::Tensor projected;
    if (hidden.size(-1) % 2 != 0) {
      TORCH_CHECK(false, "mlp_forward: gated projection width must be even");
    }
    const auto split = hidden.size(-1) / 2;
    auto a = hidden.slice(-1, 0, split);
    auto b = hidden.slice(-1, split, hidden.size(-1));
    projected = ApplyActivation(a, activation) * b;
    return at::linear(projected, w_out_weight, w_out_bias);
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
  m.def("runtime_info", &RuntimeInfo);
  m.def("has_op", &HasOp, py::arg("name"));
  m.def("resolve_linear_backend", &ResolveLinearBackendPy, py::arg("requested") = "auto");
  m.def("rms_norm_forward", &RmsNormForward, py::arg("x"), py::arg("weight") = py::none(),
        py::arg("eps") = 1e-6);
  m.def("add_rms_norm_forward", &AddRmsNormForward, py::arg("x"), py::arg("update"),
        py::arg("weight") = py::none(), py::arg("residual_scale") = 1.0, py::arg("eps") = 1e-6);
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
  m.def("prepare_attention_mask_forward", &PrepareAttentionMaskForward, py::arg("mask"), py::arg("batch_size"),
        py::arg("num_heads"), py::arg("tgt_len"), py::arg("src_len"), py::arg("position_ids") = py::none());
  m.def("attention_forward", &NativeAttentionForward, py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("attn_mask") = py::none(), py::arg("is_causal") = false, py::arg("scale") = py::none());
  m.def("temperature_forward", &TemperatureForward, py::arg("logits"), py::arg("tau"));
  m.def("topk_mask_forward", &TopkMaskForward, py::arg("logits"), py::arg("k"));
  m.def("topp_mask_forward", &ToppMaskForward, py::arg("logits"), py::arg("p"));
  m.def("presence_frequency_penalty_forward", &PresenceFrequencyPenaltyForward, py::arg("logits"),
        py::arg("counts"), py::arg("alpha_presence"), py::arg("alpha_frequency"));
  m.def("sample_next_token_forward", &SampleNextTokenForward, py::arg("logits"), py::arg("do_sample"));
  m.def("repetition_penalty_forward", &RepetitionPenaltyForward, py::arg("logits"), py::arg("token_ids"),
        py::arg("penalty"));
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
