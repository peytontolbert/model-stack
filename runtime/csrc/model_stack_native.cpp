#include <torch/extension.h>
#include <ATen/TensorIndexing.h>
#include <ATen/ops/_unique2.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/index_select.h>
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
torch::Tensor CublasLtLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias);
bool HasCublasLtLinearBackend();
#else
bool HasCudaRmsNormKernel() {
  return false;
}
bool HasCublasLtLinearBackend() {
  return false;
}
#endif

namespace {

constexpr int kAbiVersion = MODEL_STACK_ABI_VERSION;

std::map<std::string, bool> NativeOpMap() {
  return {
      {"embedding", true},
      {"linear", true},
      {"mlp", true},
      {"qkv_projection", true},
      {"rms_norm", true},
      {"rope", true},
      {"kv_cache_append", true},
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

py::dict RuntimeInfo() {
  py::dict info;
  info["abi_version"] = kAbiVersion;
#if MODEL_STACK_WITH_CUDA
  info["compiled_with_cuda"] = true;
#else
  info["compiled_with_cuda"] = false;
#endif
  info["native_ops"] = std::vector<std::string>{
      "embedding", "linear", "mlp", "qkv_projection", "rms_norm", "rope", "kv_cache_append", "attention_decode", "attention_prefill", "sampling"};
  info["planned_ops"] = std::vector<std::string>{
      "embedding", "linear", "mlp", "qkv_projection", "rms_norm", "rope", "kv_cache_append", "attention_decode",
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
  if (HasCublasLtLinearBackend()) {
    cuda_backend_ops.push_back("linear");
    cuda_backend_ops.push_back("qkv_projection");
  }
  info["cuda_backend_ops"] = cuda_backend_ops;
  return info;
}

bool HasOp(const std::string& name) {
  const auto ops = NativeOpMap();
  const auto it = ops.find(name);
  return it != ops.end() && it->second;
}

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
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "attention_forward: head mismatch");
  TORCH_CHECK(k.size(2) == v.size(2), "attention_forward: source-length mismatch");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "attention_forward: head_dim mismatch");

  const double scale_value = scale.has_value() ? scale.value() : (1.0 / std::sqrt(static_cast<double>(q.size(3))));
  auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale_value;

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
    auto src = k.size(2);
    auto causal = torch::ones({tgt, src}, torch::TensorOptions().dtype(torch::kBool).device(q.device())).triu(1);
    scores = scores.masked_fill(causal.view({1, 1, tgt, src}), -std::numeric_limits<float>::infinity());
  }

  auto probs = torch::softmax(scores.to(torch::kFloat32), -1).to(q.scalar_type());
  return torch::matmul(probs, v);
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
  const auto resolved_backend = ResolveLinearBackend(backend);
  auto hidden = LinearForward(x, w_in_weight, w_in_bias, resolved_backend);
  torch::Tensor projected;
  if (gated) {
    TORCH_CHECK(hidden.size(-1) % 2 == 0, "mlp_forward: gated projection width must be even");
    const auto split = hidden.size(-1) / 2;
    auto a = hidden.slice(-1, 0, split);
    auto b = hidden.slice(-1, split, hidden.size(-1));
    projected = ApplyActivation(a, activation) * b;
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
  m.def("apply_rotary_forward", &ApplyRotaryForward, py::arg("q"), py::arg("k"), py::arg("cos"),
        py::arg("sin"));
  m.def("kv_cache_append_forward", &KvCacheAppendForward, py::arg("k_cache") = py::none(),
        py::arg("v_cache") = py::none(), py::arg("k_new"), py::arg("v_new"));
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
  m.def("mlp_forward", &MlpForward, py::arg("x"), py::arg("w_in_weight"), py::arg("w_in_bias") = py::none(),
        py::arg("w_out_weight"), py::arg("w_out_bias") = py::none(), py::arg("activation") = "gelu",
        py::arg("gated") = false, py::arg("backend") = "auto");
  m.def("qkv_projection_forward", &QkvProjectionForward, py::arg("x"), py::arg("q_weight"),
        py::arg("q_bias") = py::none(), py::arg("k_weight"), py::arg("k_bias") = py::none(),
        py::arg("v_weight"), py::arg("v_bias") = py::none(), py::arg("backend") = "auto");
  m.def("embedding_forward", &EmbeddingForward, py::arg("weight"), py::arg("indices"),
        py::arg("padding_idx") = -1);
}
