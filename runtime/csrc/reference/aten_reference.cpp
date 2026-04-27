#include "aten_reference.h"

#include <ATen/ops/gelu.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/silu.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace {

std::string NormalizeActivation(const std::string& name) {
  std::string out = name;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return out;
}

torch::Tensor ApplyActivationReference(
    const torch::Tensor& x,
    const std::string& activation) {
  const auto act = NormalizeActivation(activation);
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

std::pair<torch::Tensor, torch::Tensor> ExpandAttentionKvHeadsReference(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  auto k_all = k;
  auto v_all = v;
  if (q.size(1) != k.size(1)) {
    const auto repeat = q.size(1) / k.size(1);
    auto head_index = torch::arange(k.size(1), torch::TensorOptions().dtype(torch::kLong).device(k.device()))
                          .repeat_interleave(repeat);
    k_all = at::index_select(k, 1, head_index);
    v_all = at::index_select(v, 1, head_index);
  }
  return {k_all, v_all};
}

}  // namespace

torch::Tensor ReferenceLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias) {
  return at::linear(x, weight, bias);
}

torch::Tensor ReferenceMlpForward(
    const torch::Tensor& x,
    const torch::Tensor& w_in_weight,
    const c10::optional<torch::Tensor>& w_in_bias,
    const torch::Tensor& w_out_weight,
    const c10::optional<torch::Tensor>& w_out_bias,
    const std::string& activation,
    bool gated) {
  auto hidden = ReferenceLinearForward(x, w_in_weight, w_in_bias);
  torch::Tensor projected;
  if (gated) {
    TORCH_CHECK(hidden.size(-1) % 2 == 0, "ReferenceMlpForward: gated projection width must be even");
    const auto split = hidden.size(-1) / 2;
    auto a = hidden.slice(-1, 0, split);
    auto b = hidden.slice(-1, split, hidden.size(-1));
    projected = ApplyActivationReference(a, activation) * b;
  } else {
    projected = ApplyActivationReference(hidden, activation);
  }
  return ReferenceLinearForward(projected, w_out_weight, w_out_bias);
}

torch::Tensor ReferenceAttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale) {
  auto [k_all, v_all] = ExpandAttentionKvHeadsReference(q, k, v);

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

  if (is_causal && q.size(2) > 1) {
    const auto tgt = q.size(2);
    const auto src = k_all.size(2);
    auto causal = torch::ones({tgt, src}, torch::TensorOptions().dtype(torch::kBool).device(q.device())).triu(1);
    scores = scores.masked_fill(causal.view({1, 1, tgt, src}), -std::numeric_limits<float>::infinity());
  }

  auto probs = torch::softmax(scores.to(torch::kFloat32), -1).to(q.scalar_type());
  return torch::matmul(probs, v_all);
}

torch::Tensor ReferenceAttentionPartitionedForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale,
    int64_t kv_chunk_size) {
  TORCH_CHECK(kv_chunk_size > 0, "ReferenceAttentionPartitionedForward: kv_chunk_size must be positive");

  auto [k_all, v_all] = ExpandAttentionKvHeadsReference(q, k, v);

  const double scale_value =
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(static_cast<double>(q.size(3))));
  const auto tgt_len = q.size(2);
  const auto src_len = k_all.size(2);
  const auto float_opts = q.options().dtype(torch::kFloat32);
  auto running_max = torch::full({q.size(0), q.size(1), tgt_len, 1}, -std::numeric_limits<float>::infinity(), float_opts);
  auto running_denom = torch::zeros({q.size(0), q.size(1), tgt_len, 1}, float_opts);
  auto numerator = torch::zeros({q.size(0), q.size(1), tgt_len, v_all.size(3)}, float_opts);

  const auto v_float = v_all.to(torch::kFloat32);

  for (int64_t chunk_start = 0; chunk_start < src_len; chunk_start += kv_chunk_size) {
    const auto chunk_len = std::min<int64_t>(kv_chunk_size, src_len - chunk_start);
    auto k_chunk = k_all.narrow(2, chunk_start, chunk_len);
    auto v_chunk = v_float.narrow(2, chunk_start, chunk_len);

    auto scores =
        torch::matmul(q, k_chunk.transpose(-2, -1)).to(torch::kFloat32) * static_cast<float>(scale_value);

    if (attn_mask.has_value() && attn_mask.value().defined()) {
      auto mask_chunk = attn_mask.value().narrow(-1, chunk_start, chunk_len);
      if (mask_chunk.scalar_type() == torch::kBool) {
        scores = scores.masked_fill(mask_chunk, -std::numeric_limits<float>::infinity());
      } else {
        scores = scores + mask_chunk.to(torch::kFloat32);
      }
    }

    if (is_causal && tgt_len > 1) {
      auto q_pos = torch::arange(tgt_len, torch::TensorOptions().dtype(torch::kLong).device(q.device()))
                       .view({tgt_len, 1});
      auto k_pos = torch::arange(
                       chunk_start,
                       chunk_start + chunk_len,
                       torch::TensorOptions().dtype(torch::kLong).device(q.device()))
                       .view({1, chunk_len});
      auto causal = k_pos > q_pos;
      scores = scores.masked_fill(causal.view({1, 1, tgt_len, chunk_len}), -std::numeric_limits<float>::infinity());
    }

    auto local_max = std::get<0>(scores.max(-1, true));
    auto safe_local_max = torch::where(
        torch::isfinite(local_max),
        local_max,
        torch::zeros_like(local_max));
    auto exp_scores = torch::exp(scores - safe_local_max);
    auto local_sum = exp_scores.sum(-1, true);
    auto local_weighted = torch::matmul(exp_scores, v_chunk);

    auto new_max = torch::maximum(running_max, local_max);
    auto safe_new_max = torch::where(
        torch::isfinite(new_max),
        new_max,
        torch::zeros_like(new_max));
    auto alpha = torch::where(
        torch::isfinite(running_max),
        torch::exp(running_max - safe_new_max),
        torch::zeros_like(running_max));
    auto beta = torch::where(
        torch::isfinite(local_max),
        torch::exp(local_max - safe_new_max),
        torch::zeros_like(local_max));

    numerator = numerator * alpha + local_weighted * beta;
    running_denom = running_denom * alpha + local_sum * beta;
    running_max = new_max;
  }

  auto out = numerator / running_denom.clamp_min(1e-20f);
  return out.to(q.scalar_type());
}
