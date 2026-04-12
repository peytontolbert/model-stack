#pragma once

#include <torch/extension.h>

#include <string>

torch::Tensor ReferenceLinearForward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias);

torch::Tensor ReferenceMlpForward(
    const torch::Tensor& x,
    const torch::Tensor& w_in_weight,
    const c10::optional<torch::Tensor>& w_in_bias,
    const torch::Tensor& w_out_weight,
    const c10::optional<torch::Tensor>& w_out_bias,
    const std::string& activation,
    bool gated);

torch::Tensor ReferenceAttentionForward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_mask,
    bool is_causal,
    const c10::optional<double>& scale);
