import torch


def per_channel_absmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    return x.abs().amax(dim=axis, keepdim=True)


def nf4_quantize(x: torch.Tensor, scale: torch.Tensor):
    # Simplified placeholder quantization to 16 buckets (NF4-like)
    q = torch.clamp((x / (scale + 1e-6)) * 7.0, -7.0, 7.0)
    q = torch.round(q).to(torch.int8)
    meta = scale
    return q, meta


def nf4_dequantize(qx: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    return (qx.float() / 7.0) * meta


def int8_matmul_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_scales: torch.Tensor, k_scales: torch.Tensor, v_scales: torch.Tensor) -> torch.Tensor:
    # Dequantize then matmul; placeholder glue
    qf = q.float() * q_scales
    kf = k.float() * k_scales
    vf = v.float() * v_scales
    attn = torch.softmax(qf @ kf.transpose(-2, -1) / (qf.shape[-1] ** 0.5), dim=-1)
    return attn @ vf


def fp8_linear(x: torch.Tensor, weight_fp8: torch.Tensor, amax_tracker, scale: float, bias: torch.Tensor | None = None) -> torch.Tensor:
    # Dequantize-like multiply
    amax_tracker.update(weight_fp8)
    y = x @ (weight_fp8.float() * scale).t()
    if bias is not None:
        y = y + bias
    return y


