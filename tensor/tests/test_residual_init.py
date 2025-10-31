import torch
import torch.nn as nn
from residual import residual_bias_dropout_add, gated_residual_add, prenorm, postnorm, residual_alpha_schedule
from init import init_qkv_proj_, init_out_proj_scaled_, mu_param_linear_init_, deepnet_residual_scale


def test_residual_functions_shapes():
    x = torch.randn(2, 4, 8)
    y = torch.randn(2, 4, 8)
    b = torch.zeros(8)
    out = residual_bias_dropout_add(x, y, b, p=0.0, training=False)
    assert out.shape == x.shape
    out2 = gated_residual_add(x, y, torch.ones_like(y))
    assert out2.shape == x.shape


def test_prenorm_postnorm():
    x = torch.randn(2, 3)
    ln = nn.LayerNorm(3)
    f = lambda t: t * 2
    y1 = prenorm(x, ln, f)
    y2 = postnorm(x, ln, f)
    assert y1.shape == x.shape and y2.shape == x.shape


def test_init_helpers():
    lin = nn.Linear(8, 8)
    init_qkv_proj_(lin, d_model=8, n_heads=2)
    init_out_proj_scaled_(lin, n_layers=8)
    mu_param_linear_init_(lin, fan_in=8)
    s = deepnet_residual_scale(16)
    assert s > 0


