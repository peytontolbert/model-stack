from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensor.mlp as tensor_mlp_mod
import compress.quantization as quant_mod


def test_mlp_no_dropout_uses_runtime_mlp_module(monkeypatch):
    seen = {}

    def fake_runtime_mlp_module(x, w_in_module, w_out_module, *, activation, gated, backend=None):
        seen["x"] = tuple(x.shape)
        seen["w_in"] = w_in_module
        seen["w_out"] = w_out_module
        seen["activation"] = activation
        seen["gated"] = gated
        seen["backend"] = backend
        return torch.full(x.shape[:-1] + (w_out_module.out_features,), 2.0, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(tensor_mlp_mod, "runtime_mlp_module", fake_runtime_mlp_module)

    mlp = tensor_mlp_mod.MLP(4, 8, activation="silu", dropout_p=0.0)
    x = torch.randn(2, 3, 4)
    out = mlp(x)

    assert out.shape == (2, 3, 4)
    assert torch.all(out == 2.0)
    assert seen["x"] == (2, 3, 4)
    assert seen["w_in"] is mlp.w_in
    assert seen["w_out"] is mlp.w_out
    assert seen["activation"] == "silu"
    assert seen["gated"] is False
    assert seen["backend"] is None


def test_mlp_no_dropout_threads_leaky_relu_half_squared_activation(monkeypatch):
    seen = {}

    def fake_runtime_mlp_module(x, w_in_module, w_out_module, *, activation, gated, backend=None):
        seen["activation"] = activation
        seen["gated"] = gated
        del w_in_module, w_out_module, backend
        return torch.zeros(x.shape[:-1] + (x.shape[-1],), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(tensor_mlp_mod, "runtime_mlp_module", fake_runtime_mlp_module)

    mlp = tensor_mlp_mod.MLP(4, 8, activation="leaky_relu_0p5_squared", dropout_p=0.0)
    x = torch.randn(2, 3, 4)
    out = mlp(x)

    assert out.shape == (2, 3, 4)
    assert seen["activation"] == "leaky_relu_0p5_squared"
    assert seen["gated"] is False


def test_mlp_dropout_path_uses_runtime_linear_module(monkeypatch):
    calls = []

    def fake_runtime_linear_module(x, module, *, backend=None):
        calls.append((tuple(x.shape), module, backend))
        return torch.full(x.shape[:-1] + (module.out_features,), float(len(calls)), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(tensor_mlp_mod, "runtime_linear_module", fake_runtime_linear_module)

    mlp = tensor_mlp_mod.MLP(4, 8, activation="silu", dropout_p=0.1)
    mlp.train()
    x = torch.randn(2, 3, 4)
    out = mlp(x)

    assert out.shape == (2, 3, 4)
    assert len(calls) == 2
    assert calls[0][0] == (2, 3, 4)
    assert calls[0][1] is mlp.w_in
    assert calls[0][2] is None
    assert calls[1][1] is mlp.w_out
    assert calls[1][2] is None


def test_mlp_uses_bitnet_wrappers_without_dense_runtime_mlp(monkeypatch):
    calls = []

    def fake_dense_linear(x, weight, bias=None):
        calls.append((tuple(x.shape), tuple(weight.shape), bias is not None))
        return torch.full(x.shape[:-1] + (weight.shape[0],), float(len(calls)), dtype=x.dtype, device=x.device)

    def fail_runtime_bitnet_linear(*args, **kwargs):
        raise AssertionError("none-mode BitNet MLP should not use runtime_bitnet_linear")

    def fail_runtime_bitnet_int8_linear_from_float(*args, **kwargs):
        raise AssertionError("none-mode BitNet MLP should not use runtime_bitnet_int8_linear_from_float")

    monkeypatch.setattr(torch.nn.functional, "linear", fake_dense_linear)
    monkeypatch.setattr(quant_mod, "runtime_bitnet_linear", fail_runtime_bitnet_linear)
    monkeypatch.setattr(quant_mod, "runtime_bitnet_int8_linear_from_float", fail_runtime_bitnet_int8_linear_from_float)

    mlp = tensor_mlp_mod.MLP(4, 8, activation="silu", dropout_p=0.0)
    w_in = quant_mod.QuantizedLinearBitNet(4, 8, bias=True).from_float(mlp.w_in)
    w_out = quant_mod.QuantizedLinearBitNet(8, 4, bias=True).from_float(mlp.w_out)
    mlp.w_in = w_in
    mlp.w_out = w_out
    x = torch.randn(2, 3, 4)
    out = mlp(x)

    assert out.shape == (2, 3, 4)
    assert len(calls) == 2
    assert calls[0][0] == (2, 3, 4)
    assert calls[0][1] == (8, 4)
    assert calls[0][2] is True
    assert calls[1][0] == (2, 3, 8)
    assert calls[1][1] == (4, 8)
    assert calls[1][2] is True


def test_mlp_uses_bitnet_int8_helper_when_dynamic_int8_is_enabled(monkeypatch):
    calls = []

    def fail_dense_linear(x, weight, bias=None):
        del x, weight, bias
        raise AssertionError("dynamic_int8 BitNet MLP should not use dense linear fallback")

    def fail_runtime_bitnet_linear(*args, **kwargs):
        raise AssertionError("dynamic_int8 BitNet MLP should use the int8 helper, not runtime_bitnet_linear")

    def fake_runtime_bitnet_int8_linear_from_float(
        x,
        qweight,
        inv_scale,
        bias=None,
        *,
        pre_scale=None,
        act_quant_mode="dynamic_int8",
        act_scale=None,
        act_quant_bits=8,
        act_quant_method="absmax",
        act_quant_percentile=0.999,
    ):
        del inv_scale, bias, pre_scale, act_scale, act_quant_method, act_quant_percentile
        calls.append((tuple(x.shape), tuple(qweight.shape), act_quant_mode, act_quant_bits))
        return torch.full(x.shape[:-1] + (qweight.shape[0],), float(len(calls)), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(torch.nn.functional, "linear", fail_dense_linear)
    monkeypatch.setattr(quant_mod, "runtime_bitnet_linear", fail_runtime_bitnet_linear)
    monkeypatch.setattr(quant_mod, "runtime_bitnet_int8_linear_from_float", fake_runtime_bitnet_int8_linear_from_float)

    mlp = tensor_mlp_mod.MLP(4, 8, activation="silu", dropout_p=0.0)
    w_in = quant_mod.QuantizedLinearBitNet(4, 8, bias=True).from_float(mlp.w_in, activation_quant="dynamic_int8")
    w_out = quant_mod.QuantizedLinearBitNet(8, 4, bias=True).from_float(mlp.w_out, activation_quant="dynamic_int8")
    mlp.w_in = w_in
    mlp.w_out = w_out
    x = torch.randn(2, 3, 4)
    out = mlp(x)

    assert out.shape == (2, 3, 4)
    assert len(calls) == 2
    assert calls[0] == ((2, 3, 4), (8, 4), "dynamic_int8", 8)
    assert calls[1] == ((2, 3, 8), (4, 8), "dynamic_int8", 8)
