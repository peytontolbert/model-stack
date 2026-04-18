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

    def fake_runtime_bitnet_linear(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None):
        del scale_values, segment_offsets, bias
        calls.append((tuple(x.shape), tuple(packed_weight.shape), int(layout_header[3].item())))
        return torch.full(x.shape[:-1] + (int(layout_header[3].item()),), float(len(calls)), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(quant_mod, "runtime_bitnet_linear", fake_runtime_bitnet_linear)

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
    assert calls[0][2] == 8
    assert calls[1][2] == 4
