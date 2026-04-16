from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runtime.ops as runtime_ops_mod
import tensor.activations as activations_mod


def test_runtime_activation_dispatches_to_native_binding(monkeypatch):
    x = torch.randn(2, 3)
    seen = {}

    class FakeModule:
        def activation_forward(self, x_in, activation):
            seen["x"] = x_in
            seen["activation"] = activation
            return x_in + 2

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "activation")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeModule())

    out = runtime_ops_mod.activation(x, "silu")

    assert torch.equal(out, x + 2)
    assert torch.equal(seen["x"], x)
    assert seen["activation"] == "silu"


def test_runtime_gated_activation_dispatches_to_native_binding(monkeypatch):
    x = torch.randn(2, 3)
    gate = torch.randn(2, 3)
    seen = {}

    class FakeModule:
        def gated_activation_forward(self, packed, activation):
            seen["packed"] = packed
            seen["activation"] = activation
            return packed[..., : packed.shape[-1] // 2] + 3

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "gated_activation")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeModule())

    out = runtime_ops_mod.gated_activation(x, gate, "swiglu")

    assert torch.equal(out, x + 3)
    assert torch.equal(seen["packed"], torch.cat([x, gate], dim=-1))
    assert seen["activation"] == "swiglu"


def test_tensor_gelu_exact_prefers_runtime_activation(monkeypatch):
    x = torch.randn(2, 3)
    seen = {}

    def fake_runtime_activation(x_in, activation):
        seen["x"] = x_in
        seen["activation"] = activation
        return x_in + 1

    monkeypatch.setattr(activations_mod, "runtime_activation", fake_runtime_activation)

    out = activations_mod.gelu(x, approx="exact")

    assert torch.equal(out, x + 1)
    assert torch.equal(seen["x"], x)
    assert seen["activation"] == "gelu"


def test_tensor_gelu_tanh_keeps_reference_path(monkeypatch):
    x = torch.randn(2, 3)

    def fail_runtime_activation(x_in, activation):
        del x_in, activation
        raise AssertionError("runtime activation path should not be used for tanh GELU")

    monkeypatch.setattr(activations_mod, "runtime_activation", fail_runtime_activation)

    out = activations_mod.gelu(x, approx="tanh")

    assert torch.allclose(out, F.gelu(x, approximate="tanh"))


def test_tensor_with_bias_act_gated_supported_prefers_runtime_gated_activation(monkeypatch):
    x = torch.randn(2, 3)
    bias = torch.randn(3)
    gate = torch.randn(2, 3)
    seen = {}

    def fake_runtime_gated_activation(x_in, gate_in, activation):
        seen["x"] = x_in
        seen["gate"] = gate_in
        seen["activation"] = activation
        return x_in - gate_in

    monkeypatch.setattr(activations_mod, "runtime_gated_activation", fake_runtime_gated_activation)

    out = activations_mod.with_bias_act(x, bias=bias, act="silu", gate=gate)

    expected_x = x + bias
    assert torch.equal(out, expected_x - gate)
    assert torch.equal(seen["x"], expected_x)
    assert torch.equal(seen["gate"], gate)
    assert seen["activation"] == "silu"


def test_tensor_with_bias_act_quick_gelu_gated_keeps_reference_path(monkeypatch):
    x = torch.randn(2, 3)
    gate = torch.randn(2, 3)

    def fail_runtime_gated_activation(x_in, gate_in, activation):
        del x_in, gate_in, activation
        raise AssertionError("runtime gated activation should not be used for quick GELU")

    monkeypatch.setattr(activations_mod, "runtime_gated_activation", fail_runtime_gated_activation)

    out = activations_mod.with_bias_act(x, act="quick_gelu", gate=gate)

    assert torch.allclose(out, (x * torch.sigmoid(1.702 * x)) * gate)


def test_runtime_linear_uses_eager_reference_when_grad_enabled(monkeypatch):
    x = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)

    class FakeModule:
        def linear_forward(self, x_in, weight_in, bias_in, backend):
            del x_in, weight_in, bias_in, backend
            raise AssertionError("native linear should not run on grad-enabled path")

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "linear")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeModule())

    out = runtime_ops_mod.linear(x, weight, bias)
    ref = F.linear(x, weight, bias)

    assert torch.allclose(out, ref)
    out.sum().backward()
    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None


def test_runtime_activation_uses_eager_reference_when_grad_enabled(monkeypatch):
    x = torch.randn(2, 3, requires_grad=True)

    class FakeModule:
        def activation_forward(self, x_in, activation):
            del x_in, activation
            raise AssertionError("native activation should not run on grad-enabled path")

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "activation")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeModule())

    out = runtime_ops_mod.activation(x, "gelu")
    ref = F.gelu(x)

    assert torch.allclose(out, ref)
    out.sum().backward()
    assert x.grad is not None


def test_runtime_gated_activation_uses_eager_reference_when_grad_enabled(monkeypatch):
    x = torch.randn(2, 3, requires_grad=True)
    gate = torch.randn(2, 3, requires_grad=True)

    class FakeModule:
        def gated_activation_forward(self, packed, activation):
            del packed, activation
            raise AssertionError("native gated activation should not run on grad-enabled path")

    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: name == "gated_activation")
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeModule())

    out = runtime_ops_mod.gated_activation(x, gate, "swiglu")
    ref = F.silu(x) * gate

    assert torch.allclose(out, ref)
    out.sum().backward()
    assert x.grad is not None
    assert gate.grad is not None


def test_runtime_mlp_uses_eager_reference_when_grad_enabled(monkeypatch):
    x = torch.randn(2, 3, 4, requires_grad=True)
    w_in = torch.randn(10, 4, requires_grad=True)
    b_in = torch.randn(10, requires_grad=True)
    w_out = torch.randn(4, 5, requires_grad=True)
    b_out = torch.randn(4, requires_grad=True)

    class FakeModule:
        def mlp_forward(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("native mlp should not run on grad-enabled path")

        def linear_forward(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("native linear should not run on grad-enabled path")

        def activation_forward(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("native activation should not run on grad-enabled path")

        def gated_activation_forward(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("native gated activation should not run on grad-enabled path")

    monkeypatch.setattr(
        runtime_ops_mod,
        "has_native_op",
        lambda name: name in {"mlp", "linear", "activation", "gated_activation"},
    )
    monkeypatch.setattr(runtime_ops_mod, "native_module", lambda: FakeModule())

    out = runtime_ops_mod.mlp(
        x,
        w_in,
        b_in,
        w_out,
        b_out,
        activation="swiglu",
        gated=True,
    )

    hidden = F.linear(x, w_in, b_in)
    a, b = hidden.chunk(2, dim=-1)
    ref = F.linear(F.silu(a) * b, w_out, b_out)

    assert torch.allclose(out, ref)
    out.sum().backward()
    assert x.grad is not None
    assert w_in.grad is not None
    assert w_out.grad is not None
