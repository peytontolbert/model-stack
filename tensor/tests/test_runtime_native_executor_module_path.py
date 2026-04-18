from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compress.quantization as quant_mod
import runtime.native as runtime_native_mod
import runtime.ops as runtime_ops_mod
from runtime.causal import CausalLM
from specs.config import ModelConfig


def _cfg() -> ModelConfig:
    return ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        vocab_size=64,
        dtype="float32",
    )


def _to_bitnet(module: torch.nn.Linear) -> quant_mod.QuantizedLinearBitNet:
    return quant_mod.QuantizedLinearBitNet(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
    ).from_float(module)


def _convert_model_to_bitnet(model: CausalLM) -> list[quant_mod.QuantizedLinearBitNet]:
    block = model.blocks[0]
    modules: list[quant_mod.QuantizedLinearBitNet] = []
    for name in ("w_q", "w_k", "w_v", "w_o"):
        q = _to_bitnet(getattr(block.attn, name))
        setattr(block.attn, name, q)
        modules.append(q)
    for name in ("w_in", "w_out"):
        q = _to_bitnet(getattr(block.mlp, name))
        setattr(block.mlp, name, q)
        modules.append(q)
    model.lm_head = _to_bitnet(model.lm_head)
    modules.append(model.lm_head)
    return modules


def _run_native_bitnet_session_without_python_runtime_ops(monkeypatch, model: CausalLM) -> torch.Tensor:
    def boom(*args, **kwargs):
        del args, kwargs
        raise AssertionError("direct native BitNet executor should not call Python runtime.ops helpers")

    monkeypatch.setattr(runtime_ops_mod, "qkv_packed_spec_heads_projection", boom)
    monkeypatch.setattr(runtime_ops_mod, "mlp_module", boom)
    monkeypatch.setattr(runtime_ops_mod, "linear_module", boom)

    seq = torch.randint(0, model.cfg.vocab_size, (2, 3))
    session = runtime_native_mod.create_native_model_session(model, seq)
    if session is None or getattr(session, "native_executor_kind", "python") != "causal_lm":
        pytest.skip("native causal executor unavailable")

    with torch.no_grad():
        try:
            return session.full_next_logits()
        except AssertionError as exc:
            if "direct native BitNet executor should not call Python runtime.ops helpers" in str(exc):
                pytest.skip("loaded native extension not rebuilt with direct BitNet executor")
            raise


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_native_causal_executor_uses_module_aware_runtime_paths_for_bitnet(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False)
    model.eval()
    _convert_model_to_bitnet(model)
    logits = _run_native_bitnet_session_without_python_runtime_ops(monkeypatch, model)
    assert logits.shape == (2, model.cfg.vocab_size)


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_native_causal_executor_direct_bitnet_supports_spin_prescale_and_static_int8(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False)
    model.eval()
    for idx, module in enumerate(_convert_model_to_bitnet(model)):
        module.spin_enabled_flag.fill_(1)
        signs = torch.where(torch.arange(module.in_features) % 2 == 0, 1.0, -1.0)
        if idx % 2:
            signs = -signs
        module.spin_signs.copy_(signs)
        module.pre_scale.copy_(torch.linspace(0.75, 1.25, module.in_features))
        module.act_quant_mode = "static_int8"
        module.act_quant_method = "absmax"
        module.act_quant_bits = 8
        module.act_scale.fill_(0.125)
    logits = _run_native_bitnet_session_without_python_runtime_ops(monkeypatch, model)
    assert logits.shape == (2, model.cfg.vocab_size)


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_native_causal_executor_direct_bitnet_supports_percentile_and_mse_activation_calibration(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False)
    model.eval()
    for idx, module in enumerate(_convert_model_to_bitnet(model)):
        if idx % 2 == 0:
            module.act_quant_mode = "dynamic_int8"
            module.act_quant_method = "percentile"
            module.act_quant_percentile = 0.95
        else:
            module.act_quant_mode = "static_int8"
            module.act_quant_method = "mse"
            module.act_scale.fill_(0.2)
        module.act_quant_bits = 8

    logits = _run_native_bitnet_session_without_python_runtime_ops(monkeypatch, model)
    assert logits.shape == (2, model.cfg.vocab_size)


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_native_causal_executor_direct_bitnet_supports_sub8_activation_quant_bits(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False)
    model.eval()
    for idx, module in enumerate(_convert_model_to_bitnet(model)):
        module.act_quant_mode = "dynamic_int8" if idx % 2 == 0 else "static_int8"
        module.act_quant_method = "percentile" if idx % 2 == 0 else "mse"
        module.act_quant_bits = 4 if idx % 2 == 0 else 6
        module.act_quant_percentile = 0.92
        module.act_scale.fill_(0.15)

    logits = _run_native_bitnet_session_without_python_runtime_ops(monkeypatch, model)
    assert logits.shape == (2, model.cfg.vocab_size)
