from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compress.quantization as quant_mod
import runtime.generation as runtime_generation_mod
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


def _disable_python_runtime_ops(monkeypatch) -> None:
    def boom(*args, **kwargs):
        del args, kwargs
        raise AssertionError("direct native BitNet executor should not call Python runtime.ops helpers")

    monkeypatch.setattr(runtime_ops_mod, "qkv_packed_spec_heads_projection", boom)
    monkeypatch.setattr(runtime_ops_mod, "resolve_packed_qkv_module_spec", boom)
    monkeypatch.setattr(runtime_ops_mod, "head_output_packed_projection", boom)
    monkeypatch.setattr(runtime_ops_mod, "resolve_packed_linear_module_spec", boom)
    monkeypatch.setattr(runtime_ops_mod, "mlp_module", boom)
    monkeypatch.setattr(runtime_ops_mod, "linear_module", boom)


def _run_native_bitnet_session_without_python_runtime_ops(
    monkeypatch,
    model: CausalLM,
) -> torch.Tensor:
    _disable_python_runtime_ops(monkeypatch)

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


def _run_native_bitnet_session(model: CausalLM) -> torch.Tensor:
    seq = torch.randint(0, model.cfg.vocab_size, (2, 3))
    session = runtime_native_mod.create_native_model_session(model, seq)
    if session is None or getattr(session, "native_executor_kind", "python") != "causal_lm":
        pytest.skip("native causal executor unavailable")
    with torch.no_grad():
        return session.full_next_logits()


def _create_session(model: CausalLM) -> tuple[object | None, torch.Tensor]:
    seq = torch.randint(0, model.cfg.vocab_size, (2, 3))
    return runtime_native_mod.create_native_model_session(model, seq), seq


def _create_session_with_seq(model: CausalLM, seq: torch.Tensor) -> object | None:
    return runtime_native_mod.create_native_model_session(model, seq)


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_native_causal_executor_uses_module_aware_runtime_paths_for_bitnet(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False)
    model.eval()
    _convert_model_to_bitnet(model)
    logits = _run_native_bitnet_session_without_python_runtime_ops(monkeypatch, model)
    assert logits.shape == (2, model.cfg.vocab_size)


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_native_causal_executor_direct_bitnet_uses_compute_backend_cache(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False)
    model.eval()
    modules = _convert_model_to_bitnet(model)
    counters: list[dict[str, int]] = []
    for module in modules:
        seen = {"calls": 0}
        original = module._compute_backend_weight

        def wrapped(*, device=None, _original=original, _seen=seen):
            _seen["calls"] += 1
            return _original(device=device)

        monkeypatch.setattr(module, "_compute_backend_weight", wrapped)
        counters.append(seen)

    logits = _run_native_bitnet_session_without_python_runtime_ops(monkeypatch, model)

    assert logits.shape == (2, model.cfg.vocab_size)
    assert all(entry["calls"] > 0 for entry in counters)


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_native_causal_executor_direct_bitnet_uses_decode_backend_cache(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False)
    model.eval()
    modules = _convert_model_to_bitnet(model)
    counters: list[dict[str, int]] = []
    for module in modules:
        seen = {"calls": 0}
        original = module._decode_backend_weight

        def wrapped(*, device=None, _original=original, _seen=seen):
            _seen["calls"] += 1
            return _original(device=device)

        monkeypatch.setattr(module, "_decode_backend_weight", wrapped)
        counters.append(seen)

    logits = _run_native_bitnet_session_without_python_runtime_ops(monkeypatch, model)

    assert logits.shape == (2, model.cfg.vocab_size)
    assert all(entry["calls"] > 0 for entry in counters)


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
def test_native_model_session_runs_bitnet_w2a8_calibrated_models_on_native_executor(monkeypatch):
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
def test_native_model_session_runs_bitnet_w2a8_sub8_models_on_native_executor(monkeypatch):
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


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_native_causal_executor_direct_bitnet_ignores_irrelevant_act_metadata_for_fused_qkv(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False)
    model.eval()
    modules = _convert_model_to_bitnet(model)
    block = model.blocks[0]

    block.attn.w_q.act_quant_mode = "none"
    block.attn.w_q.act_quant_method = "absmax"
    block.attn.w_q.act_quant_percentile = 0.999

    block.attn.w_k.act_quant_mode = "none"
    block.attn.w_k.act_quant_method = "mse"
    block.attn.w_k.act_quant_percentile = 0.75

    block.attn.w_v.act_quant_mode = "none"
    block.attn.w_v.act_quant_method = "percentile"
    block.attn.w_v.act_quant_percentile = 0.5

    for module in modules:
        module.act_quant_bits = 8

    logits = _run_native_bitnet_session_without_python_runtime_ops(monkeypatch, model)
    assert logits.shape == (2, model.cfg.vocab_size)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not runtime_native_mod.native_model_session_available(),
    reason="CUDA native causal executor unavailable",
)
def test_native_causal_executor_direct_bitnet_noquant_prefill_uses_compute_backend_family(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False).cuda()
    model.eval()
    modules = _convert_model_to_bitnet(model)
    for module in modules:
        module.act_quant_mode = "none"
        module._compute_backend_weight(device="cuda")

    _disable_python_runtime_ops(monkeypatch)
    seq = torch.randint(0, model.cfg.vocab_size, (2, 3), device="cuda")

    baseline_session = _create_session_with_seq(model, seq)
    if baseline_session is None or getattr(baseline_session, "native_executor_kind", "python") != "causal_lm":
        pytest.skip("native causal executor unavailable")

    with torch.no_grad():
        baseline = baseline_session.full_next_logits()

    for module in modules:
        module.packed_weight.zero_()

    corrupted_session = _create_session_with_seq(model, seq)
    if corrupted_session is None or getattr(corrupted_session, "native_executor_kind", "python") != "causal_lm":
        pytest.skip("native causal executor unavailable")

    with torch.no_grad():
        actual = corrupted_session.full_next_logits()

    assert torch.allclose(actual, baseline, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_runtime_generation_session_native_executor_supports_bitnet_w2a8_cached_decode(monkeypatch):
    model = CausalLM(_cfg(), block_variant="llama", tie_weights=False)
    model.eval()
    for idx, module in enumerate(_convert_model_to_bitnet(model)):
        module.act_quant_mode = "dynamic_int8" if idx % 2 == 0 else "static_int8"
        module.act_quant_method = "percentile" if idx % 2 == 0 else "mse"
        module.act_quant_bits = 4 if idx % 2 == 0 else 6
        module.act_quant_percentile = 0.92
        module.act_scale.fill_(0.15)

    session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        cache_pagesize=16,
        cache_backend="native-paged",
    )
    if not session.uses_native_session or session.native_executor_kind != "causal_lm":
        pytest.skip("native causal executor unavailable")

    _disable_python_runtime_ops(monkeypatch)

    def boom(*args, **kwargs):
        del args, kwargs
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    prefill = session.prefill_next_logits()
    assert prefill is not None
    assert prefill.shape == (1, model.cfg.vocab_size)

    session.append(torch.argmax(prefill, dim=-1, keepdim=True))
    decode = session.decode_next_logits()
    assert decode is not None
    assert decode.shape == (1, model.cfg.vocab_size)
    rope_cos = getattr(model.blocks[0].attn, "_rope_cos", None)
    rope_sin = getattr(model.blocks[0].attn, "_rope_sin", None)
    assert isinstance(rope_cos, torch.Tensor)
    assert isinstance(rope_sin, torch.Tensor)


@pytest.mark.skipif(not runtime_native_mod.native_model_session_available(), reason="native causal executor unavailable")
def test_runtime_generation_session_native_bitnet_w2a8_cached_decode_with_alibi(monkeypatch):
    model = CausalLM(_cfg(), block_variant="gpt", tie_weights=False, use_alibi=True)
    model.eval()
    for idx, module in enumerate(_convert_model_to_bitnet(model)):
        module.act_quant_mode = "dynamic_int8" if idx % 2 == 0 else "static_int8"
        module.act_quant_method = "percentile" if idx % 2 == 0 else "mse"
        module.act_quant_bits = 4 if idx % 2 == 0 else 6
        module.act_quant_percentile = 0.92
        module.act_scale.fill_(0.15)

    seq = torch.tensor([[1, 2]], dtype=torch.long)
    native_session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        seq,
        cache_pagesize=16,
        cache_backend="native-paged",
    )
    if not native_session.uses_native_session or native_session.native_executor_kind != "causal_lm":
        pytest.skip("native causal executor unavailable")

    _disable_python_runtime_ops(monkeypatch)

    def boom(*args, **kwargs):
        del args, kwargs
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    native_prefill = native_session.prefill_next_logits()
    assert native_prefill is not None

    next_id = torch.argmax(native_prefill, dim=-1, keepdim=True)
    native_session.append(next_id)

    native_decode = native_session.decode_next_logits()
    assert native_decode is not None
    assert native_decode.shape == (1, model.cfg.vocab_size)
