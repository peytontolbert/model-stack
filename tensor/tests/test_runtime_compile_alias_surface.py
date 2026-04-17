from __future__ import annotations

import model as model_pkg
import model.compile as model_compile_mod
import model.lm as model_lm_mod
import runtime as runtime_pkg
import runtime.compile as runtime_compile_mod


def test_model_compile_is_runtime_shim():
    assert model_compile_mod.maybe_compile is runtime_compile_mod.maybe_compile


def test_runtime_and_model_packages_export_compile_surface():
    assert runtime_pkg.maybe_compile is runtime_compile_mod.maybe_compile
    assert model_pkg.maybe_compile is runtime_compile_mod.maybe_compile


def test_transformer_lm_aliases_causal_lm():
    assert model_lm_mod.TransformerLM is model_lm_mod.CausalLM
    assert model_pkg.TransformerLM is model_pkg.CausalLM
