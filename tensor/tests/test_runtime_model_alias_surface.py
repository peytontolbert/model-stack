from __future__ import annotations

import model.checkpoint as model_checkpoint_mod
import model.compile as model_compile_mod
import model.export as model_export_mod
import model.factory as model_factory_mod
import model.generate as model_generate_mod
import model.hf_llama_loader as model_hf_llama_loader_mod
import model.hf_snapshot as model_hf_snapshot_mod
import model.inspect as model_inspect_mod
import model.llama_bootstrap as model_llama_bootstrap_mod
import model.registry as model_registry_mod
import model.runtime_utils as model_runtime_utils_mod
import model.utils as model_utils_mod
import runtime.checkpoint as runtime_checkpoint_mod
import runtime.compile as runtime_compile_mod
import runtime.factory as runtime_factory_mod
import runtime.inspect as runtime_inspect_mod
import runtime.model_export as runtime_model_export_mod
import runtime.model_generate as runtime_model_generate_mod
import runtime.model_registry as runtime_model_registry_mod
import runtime.runtime_utils as runtime_runtime_utils_mod


def test_model_modules_are_runtime_aliases():
    assert model_factory_mod is runtime_factory_mod
    assert model_registry_mod is runtime_model_registry_mod
    assert model_checkpoint_mod is runtime_checkpoint_mod
    assert model_hf_snapshot_mod is runtime_checkpoint_mod
    assert model_hf_llama_loader_mod is runtime_checkpoint_mod
    assert model_llama_bootstrap_mod is runtime_checkpoint_mod
    assert model_inspect_mod is runtime_inspect_mod
    assert model_runtime_utils_mod is runtime_runtime_utils_mod
    assert model_utils_mod is runtime_runtime_utils_mod
    assert model_compile_mod is runtime_compile_mod
    assert model_generate_mod is runtime_model_generate_mod
    assert model_export_mod is runtime_model_export_mod
