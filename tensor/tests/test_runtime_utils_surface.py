from __future__ import annotations

import model as model_pkg
import model.runtime_utils as model_runtime_utils_mod
import runtime as runtime_pkg
import runtime.runtime_utils as runtime_runtime_utils_mod


def test_model_runtime_utils_is_runtime_shim():
    assert model_runtime_utils_mod.resolve_layer_modules is runtime_runtime_utils_mod.resolve_layer_modules
    assert model_runtime_utils_mod.find_module_by_relpath is runtime_runtime_utils_mod.find_module_by_relpath
    assert model_runtime_utils_mod.apply_weight_deltas is runtime_runtime_utils_mod.apply_weight_deltas
    assert model_runtime_utils_mod.prepare_head_weight is runtime_runtime_utils_mod.prepare_head_weight
    assert model_runtime_utils_mod.local_logits_last is runtime_runtime_utils_mod.local_logits_last


def test_runtime_and_model_packages_export_runtime_utils_surface():
    assert runtime_pkg.resolve_layer_modules is runtime_runtime_utils_mod.resolve_layer_modules
    assert runtime_pkg.find_module_by_relpath is runtime_runtime_utils_mod.find_module_by_relpath
    assert runtime_pkg.apply_weight_deltas is runtime_runtime_utils_mod.apply_weight_deltas
    assert runtime_pkg.prepare_head_weight is runtime_runtime_utils_mod.prepare_head_weight
    assert runtime_pkg.local_logits_last is runtime_runtime_utils_mod.local_logits_last

    assert model_pkg.resolve_layer_modules is runtime_runtime_utils_mod.resolve_layer_modules
    assert model_pkg.find_module_by_relpath is runtime_runtime_utils_mod.find_module_by_relpath
    assert model_pkg.apply_weight_deltas is runtime_runtime_utils_mod.apply_weight_deltas
    assert model_pkg.prepare_head_weight is runtime_runtime_utils_mod.prepare_head_weight
    assert model_pkg.local_logits_last is runtime_runtime_utils_mod.local_logits_last
