from __future__ import annotations

import model as model_pkg
import model.utils as model_utils_mod
import runtime as runtime_pkg
import runtime.runtime_utils as runtime_runtime_utils_mod


def test_model_utils_is_runtime_shim():
    assert model_utils_mod.num_parameters is runtime_runtime_utils_mod.num_parameters
    assert model_utils_mod.num_trainable_parameters is runtime_runtime_utils_mod.num_trainable_parameters
    assert model_utils_mod.to_dtype is runtime_runtime_utils_mod.to_dtype
    assert model_utils_mod.to_device is runtime_runtime_utils_mod.to_device


def test_runtime_and_model_packages_export_model_utils_surface():
    assert runtime_pkg.num_parameters is runtime_runtime_utils_mod.num_parameters
    assert runtime_pkg.num_trainable_parameters is runtime_runtime_utils_mod.num_trainable_parameters
    assert runtime_pkg.to_dtype is runtime_runtime_utils_mod.to_dtype
    assert runtime_pkg.to_device is runtime_runtime_utils_mod.to_device

    assert model_pkg.num_parameters is runtime_runtime_utils_mod.num_parameters
    assert model_pkg.num_trainable_parameters is runtime_runtime_utils_mod.num_trainable_parameters
    assert model_pkg.to_dtype is runtime_runtime_utils_mod.to_dtype
    assert model_pkg.to_device is runtime_runtime_utils_mod.to_device
