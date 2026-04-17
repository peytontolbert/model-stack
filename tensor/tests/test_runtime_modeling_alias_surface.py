from __future__ import annotations

import runtime.model_surface as runtime_model_surface_mod
import runtime.modeling as runtime_modeling_mod


def test_runtime_modeling_is_runtime_model_surface_alias():
    assert runtime_modeling_mod is runtime_model_surface_mod
