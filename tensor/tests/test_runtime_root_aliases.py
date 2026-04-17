from __future__ import annotations

import masking as masking_mod
import model.lm as model_lm_mod
import positional as positional_mod
import runtime.causal as runtime_causal_mod
import runtime.positional as runtime_positional_mod
import runtime.sampling as runtime_sampling_mod
import sampling as sampling_mod
import tensor.masking as tensor_masking_mod
import tensor.positional as tensor_positional_mod
import tensor.sampling as tensor_sampling_mod


def test_root_compat_modules_are_runtime_or_tensor_aliases():
    assert positional_mod is runtime_positional_mod
    assert sampling_mod is runtime_sampling_mod
    assert masking_mod is tensor_masking_mod
    assert model_lm_mod is runtime_causal_mod


def test_tensor_compat_modules_are_runtime_aliases():
    assert tensor_positional_mod is runtime_positional_mod
    assert tensor_sampling_mod is runtime_sampling_mod
