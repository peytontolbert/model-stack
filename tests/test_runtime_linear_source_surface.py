from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_linear_auto_backend_exposes_sm8x_aten_selector() -> None:
    source = _read("runtime/csrc/model_stack_native.cpp")

    assert "PreferAtenLinearBackendForAuto" in source
    assert "ResolveLinearBackendForTensor" in source
    assert "MODEL_STACK_DISABLE_SM8X_ATEN_LINEAR_AUTO" in source
    assert 'return "aten";' in source


def test_linear_forward_uses_tensor_aware_backend_resolution() -> None:
    source = _read("runtime/csrc/model_stack_native.cpp")

    assert 'const auto resolved_backend = ResolveLinearBackendForTensor(backend, x);' in source
    assert 'if (resolved_backend == "cublaslt" && x.is_cuda())' in source


def test_hopper_bitnet_dense_decode_cache_defaults_to_full() -> None:
    source = _read("runtime/csrc/model_stack_native.cpp")

    assert "HopperDenseBitNetDecodeCacheMode" in source
    assert 'if (env_value == nullptr) {\n    return "full";\n  }' in source
    assert 'if (normalized.empty() || normalized == "qkv" || normalized == "qkv_only")' in source
