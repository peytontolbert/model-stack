from __future__ import annotations

import runtime as runtime_pkg
import runtime.generation as runtime_generation_mod
import runtime.model_registry as runtime_model_registry_mod
import runtime.model_surface as runtime_model_surface_mod
import runtime.native as runtime_native_mod
import serve.runtime as serve_runtime_mod


def test_runtime_package_exports_cuda_inference_surface():
    assert runtime_pkg.build is runtime_model_registry_mod.build
    assert runtime_pkg.ModelRuntime is serve_runtime_mod.ModelRuntime
    assert runtime_pkg.RuntimeConfig is serve_runtime_mod.RuntimeConfig
    assert runtime_pkg.resolve_generation_sampling_mode is runtime_generation_mod.resolve_generation_sampling_mode
    assert runtime_pkg.resolve_linear_backend is runtime_native_mod.resolve_linear_backend
    assert runtime_pkg.native_module is runtime_native_mod.native_module
    assert runtime_pkg.native_paged_kv_cache_available is runtime_native_mod.native_paged_kv_cache_available
    assert runtime_pkg.create_native_paged_kv_cache_state is runtime_native_mod.create_native_paged_kv_cache_state
    assert runtime_pkg.cuda_kernel_ops is runtime_native_mod.cuda_kernel_ops
    assert runtime_pkg.cuda_inference_ops is runtime_native_mod.cuda_inference_ops
    assert runtime_pkg.cuda_composite_ops is runtime_native_mod.cuda_composite_ops
    assert runtime_pkg.full_cuda_inference_available is runtime_native_mod.full_cuda_inference_available


def test_runtime_model_surface_exports_runtime_inference_helpers():
    assert runtime_model_surface_mod.build is runtime_model_registry_mod.build
    assert runtime_model_surface_mod.ModelRuntime is serve_runtime_mod.ModelRuntime
    assert runtime_model_surface_mod.RuntimeConfig is serve_runtime_mod.RuntimeConfig


def test_runtime_info_normalizes_cuda_inference_metadata(monkeypatch):
    monkeypatch.setattr(
        runtime_native_mod,
        "runtime_status",
        lambda: runtime_native_mod.NativeRuntimeStatus(
            available=True,
            module_name="_model_stack_native",
            info={
                "compiled_with_cuda": True,
                "native_ops": ["linear", "mlp", "sampling"],
                "cuda_backend_ops": ["linear", "sampling"],
            },
            error=None,
        ),
    )

    info = runtime_native_mod.runtime_info()

    assert info["cuda_backend_ops"] == ["linear", "sampling"]
    assert info["cuda_kernel_ops"] == ["linear", "sampling"]
    assert info["cuda_inference_ops"] == ["linear", "mlp", "sampling"]
    assert info["cuda_composite_ops"] == ["mlp"]
    assert info["full_cuda_inference"] is True
    assert runtime_native_mod.cuda_kernel_ops() == ["linear", "sampling"]
    assert runtime_native_mod.cuda_inference_ops() == ["linear", "mlp", "sampling"]
    assert runtime_native_mod.cuda_composite_ops() == ["mlp"]
    assert runtime_native_mod.full_cuda_inference_available() is True


def test_runtime_info_without_cuda_reports_no_cuda_inference(monkeypatch):
    monkeypatch.setattr(
        runtime_native_mod,
        "runtime_status",
        lambda: runtime_native_mod.NativeRuntimeStatus(
            available=True,
            module_name="_model_stack_native",
            info={
                "compiled_with_cuda": False,
                "native_ops": ["linear", "sampling"],
                "cuda_backend_ops": ["linear"],
            },
            error=None,
        ),
    )

    info = runtime_native_mod.runtime_info()

    assert info["cuda_inference_ops"] == []
    assert info["cuda_composite_ops"] == []
    assert info["full_cuda_inference"] is False
    assert runtime_native_mod.full_cuda_inference_available() is False
