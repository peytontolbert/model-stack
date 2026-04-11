from __future__ import annotations

import os

from setuptools import setup


def env_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def native_extensions():
    if not env_enabled("MODEL_STACK_BUILD_NATIVE", "0"):
        return []

    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

    use_cuda = CUDA_HOME is not None and env_enabled("MODEL_STACK_BUILD_CUDA", "1")
    sources = ["runtime/csrc/model_stack_native.cpp"]
    define_macros = [("MODEL_STACK_ABI_VERSION", "1")]
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}

    extension_cls = CppExtension
    if use_cuda:
        extension_cls = CUDAExtension
        sources.append("runtime/csrc/backend/cuda_rms_norm.cu")
        sources.append("runtime/csrc/backend/cublaslt_linear.cu")
        define_macros.append(("MODEL_STACK_WITH_CUDA", "1"))
        extra_compile_args["nvcc"] = [
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
        ]
    else:
        define_macros.append(("MODEL_STACK_WITH_CUDA", "0"))

    ext = extension_cls(
        "_model_stack_native",
        sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
    return [ext], {"build_ext": BuildExtension}


extensions, cmdclass = native_extensions() if env_enabled("MODEL_STACK_BUILD_NATIVE", "0") else ([], {})
setup(ext_modules=extensions, cmdclass=cmdclass)
