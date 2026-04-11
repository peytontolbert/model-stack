from __future__ import annotations

import os

from setuptools import setup


def env_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def native_extensions():
    if not env_enabled("MODEL_STACK_BUILD_NATIVE", "0"):
        return []

    from torch.utils.cpp_extension import BuildExtension, CppExtension

    ext = CppExtension(
        "_model_stack_native",
        ["runtime/csrc/model_stack_native.cpp"],
        define_macros=[
            ("MODEL_STACK_ABI_VERSION", "1"),
            ("MODEL_STACK_WITH_CUDA", "0"),
        ],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    )
    return [ext], {"build_ext": BuildExtension}


extensions, cmdclass = native_extensions() if env_enabled("MODEL_STACK_BUILD_NATIVE", "0") else ([], {})
setup(ext_modules=extensions, cmdclass=cmdclass)
