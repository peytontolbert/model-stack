from __future__ import annotations

import os

from setuptools import setup


def env_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def native_extensions():
    if not env_enabled("MODEL_STACK_BUILD_NATIVE", "0"):
        return []

    from pybind11.setup_helpers import Pybind11Extension

    return [
        Pybind11Extension(
            "_model_stack_native",
            ["runtime/csrc/model_stack_native.cpp"],
            cxx_std=17,
            define_macros=[
                ("MODEL_STACK_ABI_VERSION", "1"),
                ("MODEL_STACK_WITH_CUDA", "0"),
            ],
            extra_compile_args=["-O3"],
        )
    ]


setup(ext_modules=native_extensions())
