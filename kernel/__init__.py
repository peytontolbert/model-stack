from .registry import KernelRegistry, register, register_lazy, get, has, available

# Ensure standard kernels are registered (lazy where appropriate)
from . import flash as _flash  # noqa: F401
from . import rope as _rope  # noqa: F401
from . import triton as _triton  # noqa: F401

__all__ = [
    "KernelRegistry",
    "register",
    "register_lazy",
    "get",
    "has",
    "available",
]


