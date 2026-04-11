from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


NATIVE_MODULE_NAME = "_model_stack_native"
DISABLE_ENV = "MODEL_STACK_DISABLE_NATIVE"


@dataclass(frozen=True)
class NativeRuntimeStatus:
    available: bool
    module_name: str
    info: dict[str, Any]
    error: str | None = None


@lru_cache(maxsize=1)
def runtime_status() -> NativeRuntimeStatus:
    if os.getenv(DISABLE_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}:
        return NativeRuntimeStatus(
            available=False,
            module_name=NATIVE_MODULE_NAME,
            info={"disabled_by_env": True},
            error=f"{DISABLE_ENV} is set",
        )

    try:
        module = importlib.import_module(NATIVE_MODULE_NAME)
    except Exception as exc:
        return NativeRuntimeStatus(
            available=False,
            module_name=NATIVE_MODULE_NAME,
            info={"disabled_by_env": False},
            error=str(exc),
        )

    try:
        info = dict(module.runtime_info())
    except Exception as exc:
        return NativeRuntimeStatus(
            available=False,
            module_name=NATIVE_MODULE_NAME,
            info={},
            error=f"native module loaded but runtime_info failed: {exc}",
        )

    return NativeRuntimeStatus(
        available=True,
        module_name=NATIVE_MODULE_NAME,
        info=info,
        error=None,
    )


def native_available() -> bool:
    return runtime_status().available


def runtime_info() -> dict[str, Any]:
    return dict(runtime_status().info)


def native_module():
    if not native_available():
        return None
    return importlib.import_module(NATIVE_MODULE_NAME)


def has_native_op(name: str) -> bool:
    module = native_module()
    if module is None or not hasattr(module, "has_op"):
        return False
    try:
        return bool(module.has_op(name))
    except Exception:
        return False
