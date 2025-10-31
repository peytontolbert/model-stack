from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class KernelRegistry:
    """Lightweight global registry for optional kernel implementations.

    Names are stored in lowercase and can be namespaced via dots, e.g.,
    "attn.flash2" or "mlp.bias_gelu". Implementations can be registered
    eagerly or lazily via a loader callable.
    """

    _impls: Dict[str, Callable[..., Any]] = {}
    _loaders: Dict[str, Callable[[], Callable[..., Any]]] = {}

    @classmethod
    def _key(cls, name: str) -> str:
        return str(name).strip().lower()

    @classmethod
    def register(cls, name: str, fn: Callable[..., Any]) -> None:
        cls._impls[cls._key(name)] = fn
        # If previously had a loader, drop it now that we have a concrete impl
        cls._loaders.pop(cls._key(name), None)

    @classmethod
    def register_lazy(cls, name: str, loader: Callable[[], Callable[..., Any]]) -> None:
        """Register a loader that returns the callable upon first access."""
        key = cls._key(name)
        if key in cls._impls:
            return
        cls._loaders[key] = loader

    @classmethod
    def has(cls, name: str) -> bool:
        key = cls._key(name)
        if key in cls._impls:
            return True
        if key in cls._loaders:
            try:
                impl = cls._loaders[key]()
                cls._impls[key] = impl
                cls._loaders.pop(key, None)
                return True
            except Exception:
                return False
        return False

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        key = cls._key(name)
        if key in cls._impls:
            return cls._impls[key]
        if key in cls._loaders:
            impl = cls._loaders[key]()
            cls._impls[key] = impl
            cls._loaders.pop(key, None)
            return impl
        raise KeyError(f"Kernel '{name}' is not registered")

    @classmethod
    def try_get(cls, name: str) -> Optional[Callable[..., Any]]:
        try:
            return cls.get(name)
        except Exception:
            return None

    @classmethod
    def available(cls) -> Dict[str, bool]:
        keys = set(cls._impls.keys()) | set(cls._loaders.keys())
        return {k: k in cls._impls for k in sorted(keys)}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a kernel implementation."""
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        KernelRegistry.register(name, fn)
        return fn
    return _decorator


def register_lazy(name: str, loader: Callable[[], Callable[..., Any]]) -> None:
    KernelRegistry.register_lazy(name, loader)


def get(name: str) -> Callable[..., Any]:
    return KernelRegistry.get(name)


def has(name: str) -> bool:
    return KernelRegistry.has(name)


def available() -> Dict[str, bool]:
    return KernelRegistry.available()
