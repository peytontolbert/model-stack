"""Compatibility shim for masking helpers that remain tensor-owned."""

from tensor import masking as _masking

__all__ = [name for name in dir(_masking) if not name.startswith("_")]

globals().update({name: getattr(_masking, name) for name in __all__})
