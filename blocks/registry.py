from typing import Callable, Dict


_BLOCK_REGISTRY: Dict[str, Callable[..., object]] = {}


def register_block(name: str, builder: Callable[..., object]) -> None:
    key = name.lower()
    if key in _BLOCK_REGISTRY:
        raise ValueError(f"Block already registered: {name}")
    _BLOCK_REGISTRY[key] = builder


def get_block_builder(name: str) -> Callable[..., object]:
    key = name.lower()
    if key not in _BLOCK_REGISTRY:
        raise KeyError(f"Unknown block: {name}")
    return _BLOCK_REGISTRY[key]


def list_blocks() -> list[str]:
    return sorted(_BLOCK_REGISTRY.keys())


