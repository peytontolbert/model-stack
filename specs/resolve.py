from typing import Callable, Dict

from .config import ModelConfig
from .ops import get_op, has_op, list_ops


def resolve_from_config(cfg: ModelConfig) -> Dict[str, Callable]:
    """Return callables for ops referenced by name in ModelConfig.

    Raises:
        KeyError: if any requested op name is not registered.
    """
    _assert_available("activations", cfg.activation)
    _assert_available("norms", cfg.norm)
    _assert_available("positional", cfg.positional)
    _assert_available("masking", cfg.masking)
    _assert_available("numerics", cfg.softmax)
    _assert_available("residual", cfg.residual)

    return {
        "activation": get_op("activations", cfg.activation),
        "norm": get_op("norms", cfg.norm),
        "positional": get_op("positional", cfg.positional),
        "masking": get_op("masking", cfg.masking),
        "softmax": get_op("numerics", cfg.softmax),
        "residual": get_op("residual", cfg.residual),
    }


def _assert_available(category: str, name: str) -> None:
    if not has_op(category, name):
        available = ", ".join(list_ops(category)[category])
        raise KeyError(f"Unknown op '{name}' for category '{category}'. Available: {available}")


