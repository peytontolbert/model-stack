from importlib import import_module

from runtime.block_config import BlockConfig, build_block_config_from_model
from runtime.block_init import apply_block_init, init_transformer_stack
from runtime.block_modules import GPTBlock, LlamaBlock, TransformerBlock
from runtime.block_schedules import drop_path_linear

_LAZY_EXPORTS = {
    "build_block": "runtime.block_factory",
    "build_block_stack": "runtime.block_factory",
    "register_block": "runtime.block_registry",
    "get_block_builder": "runtime.block_registry",
    "list_blocks": "runtime.block_registry",
}

__all__ = [
    "BlockConfig",
    "build_block_config_from_model",
    "TransformerBlock",
    "LlamaBlock",
    "GPTBlock",
    "apply_block_init",
    "init_transformer_stack",
    "build_block",
    "build_block_stack",
    "register_block",
    "get_block_builder",
    "list_blocks",
    "drop_path_linear",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'blocks' has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
