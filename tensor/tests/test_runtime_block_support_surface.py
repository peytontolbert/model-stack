from __future__ import annotations

import blocks.adapters as blocks_adapters_mod
import blocks.stack as blocks_stack_mod
import blocks.utils as blocks_utils_mod
import runtime as runtime_pkg
import runtime.block_adapters as runtime_block_adapters_mod
import runtime.block_stack as runtime_block_stack_mod
import runtime.block_utils as runtime_block_utils_mod


def test_blocks_support_modules_are_runtime_surfaces():
    assert blocks_stack_mod.TransformerStack is runtime_block_stack_mod.TransformerStack
    assert blocks_stack_mod.EncoderDecoderStack is runtime_block_stack_mod.EncoderDecoderStack
    assert blocks_adapters_mod.BottleneckAdapter is runtime_block_adapters_mod.BottleneckAdapter
    assert blocks_adapters_mod.IA3Adapter is runtime_block_adapters_mod.IA3Adapter
    assert blocks_adapters_mod.attach_adapters_to_block is runtime_block_adapters_mod.attach_adapters_to_block
    assert blocks_utils_mod.getattr_nested is runtime_block_utils_mod.getattr_nested


def test_runtime_package_exports_block_support_surface():
    assert runtime_pkg.TransformerStack is runtime_block_stack_mod.TransformerStack
    assert runtime_pkg.EncoderDecoderStack is runtime_block_stack_mod.EncoderDecoderStack
    assert runtime_pkg.BottleneckAdapter is runtime_block_adapters_mod.BottleneckAdapter
    assert runtime_pkg.IA3Adapter is runtime_block_adapters_mod.IA3Adapter
    assert runtime_pkg.attach_adapters_to_block is runtime_block_adapters_mod.attach_adapters_to_block
    assert runtime_pkg.getattr_nested is runtime_block_utils_mod.getattr_nested
