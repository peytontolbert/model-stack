from __future__ import annotations

import attn.backends as attn_backends_mod
import attn.decoding as attn_decoding_mod
import attn.gqa as attn_gqa_mod
import attn.interfaces as attn_interfaces_mod
import attn.kv_cache as attn_kv_cache_mod
import attn.moe as attn_moe_mod
import attn.optim_utils as attn_optim_utils_mod
import attn.quant as attn_quant_mod
import attn.reference as attn_reference_mod
import blocks.config as blocks_config_mod
import blocks.factory as blocks_factory_mod
import blocks.init as blocks_init_mod
import blocks.inspect as blocks_inspect_mod
import blocks.native_fusion as blocks_native_fusion_mod
import blocks.policies as blocks_policies_mod
import blocks.registry as blocks_registry_mod
import blocks.schedules as blocks_schedules_mod
import blocks.targets as blocks_targets_mod
import blocks.utils as blocks_utils_mod
import runtime.attention as runtime_attention_mod
import runtime.attention_interfaces as runtime_attention_interfaces_mod
import runtime.attention_reference as runtime_attention_reference_mod
import runtime.block_config as runtime_block_config_mod
import runtime.block_factory as runtime_block_factory_mod
import runtime.block_init as runtime_block_init_mod
import runtime.block_policies as runtime_block_policies_mod
import runtime.block_registry as runtime_block_registry_mod
import runtime.block_schedules as runtime_block_schedules_mod
import runtime.block_targets as runtime_block_targets_mod
import runtime.block_utils as runtime_block_utils_mod
import runtime.blocks as runtime_blocks_mod
import runtime.decoding as runtime_decoding_mod
import runtime.gqa as runtime_gqa_mod
import runtime.inspect as runtime_inspect_mod
import runtime.kv_cache as runtime_kv_cache_mod
import runtime.moe as runtime_moe_mod
import runtime.optim_utils as runtime_optim_utils_mod
import runtime.quant as runtime_quant_mod


def test_attn_helper_modules_are_runtime_aliases():
    assert attn_backends_mod is runtime_attention_mod
    assert attn_decoding_mod is runtime_decoding_mod
    assert attn_gqa_mod is runtime_gqa_mod
    assert attn_interfaces_mod is runtime_attention_interfaces_mod
    assert attn_kv_cache_mod is runtime_kv_cache_mod
    assert attn_moe_mod is runtime_moe_mod
    assert attn_optim_utils_mod is runtime_optim_utils_mod
    assert attn_quant_mod is runtime_quant_mod
    assert attn_reference_mod is runtime_attention_reference_mod


def test_blocks_helper_modules_are_runtime_aliases():
    assert blocks_factory_mod is runtime_block_factory_mod
    assert blocks_registry_mod is runtime_block_registry_mod
    assert blocks_config_mod is runtime_block_config_mod
    assert blocks_init_mod is runtime_block_init_mod
    assert blocks_policies_mod is runtime_block_policies_mod
    assert blocks_schedules_mod is runtime_block_schedules_mod
    assert blocks_targets_mod is runtime_block_targets_mod
    assert blocks_inspect_mod is runtime_inspect_mod
    assert blocks_native_fusion_mod is runtime_blocks_mod
    assert blocks_utils_mod is runtime_block_utils_mod
