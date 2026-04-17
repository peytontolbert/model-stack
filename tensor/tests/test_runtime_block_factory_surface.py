from __future__ import annotations

import blocks
import blocks.config as blocks_config_mod
import blocks.cross_attn_block as blocks_cross_attn_mod
import blocks.decoder_block as blocks_decoder_mod
import blocks.dilated_local_attn_block as blocks_dilated_mod
import blocks.encoder_block as blocks_encoder_mod
import blocks.factory as blocks_factory_mod
import blocks.gpt_block as blocks_gpt_mod
import blocks.init as blocks_init_mod
import blocks.local_attn_block as blocks_local_mod
import blocks.llama_block as blocks_llama_mod
import blocks.moe_block as blocks_moe_mod
import blocks.parallel_block as blocks_parallel_mod
import blocks.policies as blocks_policies_mod
import blocks.prefix_lm_block as blocks_prefix_mod
import blocks.registry as blocks_registry_mod
import blocks.schedules as blocks_schedules_mod
import blocks.segment_bidir_attn_block as blocks_segment_mod
import blocks.shared as blocks_shared_mod
import blocks.strided_attn_block as blocks_strided_mod
import blocks.targets as blocks_targets_mod
import blocks.transformer_block as blocks_transformer_mod
import blocks.window_pattern_attn_block as blocks_window_mod
import blocks.banded_attn_block as blocks_banded_mod
import blocks.block_sparse_attn_block as blocks_block_sparse_mod
import runtime as runtime_pkg
import runtime.block_config as runtime_block_config_mod
import runtime.block_factory as runtime_block_factory_mod
import runtime.block_init as runtime_block_init_mod
import runtime.block_modules as runtime_block_modules_mod
import runtime.block_policies as runtime_block_policies_mod
import runtime.block_registry as runtime_block_registry_mod
import runtime.block_schedules as runtime_block_schedules_mod
import runtime.block_shared as runtime_block_shared_mod
import runtime.block_targets as runtime_block_targets_mod
import runtime.local_attn_block as runtime_local_mod
import runtime.prefix_lm_block as runtime_prefix_mod
import runtime.banded_attn_block as runtime_banded_mod
import runtime.dilated_local_attn_block as runtime_dilated_mod
import runtime.block_sparse_attn_block as runtime_block_sparse_mod
import runtime.segment_bidir_attn_block as runtime_segment_mod
import runtime.window_pattern_attn_block as runtime_window_mod
import runtime.strided_attn_block as runtime_strided_mod


def test_blocks_factory_and_registry_are_runtime_shims():
    assert blocks_factory_mod.build_block is runtime_block_factory_mod.build_block
    assert blocks_factory_mod.build_block_stack is runtime_block_factory_mod.build_block_stack
    assert blocks_registry_mod.register_block is runtime_block_registry_mod.register_block
    assert blocks_registry_mod.get_block_builder is runtime_block_registry_mod.get_block_builder
    assert blocks_registry_mod.list_blocks is runtime_block_registry_mod.list_blocks


def test_blocks_config_init_schedule_policy_and_target_are_runtime_shims():
    assert blocks_config_mod.BlockConfig is runtime_block_config_mod.BlockConfig
    assert blocks_config_mod.build_block_config_from_model is runtime_block_config_mod.build_block_config_from_model
    assert blocks_init_mod.apply_block_init is runtime_block_init_mod.apply_block_init
    assert blocks_init_mod.init_transformer_stack is runtime_block_init_mod.init_transformer_stack
    assert blocks_schedules_mod.drop_path_linear is runtime_block_schedules_mod.drop_path_linear
    assert blocks_policies_mod.NormPolicy == runtime_block_policies_mod.NormPolicy
    assert blocks_policies_mod.validate_policy is runtime_block_policies_mod.validate_policy
    assert blocks_targets_mod.targets_map is runtime_block_targets_mod.targets_map


def test_blocks_package_lazy_exports_factory_surface():
    assert blocks.build_block is runtime_block_factory_mod.build_block
    assert blocks.build_block_stack is runtime_block_factory_mod.build_block_stack
    assert blocks.register_block is runtime_block_registry_mod.register_block
    assert blocks.get_block_builder is runtime_block_registry_mod.get_block_builder
    assert blocks.list_blocks is runtime_block_registry_mod.list_blocks


def test_runtime_package_exports_block_factory_surface():
    assert runtime_pkg.build_block is runtime_block_factory_mod.build_block
    assert runtime_pkg.build_block_stack is runtime_block_factory_mod.build_block_stack
    assert runtime_pkg.register_block is runtime_block_registry_mod.register_block
    assert runtime_pkg.get_block_builder is runtime_block_registry_mod.get_block_builder
    assert runtime_pkg.list_blocks is runtime_block_registry_mod.list_blocks


def test_core_block_modules_are_runtime_shims():
    assert blocks_transformer_mod.TransformerBlock is runtime_block_modules_mod.TransformerBlock
    assert blocks_llama_mod.LlamaBlock is runtime_block_modules_mod.LlamaBlock
    assert blocks_gpt_mod.GPTBlock is runtime_block_modules_mod.GPTBlock
    assert blocks_cross_attn_mod.CrossAttentionBlock is runtime_block_modules_mod.CrossAttentionBlock
    assert blocks_encoder_mod.EncoderBlock is runtime_block_modules_mod.EncoderBlock
    assert blocks_decoder_mod.DecoderBlock is runtime_block_modules_mod.DecoderBlock
    assert blocks_parallel_mod.ParallelTransformerBlock is runtime_block_modules_mod.ParallelTransformerBlock
    assert blocks_moe_mod.MoEMLP is runtime_block_modules_mod.MoEMLP
    assert blocks_moe_mod.MoEBlock is runtime_block_modules_mod.MoEBlock


def test_specialized_block_modules_are_runtime_aliases():
    assert blocks_shared_mod.CausalSelfAttentionBlockBase is runtime_block_shared_mod.CausalSelfAttentionBlockBase
    assert blocks_local_mod.LocalAttentionBlock is runtime_local_mod.LocalAttentionBlock
    assert blocks_prefix_mod.PrefixLMBlock is runtime_prefix_mod.PrefixLMBlock
    assert blocks_banded_mod.BandedAttentionBlock is runtime_banded_mod.BandedAttentionBlock
    assert blocks_dilated_mod.DilatedLocalAttentionBlock is runtime_dilated_mod.DilatedLocalAttentionBlock
    assert blocks_block_sparse_mod.BlockSparseAttentionBlock is runtime_block_sparse_mod.BlockSparseAttentionBlock
    assert blocks_segment_mod.SegmentBidirAttentionBlock is runtime_segment_mod.SegmentBidirAttentionBlock
    assert blocks_window_mod.WindowPatternAttentionBlock is runtime_window_mod.WindowPatternAttentionBlock
    assert blocks_strided_mod.StridedAttentionBlock is runtime_strided_mod.StridedAttentionBlock


def test_runtime_package_exports_core_block_modules():
    assert runtime_pkg.TransformerBlock is runtime_block_modules_mod.TransformerBlock
    assert runtime_pkg.LlamaBlock is runtime_block_modules_mod.LlamaBlock
    assert runtime_pkg.GPTBlock is runtime_block_modules_mod.GPTBlock
    assert runtime_pkg.CrossAttentionBlock is runtime_block_modules_mod.CrossAttentionBlock
    assert runtime_pkg.EncoderBlock is runtime_block_modules_mod.EncoderBlock
    assert runtime_pkg.DecoderBlock is runtime_block_modules_mod.DecoderBlock
    assert runtime_pkg.ParallelTransformerBlock is runtime_block_modules_mod.ParallelTransformerBlock
    assert runtime_pkg.MoEMLP is runtime_block_modules_mod.MoEMLP
    assert runtime_pkg.MoEBlock is runtime_block_modules_mod.MoEBlock


def test_runtime_package_exports_specialized_block_modules():
    assert runtime_pkg.CausalSelfAttentionBlockBase is runtime_block_shared_mod.CausalSelfAttentionBlockBase
    assert runtime_pkg.LocalAttentionBlock is runtime_local_mod.LocalAttentionBlock
    assert runtime_pkg.PrefixLMBlock is runtime_prefix_mod.PrefixLMBlock
    assert runtime_pkg.BandedAttentionBlock is runtime_banded_mod.BandedAttentionBlock
    assert runtime_pkg.DilatedLocalAttentionBlock is runtime_dilated_mod.DilatedLocalAttentionBlock
    assert runtime_pkg.BlockSparseAttentionBlock is runtime_block_sparse_mod.BlockSparseAttentionBlock
    assert runtime_pkg.SegmentBidirAttentionBlock is runtime_segment_mod.SegmentBidirAttentionBlock
    assert runtime_pkg.WindowPatternAttentionBlock is runtime_window_mod.WindowPatternAttentionBlock
    assert runtime_pkg.StridedAttentionBlock is runtime_strided_mod.StridedAttentionBlock


def test_runtime_package_exports_block_config_init_policy_and_targets():
    assert runtime_pkg.BlockConfig is runtime_block_config_mod.BlockConfig
    assert runtime_pkg.build_block_config_from_model is runtime_block_config_mod.build_block_config_from_model
    assert runtime_pkg.apply_block_init is runtime_block_init_mod.apply_block_init
    assert runtime_pkg.init_transformer_stack is runtime_block_init_mod.init_transformer_stack
    assert runtime_pkg.drop_path_linear is runtime_block_schedules_mod.drop_path_linear
    assert runtime_pkg.NormPolicy == runtime_block_policies_mod.NormPolicy
    assert runtime_pkg.validate_policy is runtime_block_policies_mod.validate_policy
    assert runtime_pkg.targets_map is runtime_block_targets_mod.targets_map
