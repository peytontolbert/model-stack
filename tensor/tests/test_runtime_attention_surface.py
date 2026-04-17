from __future__ import annotations

import attn
import attn.backends as attn_backends_mod
import attn.decoding as attn_decoding_mod
import attn.eager as attn_eager_mod
import attn.factory as attn_factory_mod
import attn.flash as attn_flash_mod
import attn.gqa as attn_gqa_mod
import attn.interfaces as attn_interfaces_mod
import attn.moe as attn_moe_mod
import attn.optim_utils as attn_optim_utils_mod
import attn.quant as attn_quant_mod
import attn.reference as attn_reference_mod
import attn.triton as attn_triton_mod
import attn.xformers as attn_xformers_mod
import blocks.cross_attn_block as cross_attn_block_mod
import blocks.gpt_block as gpt_block_mod
import blocks.llama_block as llama_block_mod
import blocks.moe_block as moe_block_mod
import blocks.transformer_block as transformer_block_mod
import kernel.bench as kernel_bench_mod
import runtime as runtime_pkg
import runtime.attention as runtime_attention_mod
import runtime.attention_factory as runtime_attention_factory_mod
import runtime.attention_interfaces as runtime_attention_interfaces_mod
import runtime.attention_modules as runtime_attention_modules_mod
import runtime.attention_reference as runtime_attention_reference_mod
import runtime.decoding as runtime_decoding_mod
import runtime.gqa as runtime_gqa_mod
import runtime.kv_cache as runtime_kv_cache_mod
import runtime.moe as runtime_moe_mod
import runtime.optim_utils as runtime_optim_utils_mod
import runtime.quant as runtime_quant_mod


def test_attn_shims_delegate_to_runtime_attention():
    assert attn_backends_mod._read_backend_from_env_or_file is runtime_attention_mod._read_backend_from_env_or_file
    assert attn_backends_mod.select_attention_backend is runtime_attention_mod.select_attention_backend
    assert attn_backends_mod.scaled_dot_product_attention is runtime_attention_mod.scaled_dot_product_attention
    assert attn_factory_mod.build_attention is runtime_attention_factory_mod.build_attention
    assert attn_eager_mod.EagerAttention is runtime_attention_modules_mod.EagerAttention
    assert attn_flash_mod.FlashAttention is runtime_attention_modules_mod.FlashAttention
    assert attn_triton_mod.TritonAttention is runtime_attention_modules_mod.TritonAttention
    assert attn_xformers_mod.XFormersAttention is runtime_attention_modules_mod.XFormersAttention
    assert attn_interfaces_mod.Attention is runtime_attention_interfaces_mod.Attention
    assert attn_interfaces_mod.KVCache is runtime_attention_interfaces_mod.KVCache
    assert attn_reference_mod.compute_attention_scores is runtime_attention_reference_mod.compute_attention_scores
    assert attn_reference_mod.compute_attention_probs is runtime_attention_reference_mod.compute_attention_probs
    assert attn_reference_mod.apply_attention_probs is runtime_attention_reference_mod.apply_attention_probs
    assert attn_reference_mod.attention_reference is runtime_attention_reference_mod.attention_reference
    assert attn_gqa_mod.share_kv_heads is runtime_gqa_mod.share_kv_heads
    assert attn_gqa_mod.rearrange_qkv_for_mqa_gqa is runtime_gqa_mod.rearrange_qkv_for_mqa_gqa
    assert attn_decoding_mod.beam_search is runtime_decoding_mod.beam_search
    assert attn_decoding_mod.mirostat_step is runtime_decoding_mod.mirostat_step
    assert attn_decoding_mod.apply_regex_constraints is runtime_decoding_mod.apply_regex_constraints
    assert attn_moe_mod.topk_router is runtime_moe_mod.topk_router
    assert attn_moe_mod.load_balance_loss is runtime_moe_mod.load_balance_loss
    assert attn_moe_mod.combine_expert_outputs is runtime_moe_mod.combine_expert_outputs
    assert attn_moe_mod.expert_parallel_partition is runtime_moe_mod.expert_parallel_partition
    assert attn_quant_mod.per_channel_absmax is runtime_quant_mod.per_channel_absmax
    assert attn_quant_mod.nf4_quantize is runtime_quant_mod.nf4_quantize
    assert attn_quant_mod.nf4_dequantize is runtime_quant_mod.nf4_dequantize
    assert attn_quant_mod.int8_matmul_qkv is runtime_quant_mod.int8_matmul_qkv
    assert attn_quant_mod.fp8_linear is runtime_quant_mod.fp8_linear
    assert attn_optim_utils_mod.global_grad_norm is runtime_optim_utils_mod.global_grad_norm
    assert attn_optim_utils_mod.clip_grad_norm_ is runtime_optim_utils_mod.clip_grad_norm_
    assert attn_optim_utils_mod.unitwise_l2_norm is runtime_optim_utils_mod.unitwise_l2_norm
    assert attn_optim_utils_mod.unitwise_clip_ is runtime_optim_utils_mod.unitwise_clip_
    assert attn_optim_utils_mod.decay_mask_from_module is runtime_optim_utils_mod.decay_mask_from_module


def test_runtime_package_exports_attention_surface():
    assert runtime_pkg._read_backend_from_env_or_file is runtime_attention_mod._read_backend_from_env_or_file
    assert runtime_pkg.Attention is runtime_attention_interfaces_mod.Attention
    assert runtime_pkg.KVCache is runtime_attention_interfaces_mod.KVCache
    assert runtime_pkg.EagerAttention is runtime_attention_modules_mod.EagerAttention
    assert runtime_pkg.FlashAttention is runtime_attention_modules_mod.FlashAttention
    assert runtime_pkg.TritonAttention is runtime_attention_modules_mod.TritonAttention
    assert runtime_pkg.XFormersAttention is runtime_attention_modules_mod.XFormersAttention
    assert runtime_pkg.select_attention_backend is runtime_attention_mod.select_attention_backend
    assert runtime_pkg.scaled_dot_product_attention is runtime_attention_mod.scaled_dot_product_attention
    assert runtime_pkg.compute_attention_scores is runtime_attention_reference_mod.compute_attention_scores
    assert runtime_pkg.compute_attention_probs is runtime_attention_reference_mod.compute_attention_probs
    assert runtime_pkg.apply_attention_probs is runtime_attention_reference_mod.apply_attention_probs
    assert runtime_pkg.build_attention is runtime_attention_factory_mod.build_attention
    assert runtime_pkg.share_kv_heads is runtime_gqa_mod.share_kv_heads
    assert runtime_pkg.rearrange_qkv_for_mqa_gqa is runtime_gqa_mod.rearrange_qkv_for_mqa_gqa
    assert runtime_pkg.beam_search is runtime_decoding_mod.beam_search
    assert runtime_pkg.mirostat_step is runtime_decoding_mod.mirostat_step
    assert runtime_pkg.apply_regex_constraints is runtime_decoding_mod.apply_regex_constraints
    assert runtime_pkg.topk_router is runtime_moe_mod.topk_router
    assert runtime_pkg.load_balance_loss is runtime_moe_mod.load_balance_loss
    assert runtime_pkg.combine_expert_outputs is runtime_moe_mod.combine_expert_outputs
    assert runtime_pkg.expert_parallel_partition is runtime_moe_mod.expert_parallel_partition
    assert runtime_pkg.per_channel_absmax is runtime_quant_mod.per_channel_absmax
    assert runtime_pkg.nf4_quantize is runtime_quant_mod.nf4_quantize
    assert runtime_pkg.nf4_dequantize is runtime_quant_mod.nf4_dequantize
    assert runtime_pkg.int8_matmul_qkv is runtime_quant_mod.int8_matmul_qkv
    assert runtime_pkg.fp8_linear is runtime_quant_mod.fp8_linear
    assert runtime_pkg.global_grad_norm is runtime_optim_utils_mod.global_grad_norm
    assert runtime_pkg.clip_grad_norm_ is runtime_optim_utils_mod.clip_grad_norm_
    assert runtime_pkg.unitwise_l2_norm is runtime_optim_utils_mod.unitwise_l2_norm
    assert runtime_pkg.unitwise_clip_ is runtime_optim_utils_mod.unitwise_clip_
    assert runtime_pkg.decay_mask_from_module is runtime_optim_utils_mod.decay_mask_from_module


def test_attn_package_exports_runtime_attention_surface():
    assert attn.select_attention_backend is runtime_attention_mod.select_attention_backend
    assert attn.scaled_dot_product_attention is runtime_attention_mod.scaled_dot_product_attention
    assert attn.EagerAttention is runtime_attention_modules_mod.EagerAttention
    assert attn.FlashAttention is runtime_attention_modules_mod.FlashAttention
    assert attn.TritonAttention is runtime_attention_modules_mod.TritonAttention
    assert attn.XFormersAttention is runtime_attention_modules_mod.XFormersAttention
    assert attn.ContiguousKVCache is runtime_kv_cache_mod.ContiguousKVCache
    assert attn.PagedKVCache is runtime_kv_cache_mod.PagedKVCache
    assert attn.share_kv_heads is runtime_gqa_mod.share_kv_heads
    assert attn.beam_search is runtime_decoding_mod.beam_search
    assert attn.topk_router is runtime_moe_mod.topk_router
    assert attn.per_channel_absmax is runtime_quant_mod.per_channel_absmax
    assert attn.global_grad_norm is runtime_optim_utils_mod.global_grad_norm


def test_blocks_and_tools_import_runtime_attention_owners():
    assert gpt_block_mod.build_attention is runtime_attention_factory_mod.build_attention
    assert llama_block_mod.build_attention is runtime_attention_factory_mod.build_attention
    assert cross_attn_block_mod.build_attention is runtime_attention_factory_mod.build_attention
    assert transformer_block_mod.Attention is runtime_attention_interfaces_mod.Attention
    assert moe_block_mod.topk_router is runtime_moe_mod.topk_router
    assert moe_block_mod.combine_expert_outputs is runtime_moe_mod.combine_expert_outputs
    assert moe_block_mod.load_balance_loss is runtime_moe_mod.load_balance_loss
    assert kernel_bench_mod.sdpa is runtime_attention_mod.scaled_dot_product_attention
    assert kernel_bench_mod.select_attention_backend is runtime_attention_mod.select_attention_backend
