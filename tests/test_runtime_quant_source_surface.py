from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_quantization_source_uses_channelwise_reduction_and_cache_invalidation() -> None:
    source = _read("compress/quantization.py")
    assert "def _channel_reduce_dims" in source
    assert "amax(dim=reduce_dims, keepdim=False)" in source
    assert "_reshape_channel_values(inv_s, weight, 0)" in source
    assert "def _load_from_state_dict" in source
    assert "self._invalidate_weight_cache()" in source
    assert "def _pack_int4_signed" in source
    assert "def _unpack_int4_signed" in source
    assert "def apply_spin_transform(" in source
    assert "def undo_spin_transform(" in source
    assert "def collect_linear_calibration_inputs(" in source
    assert "def awq_optimize_pre_scale(" in source
    assert "class QuantizedLinearInt4" in source
    assert "class QuantizedLinearNF4" in source
    assert "class QuantizedLinearBitNet" in source
    assert "runtime_int4_linear(" in source
    assert "runtime_nf4_linear(" in source
    assert "runtime_int8_linear_from_quantized_activation(" in source
    assert "runtime_bitnet_linear(" in source
    assert "def runtime_supports_packed_backend(self, backend: str) -> bool:" in source


def test_quantized_delta_source_persists_qweight_and_invalidates_cache() -> None:
    source = _read("compress/export.py")
    assert "\"qweight\": m.qweight.detach().cpu()" in source
    assert "if \"qweight\" in info:" in source
    assert "m.qweight.copy_(info[\"qweight\"].to(m.qweight.device))" in source
    assert "m._invalidate_weight_cache()" in source
    assert "\"qweight_packed\": m.qweight_packed.detach().cpu()" in source
    assert "if \"qweight_packed\" in info:" in source
    assert "m.qweight_packed.copy_(info[\"qweight_packed\"].to(m.qweight_packed.device))" in source
    assert "\"type\": \"nf4_codebook_packed\"" in source
    assert "\"weight_scale\": m.weight_scale.detach().cpu()" in source
    assert "m.weight_scale.copy_(info[\"weight_scale\"].to(m.weight_scale.device))" in source
    assert "\"spin_signs\": m.spin_signs.detach().cpu()" in source
    assert "\"spin_enabled\": bool(int(m.spin_enabled_flag.item()))" in source
    assert "\"pre_scale\": m.pre_scale.detach().cpu()" in source
    assert "\"act_scale\": m.act_scale.detach().cpu()" in source
    assert "\"weight_opt\": m.weight_opt" in source
    assert "\"act_quant_mode\": m.act_quant_mode" in source
    assert "\"packed_weight\": m.packed_weight.detach().cpu()" in source
    assert "if \"packed_weight\" in info:" in source
    assert "m.packed_weight = info[\"packed_weight\"].to(m.packed_weight.device, dtype=m.packed_weight.dtype)" in source


def test_fp8_quantization_source_threads_through_compression_and_export() -> None:
    quant_source = _read("compress/quantization.py")
    apply_source = _read("compress/apply.py")
    export_source = _read("export/exporter.py")
    delta_source = _read("compress/export.py")
    cli_source = _read("export/cli.py")
    spec_source = _read("specs/export.py")
    runtime_quant_source = _read("runtime/quant.py")
    runtime_init_source = _read("runtime/__init__.py")
    assert "class QuantizedLinearFP8" in quant_source
    assert "class QuantizedLinearNF4" in quant_source
    assert "runtime_fp8_linear(" in quant_source
    assert "elif quant_scheme == \"bitnet\":" in quant_source
    assert "elif quant_scheme == \"int4\":" in quant_source
    assert "elif quant_scheme == \"nf4\":" in quant_source
    assert "elif quant_scheme == \"fp8\":" in quant_source
    assert "\"scheme\": str = \"int8\"" in apply_source
    assert "scheme=str(quant.get(\"scheme\", \"int8\"))" in apply_source
    assert "calibration_inputs=quant.get(\"calibration_inputs\")" in apply_source
    assert "weight_opt=str(quant.get(\"weight_opt\", \"none\"))" in apply_source
    assert "activation_quant=str(quant.get(\"activation_quant\", \"none\"))" in apply_source
    assert "spin=bool(quant.get(\"spin\", False))" in apply_source
    assert "apply_compression(model, quant={**quant_cfg, \"scheme\": \"int4\"})" in export_source
    assert "apply_compression(model, quant={**quant_cfg, \"scheme\": \"nf4\"})" in export_source
    assert "apply_compression(model, quant={**quant_cfg, \"scheme\": \"fp8\"})" in export_source
    assert "apply_compression(model, quant={**quant_cfg, \"scheme\": \"bitnet\"})" in export_source
    assert "meta[\"quantize\"] = str(cfg.quantize)" in export_source
    assert "meta[\"quant_spin\"] = bool(cfg.quant_spin)" in export_source
    assert "meta[\"quant_weight_opt\"] = str(cfg.quant_weight_opt)" in export_source
    assert "meta[\"quant_activation_quant\"] = str(cfg.quant_activation_quant)" in export_source
    assert "meta[\"quant_calibration_inputs_path\"] = str(cfg.quant_calibration_inputs_path)" in export_source
    assert "def _load_quant_calibration_inputs(" in export_source
    assert "torch.load(Path(path), map_location=\"cpu\")" in export_source
    assert "\"calibration_inputs\": calibration_inputs" in export_source
    assert "\"weight_opt\": str(getattr(cfg, \"quant_weight_opt\", \"none\"))" in export_source
    assert "\"activation_quant\": \"none\"" in export_source
    assert "\"int8\", \"int4\", \"nf4\", \"fp8\", \"bitnet\"" in cli_source
    assert "quant_spin=args.quant_spin" in cli_source
    assert "\"--quant-weight-opt\"" in cli_source
    assert "\"--quant-activation-quant\"" in cli_source
    assert "\"--quant-calibration-inputs\"" in cli_source
    assert "quant_weight_opt=args.quant_weight_opt" in cli_source
    assert "quant_activation_quant=args.quant_activation_quant" in cli_source
    assert "quant_calibration_inputs_path=args.quant_calibration_inputs" in cli_source
    assert "Literal[\"int8\",\"int4\",\"nf4\",\"fp8\",\"bitnet\"]" in spec_source
    assert "quant_weight_opt: Literal[\"none\", \"awq\", \"gptq\"] = \"none\"" in spec_source
    assert "quant_activation_quant: Optional[Literal[\"static_int8\", \"dynamic_int8\"]] = None" in spec_source
    assert "quant_calibration_inputs_path: Optional[str] = None" in spec_source
    assert "def nf4_linear(" in runtime_quant_source
    assert "def int4_linear(" in runtime_quant_source
    assert "def int8_linear(" in runtime_quant_source
    assert "def int8_matmul_qkv(" in runtime_quant_source
    assert "def bitnet_linear(" in runtime_quant_source
    assert "def bitnet_linear_compute_packed(" in runtime_quant_source
    assert "def bitnet_linear_from_float(" in runtime_quant_source
    assert "prefer_hopper_library_attention" in runtime_quant_source
    assert "prefer_hopper_library_linear" not in runtime_quant_source
    assert "module.int8_linear_forward(" in runtime_quant_source
    assert "has_native_op(\"bitnet_linear_from_float\")" in runtime_quant_source
    assert "has_native_op(\"bitnet_linear_compute_packed\")" in runtime_quant_source
    assert "module.bitnet_linear_compute_packed_forward(" in runtime_quant_source
    assert "module.bitnet_linear_from_float_forward(" in runtime_quant_source
    assert "module.int8_attention_forward(" in runtime_quant_source
    assert "native_mask_ok" in runtime_quant_source
    assert "module.int4_linear_forward(x_cast, packed_cast, scale_cast, bias_cast)" in runtime_quant_source
    assert "\"nf4_linear\": \"runtime.quant\"" in runtime_init_source
    assert "\"bitnet_linear\": \"runtime.quant\"" in runtime_init_source
    assert "\"bitnet_linear_from_float\": \"runtime.quant\"" in runtime_init_source
    assert "\"int4_linear\": \"runtime.quant\"" in runtime_init_source
    assert "\"int8_linear\": \"runtime.quant\"" in runtime_init_source
    assert "\"int8_linear_grad_weight_from_float\": \"runtime.quant\"" in runtime_init_source
    assert "\"type\": \"nf4_codebook_packed\"" in delta_source
    assert "\"type\": \"bitnet_w2a8\"" in delta_source
    assert "\"type\": \"int4_pc_packed\"" in delta_source
    assert "\"type\": \"fp8_fake\"" in delta_source
    assert "m.weight_fp8.copy_(info[\"weight_fp8\"].to(m.weight_fp8.device))" in delta_source


def test_runtime_sources_use_module_aware_linear_quantization_path() -> None:
    ops_source = _read("runtime/ops.py")
    attn_source = _read("runtime/attention_modules.py")
    mlp_source = _read("tensor/mlp.py")
    causal_source = _read("runtime/causal.py")
    seq2seq_source = _read("runtime/seq2seq.py")
    heads_source = _read("runtime/heads.py")
    adapters_source = _read("runtime/block_adapters.py")
    block_source = _read("runtime/block_modules.py")
    quant_source = _read("compress/quantization.py")
    runtime_quant_source = _read("runtime/quant.py")
    backward_bench_source = _read("examples/13_parameter_golf_h100/bench_pg_bitnet_backward.py")
    generation_source = _read("runtime/generation.py")
    bitnet_frontend_source = _read("runtime/csrc/backend/bitnet/bitnet_frontend.cu")
    cuda_int8_linear_source = _read("runtime/csrc/backend/cuda_int8_linear.cu")
    cuda_quant_source = _read("runtime/csrc/backend/cuda_quant_int8_frontend.cu")
    native_source = _read("runtime/csrc/model_stack_native.cpp")
    pg_run_source = _read("examples/13_parameter_golf_h100/run_pg_8xh100.sh")
    assert "def resolve_linear_module_tensors(" in ops_source
    assert "def linear_module(" in ops_source
    assert "def mlp_module(" in ops_source
    assert "def _try_squared_activation_dynamic_int8_down_projection(" in ops_source
    assert "def _try_relu2_dynamic_int8_down_projection(" in ops_source
    assert "int8_quantize_relu2_activation_forward" in ops_source
    assert "int8_quantize_leaky_relu_half2_activation_forward" in ops_source
    assert "runtime_linear_from_quantized_input" in ops_source
    assert "def linear_module_signature(" in ops_source
    assert "def packed_linear_module_signature(" in ops_source
    assert "def resolve_packed_linear_module_spec(" in ops_source
    assert "def packed_qkv_module_signature(" in ops_source
    assert "def resolve_packed_qkv_module_spec(" in ops_source
    assert "def linear_from_packed_spec(" in ops_source
    assert "def bitnet_transform_input(" in ops_source
    assert "has_native_op(\"bitnet_transform_input\")" in ops_source
    assert "module.bitnet_transform_input_forward(" in ops_source
    assert "def bitnet_qkv_packed_heads_projection(" in ops_source
    assert "has_native_op(\"bitnet_qkv_packed_heads_projection\")" in ops_source
    assert "module.bitnet_qkv_packed_heads_projection_forward(" in ops_source
    assert "def qkv_packed_spec_heads_projection(" in ops_source
    assert "def head_output_packed_projection(" in ops_source
    assert "def runtime_weight(" in quant_source
    assert "def runtime_linear(self, x: torch.Tensor" in quant_source
    assert "def runtime_shared_int8_input_signature(" in quant_source
    assert "def runtime_quantize_int8_input(" in quant_source
    assert "def runtime_linear_from_quantized_input(" in quant_source
    assert "runtime_bitnet_int8_linear_from_float(" in quant_source
    assert "def _native_decode_graph_enabled_by_env()" in generation_source
    assert "def _native_decode_graph_enabled_for_model(" in generation_source
    assert "MODEL_STACK_ENABLE_BITNET_DYNAMIC_INT8_DECODE_GRAPH" in generation_source
    assert "def _try_build_native_decode_graph_replay(" in generation_source
    assert "MODEL_STACK_DISABLE_NATIVE_DECODE_GRAPH" in generation_source
    assert "def _clear_native_decode_graph(self) -> None:" in generation_source
    assert "def runtime_packed_linear_signature(self, backend: str):" in quant_source
    assert "def runtime_packed_linear_spec(" in quant_source
    assert "def _pre_scale_active_runtime(self) -> bool:" in quant_source
    assert "def _prefer_dynamic_int8_direct_hopper_prefill(self, x: torch.Tensor) -> bool:" in quant_source
    assert "MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED" in quant_source
    assert "(1024, 3072),  # Parameter Golf relu2 MLP up projection." in quant_source
    assert "(3072, 1024),  # Parameter Golf relu2 MLP down projection." in quant_source
    assert "(1024, 4096),  # Parameter Golf swiglu fused gate/up projection." in quant_source
    assert "model_stack::bitnet_runtime_row_quantize" in ops_source
    assert "_BITNET_RUNTIME_ROW_QUANTIZE_COMPILE_OP" in ops_source
    assert "model_stack::bitnet_int8_linear_from_float" in runtime_quant_source
    assert "_BITNET_INT8_LINEAR_FROM_FLOAT_COMPILE_OP" in runtime_quant_source
    assert "model_stack::int8_quantize_activation_transpose" in runtime_quant_source
    assert "_INT8_QUANTIZE_ACTIVATION_TRANSPOSE_COMPILE_OP" in runtime_quant_source
    assert "model_stack::int8_linear" in runtime_quant_source
    assert "_INT8_LINEAR_COMPILE_OP" in runtime_quant_source
    assert "model_stack::int8_linear_grad_weight_from_float" in runtime_quant_source
    assert "_INT8_LINEAR_GRAD_WEIGHT_FROM_FLOAT_COMPILE_OP" in runtime_quant_source
    assert "runtime_int8_linear_grad_weight_from_float(" in quant_source
    assert "model_stack::trainable_bitnet_int8_ste" in quant_source
    assert "_TRAINABLE_BITNET_INT8_STE_COMPILE_OP" in quant_source
    assert "model_stack::trainable_bitnet_int8_ste_output" in quant_source
    assert "_TRAINABLE_BITNET_INT8_STE_OUTPUT_COMPILE_OP" in quant_source
    assert "MODEL_STACK_TRAINABLE_BITNET_OUTPUT_COMPILE_OP" in quant_source
    assert "MODEL_STACK_TRAINABLE_BITNET_DISABLE_COMPILED_AUTOGRAD_FUNCTION" in quant_source
    assert "op.register_autograd(backward, setup_context=setup_context)" in quant_source
    assert "MODEL_STACK_TRAINABLE_BITNET_COMPILED_INT8_STE" in quant_source
    assert "MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE" in quant_source
    assert "MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT" in quant_source
    assert "MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT" in quant_source
    assert "def _trainable_bitnet_save_backward_context(" in quant_source
    assert "MODEL_STACK_INT8_TRANSPOSE_TILE_DIM" in cuda_quant_source
    assert "kTransposeWideTileDim = 64" in cuda_quant_source
    assert "int8_linear_grad_weight_transpose_tile_options" in native_source
    assert "pg_h100_expansion" in quant_source
    assert "(1024, 2048),  # Parameter Golf relu2 MLP up projection, MLP_MULT=2." in quant_source
    training_bench_source = _read("examples/13_parameter_golf_h100/bench_pg_bitnet_training_step.py")
    kernel_bench_source = _read("examples/13_parameter_golf_h100/bench_pg_bitnet_kernels.py")
    mlp_subgraph_bench_source = _read("examples/13_parameter_golf_h100/bench_pg_bitnet_mlp_subgraph.py")
    assert "def _reset_torch_compile_cache" in training_bench_source
    assert "--include-relu2-mlp-pair" in training_bench_source
    assert "--include-pg-block" in training_bench_source
    assert "--block-fused-qkv" in training_bench_source
    assert "--no-preset-shapes" in training_bench_source
    assert "--grad-weight-mode" in training_bench_source
    assert 'BITNET_STE_MODES = ("dynamic_int8_ste", "dynamic_int4_ste")' in training_bench_source
    assert "def _bitnet_optimized_training_env(" in training_bench_source
    assert '"MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT",' in training_bench_source
    assert '"MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT",' in training_bench_source
    assert '"bitnet_training_env": {key: value for key, value in env.items() if value is not None}' in training_bench_source
    assert "def _bitnet_shape_gate_expected_allows(" in training_bench_source
    assert '"bitnet_shape_gate_expected_allows": _bitnet_shape_gate_expected_allows(' in training_bench_source
    assert 'if mode_name in {"dynamic_int4", "dynamic_a4"}:' in kernel_bench_source
    assert '"canonical_activation_quant": str(activation_quant)' in kernel_bench_source
    assert "int(activation_quant_bits) == 8" in kernel_bench_source
    assert 'if mode_name in {"dynamic_int4", "dynamic_a4"}:' in mlp_subgraph_bench_source
    assert '"canonical_activation_quant": str(activation_quant)' in mlp_subgraph_bench_source
    components_bench_source = _read("examples/13_parameter_golf_h100/bench_pg_training_components.py")
    assert '"MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT",' in components_bench_source
    assert '"MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT",' in components_bench_source
    assert '"bitnet_training_env": {key: value for key, value in env.items() if value is not None}' in components_bench_source
    backward_bench_source = _read("examples/13_parameter_golf_h100/bench_pg_bitnet_backward.py")
    assert "--act-quant-bits" in backward_bench_source
    assert "--no-preset-shapes" in backward_bench_source
    assert '"act_quant_bits": int(act_quant_bits)' in backward_bench_source
    assert "\"MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED\": os.environ.get(" in training_bench_source
    assert "MODEL_STACK_ATTENTION_REPEAT_KV" in training_bench_source
    assert "def _maybe_repeat_kv_heads(" in training_bench_source
    assert "relu2_mlp_pair_1024_3072_1024" in training_bench_source
    assert "MODEL_STACK_BITNET_QAT=\"${MODEL_STACK_BITNET_QAT:-1}\"" in pg_run_source
    assert "MODEL_STACK_BITNET_ACT_QUANT=\"${MODEL_STACK_BITNET_ACT_QUANT:-${MODEL_STACK_BITNET_ACTIVATION_QUANT:-none}}\"" in pg_run_source
    assert "MODEL_STACK_BITNET_ACT_BITS=\"${MODEL_STACK_BITNET_ACT_BITS:-${MODEL_STACK_BITNET_ACTIVATION_BITS:-8}}\"" in pg_run_source
    assert "MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE=\"${MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE:-pg_h100_mlp}\"" in pg_run_source
    assert (
        "MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT="
        "\"${MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT:-dynamic_int8_explicit_scale}\""
        in pg_run_source
    )
    assert (
        "MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT="
        "\"${MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT:-dynamic_int8_transpose}\""
        in pg_run_source
    )
    assert "MODEL_STACK_ATTENTION_REPEAT_KV=\"${MODEL_STACK_ATTENTION_REPEAT_KV:-}\"" in pg_run_source
    assert "MODEL_STACK_MUON_DISTRIBUTED_EXCHANGE=\"${MODEL_STACK_MUON_DISTRIBUTED_EXCHANGE:-all_gather}\"" in pg_run_source
    assert "MODEL_STACK_MUON_DISTRIBUTED_SHARDING=\"${MODEL_STACK_MUON_DISTRIBUTED_SHARDING:-shape_bucket}\"" in pg_run_source
    assert "MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK=\"${MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK:-4}\"" in pg_run_source
    assert "MODEL_STACK_INT8_COLUMN_QUANT_THREADS_X=\"${MODEL_STACK_INT8_COLUMN_QUANT_THREADS_X:-32}\"" in pg_run_source
    assert "MODEL_STACK_INT8_COLUMN_QUANT_ROWS_PER_BLOCK=\"${MODEL_STACK_INT8_COLUMN_QUANT_ROWS_PER_BLOCK:-128}\"" in pg_run_source
    assert (
        "Int8LinearCutlassFusedEnabled() && !use_row1 && t10::cuda::DeviceIsSm90OrLater(qx_contig)"
        in cuda_int8_linear_source
    )
    assert "int8_grad_input_explicit_scale_speedup_vs_dense" in backward_bench_source
    assert "dense_grad_weight_ms" in backward_bench_source
    assert "--include-packed-bitnet" in backward_bench_source
    assert "--include-int8-grad-weight" in backward_bench_source
    assert "--include-int8-grad-weight-upper-bound" in backward_bench_source
    assert "int8_grad_weight_full_speedup_vs_dense" in backward_bench_source
    assert "int8_quantize_activation_transpose_forward" in backward_bench_source
    assert "int8_grad_weight_fused_full_speedup_vs_dense" in backward_bench_source
    assert "int8_grad_weight_native_composed_speedup_vs_dense" in backward_bench_source
    assert "int8_grad_weight_fused_delayed_scale_full_speedup_vs_dense" in backward_bench_source
    assert "raw_original_rowwise_int_mm_scale_correct" in backward_bench_source
    assert "original row-wise activation scales multiply inside the reduction" in backward_bench_source
    component_bench_source = _read("examples/13_parameter_golf_h100/bench_pg_training_components.py")
    assert "bench_pg_bitnet_training_step.py" in component_bench_source
    assert "muon_matrix_optimizer" in component_bench_source
    assert "backward_plus_overhead_ms" in component_bench_source
    assert "attention_repeat_kv" in component_bench_source
    assert "current_flat_alloc" in component_bench_source
    assert "cached_flat" in component_bench_source
    assert "direct_single_rank" in component_bench_source
    attention_variants_source = _read("examples/13_parameter_golf_h100/bench_pg_attention_variants.py")
    assert "enable_gqa=True" in attention_variants_source
    assert "repeat_contiguous" in attention_variants_source
    assert "expand_reshape" in attention_variants_source
    assert "speedup_vs_gqa" in attention_variants_source
    train_gpt_source = _read("other_repos/parameter-golf/train_gpt.py")
    assert "MODEL_STACK_ATTENTION_REPEAT_KV" in train_gpt_source
    assert "def maybe_repeat_kv_heads(" in train_gpt_source
    assert "k, v, enable_gqa = maybe_repeat_kv_heads(" in train_gpt_source
    assert "self._updates_flat_cache" in train_gpt_source
    assert "if not distributed:" in train_gpt_source
    assert "updates_flat.zero_()" in train_gpt_source
    assert "\"act_quant_percentile\": float(self.act_quant_percentile)" in quant_source
    assert "runtime_mlp_module(" in mlp_source
    assert "runtime_linear_module(x, self.w_in)" in mlp_source
    assert "return runtime_linear_module(x, self.w_out)" in mlp_source
    assert "runtime_linear_module(x, self.w_q)" in attn_source
    assert "runtime_packed_qkv_module_signature(self.w_q, self.w_k, self.w_v, backend=backend)" in attn_source
    assert "runtime_resolve_packed_qkv_module_spec(self.w_q, self.w_k, self.w_v, backend=backend, reference=reference)" in attn_source
    assert "runtime_qkv_packed_spec_heads_projection(" in attn_source
    assert "def _packed_output_backend(" in attn_source
    assert "return self._select_packed_backend(x, (self.w_o,))" in attn_source
    assert "def _shared_int8_qkv_input_signature(" in attn_source
    assert "def _shared_int8_qkv_projection(" in attn_source
    assert "def _supports_int8_attention_core(" in attn_source
    assert "def _int8_attention(" in attn_source
    assert "def _prefer_bitnet_w2a8_frontend_path(" not in attn_source
    assert "shared_int8_signature = self._shared_int8_qkv_input_signature()" in attn_source
    assert "has_native_op(\"int8_linear_from_float\")" in attn_source
    assert "elif shared_int8_signature is not None and not prefer_cuda_fused_int8_runtime_linear:" in attn_source
    assert "out = self._int8_attention(" in attn_source
    assert "attn_mask=add" in attn_source
    assert "runtime_head_output_packed_projection(" in attn_source
    assert "packed_output_backend = self._packed_output_backend(out)" in attn_source
    assert "runtime_resolve_linear_module_tensors(self.w_q, reference=x)" in attn_source
    assert "runtime_resolve_packed_linear_module_spec(self.w_o, backend=backend, reference=reference)" in attn_source
    assert "if callable(runtime_linear):" in attn_source
    assert "if not callable(supports_packed_backend):" in attn_source
    assert "runtime_linear_module(x, self.lm_head)" in causal_source
    assert "runtime_linear_module(x, self.lm_head)" in seq2seq_source
    assert "runtime_linear_module(pooled, self.proj)" in heads_source
    assert "runtime_linear_module(x, self.down)" in adapters_source
    assert "runtime_linear_module(x, self.router)" in block_source
    assert "runtime_supports_packed_backend" in attn_source
    native_source = _read("runtime/csrc/model_stack_native.cpp")
    native_py_source = _read("runtime/native.py")
    cache_source = _read("runtime/cache.py")
    kv_cache_source = _read("runtime/kv_cache.py")
    setup_source = _read("setup.py")
    assert "bool PyCallableAttr(" in native_source
    assert "struct BitNetModuleState" in native_source
    assert "{\"bitnet_transform_input\", true}" in native_source
    assert "{\"bitnet_linear_compute_packed\", true}" in native_source
    assert "{\"bitnet_linear_from_float\", true}" in native_source
    assert "{\"bitnet_int8_linear_from_float\", true}" in native_source
    assert "{\"bitnet_int8_fused_qkv_packed_heads_projection\", true}" in native_source
    assert "{\"bitnet_runtime_row_quantize\", true}" in native_source
    assert "{\"int8_quantize_activation\", true}" in native_source
    assert "{\"int8_quantize_activation_transpose\", true}" in native_source
    assert "{\"int8_linear_grad_weight_from_float\", true}" in native_source
    assert "{\"int8_quantize_relu2_activation\", true}" in native_source
    assert "{\"int8_quantize_leaky_relu_half2_activation\", true}" in native_source
    assert "bool HasCudaBitNetInputFrontendKernel()" in native_source
    assert "CudaInt8QuantizeActivationForward(" in native_source
    assert "CudaInt8QuantizeRelu2ActivationForward(" in native_source
    assert "CudaInt8QuantizeLeakyReluHalf2ActivationForward(" in native_source
    assert "CudaInt8LinearFromFloatPreScaleForward(" in native_source
    assert "Int8QuantizeRelu2ActivationForward" in native_source
    assert "Int8QuantizeLeakyReluHalf2ActivationForward" in native_source
    assert "int8_quantize_relu2_activation_forward" in native_source
    assert "int8_quantize_leaky_relu_half2_activation_forward" in native_source
    assert "quantize_relu2_activation_int8_rowwise_wide_cached_kernel" in cuda_quant_source
    assert "quantize_leaky_relu_half2_activation_int8_rowwise_wide_cached_kernel" in cuda_quant_source
    assert "quantize_activation_int8_rowwise_warp_pre_scale_kernel" in cuda_quant_source
    assert "QuantizeActivationInt8RowwisePreScaleCuda" in cuda_quant_source
    assert "MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK" in native_source
    assert "MODEL_STACK_DISABLE_INT8_QUANT_SHARED_CACHE" in native_source
    assert "MODEL_STACK_ENABLE_INT8_QUANT_VEC4" in native_source
    assert "MODEL_STACK_INT8_COLUMN_QUANT_ROWS_PER_BLOCK" in cuda_quant_source
    assert "MODEL_STACK_INT8_COLUMN_QUANT_THREADS_X" in cuda_quant_source
    assert "cuda_backend_ops.push_back(\"int8_quantize_activation\");" in native_source
    assert "cuda_backend_ops.push_back(\"int8_quantize_activation_transpose\");" in native_source
    assert "cuda_backend_ops.push_back(\"bitnet_transform_input\");" in native_source
    assert "cuda_backend_ops.push_back(\"bitnet_linear_compute_packed\");" in native_source
    assert "cuda_backend_ops.push_back(\"bitnet_runtime_row_quantize\");" in native_source
    assert "cuda_backend_ops.push_back(\"bitnet_linear_from_float\");" in native_source
    assert "cuda_backend_ops.push_back(\"bitnet_int8_fused_qkv_packed_heads_projection\");" in native_source
    assert "m.def(\"bitnet_transform_input_forward\"" in native_source
    assert "bitnet_linear_compute_packed_forward" in native_source
    assert "m.def(\"bitnet_linear_from_float_forward\"" in native_source
    assert "bitnet_runtime_row_quantize_forward" in native_source
    assert "m.def(\"int8_quantize_activation_forward\"" in native_source
    assert "m.def(\"int8_quantize_activation_transpose_forward\"" in native_source
    assert "m.def(\"int8_linear_grad_weight_from_float_forward\"" in native_source
    assert "m.def(\"int8_quantize_relu2_activation_forward\"" in native_source
    assert "m.def(\"int8_quantize_leaky_relu_half2_activation_forward\"" in native_source
    assert "bitnet_int8_linear_from_float_forward" in native_source
    assert "bitnet_int8_fused_qkv_packed_heads_projection_forward" in native_source
    assert "bool TryLoadBitNetModuleState(" in native_source
    assert "CudaBitNetTransformInputForward(" in native_source
    assert "CudaBitNetQuantizeActivationInt8CodesForward(" in native_source
    assert "torch::Tensor ApplyBitNetModuleInputTransforms(" in native_source
    assert "torch::Tensor BitNetLinearFromFloatForward(" in native_source
    assert "torch::Tensor BitNetLinearStateForward(" in native_source
    assert "bool DecodeGraphEligible() const" in native_source
    assert "void SetDecodeGraphEnabled(bool enabled = true)" in native_source
    assert "torch::Tensor BitNetInt8LinearFromFloatForward(" in native_source
    assert "TryBitNetGatedInt8LinearStateForward(" in native_source
    assert "bool PreferDirectBitNetRuntimeLinear(" in native_source
    assert "bool CanUseCutlassDirectBitNetPrefillPolicy(" in native_source
    assert "bool CanUseDenseBitNetPrefillPolicy(" in native_source
    assert "bool CanUseModuleBitNetRuntimePolicy(" in native_source
    assert "m.def(\"linear_module_forward\"" in native_source
    assert "\"linear_module\"" in native_py_source
    assert "\"bitnet_runtime_row_quantize\"" in native_py_source
    assert "\"int8_quantize_activation_transpose\"" in native_py_source
    assert "\"int8_quantize_relu2_activation\"" in native_py_source
    assert "\"int8_quantize_leaky_relu_half2_activation\"" in native_py_source
    assert "return BitNetLinearStateForward(x, bitnet_state);" in native_source
    assert "PyCallableAttr(module, \"_spin_enabled_runtime\")" in native_source
    assert "PyCallableAttr(module, \"_pre_scale_active_runtime\")" in native_source
    assert "const bool int8_qkv =" in native_source
    assert "bool ModuleHasRuntimeLinear(" in native_source
    assert "torch::Tensor PythonLinearModuleForward(" in native_source
    assert "if (BitNetModuleDirectSupported(module, &bitnet_state)) {" in native_source
    assert "bool BitNetActivationCalibrationMethodSupported(" in native_source
    assert "state.act_quant_percentile" in native_source
    assert "state.act_quant_bits >= 2" in native_source
    assert "bool AttentionQkvSupportsPackedBitNet(" in native_source
    assert "runtime_ops.attr(\"resolve_packed_qkv_module_spec\")" in native_source
    assert "runtime_ops.attr(\"qkv_packed_spec_heads_projection\")" in native_source
    assert "runtime_ops.attr(\"head_output_packed_projection\")" in native_source
    assert "auto w_in = mlp.attr(\"w_in\");" in native_source
    assert "auto hidden = LinearLikeModuleForward(x, w_in, \"auto\");" in native_source
    assert "return LinearLikeModuleForward(hidden, w_out, \"auto\");" in native_source
    assert "return BitNetQkvPackedHeadsProjectionForward(" in native_source
    assert "auto logits = LinearLikeModuleForward(x, moe.attr(\"router\"), \"auto\");" in native_source
    assert "auto lm_head = model.attr(\"lm_head\");" in native_source
    assert "return LinearLikeModuleForward(x, lm_head, \"auto\");" in native_source
    assert "int64_t MaxLength() const { return max_length_; }" in native_source
    assert ".def(\"max_length\", &PagedKvLayerState::MaxLength)" in native_source
    assert ".def(\"decode_graph_eligible\", &NativeModelSession::DecodeGraphEligible)" in native_source
    assert ".def(\"set_decode_graph_enabled\", &NativeModelSession::SetDecodeGraphEnabled" in native_source
    assert "return Layer(layer_idx)->MaxLength();" in native_source
    assert "return py::cast<int64_t>(layer.attr(\"max_length\")());" in native_source
    assert "if hasattr(self.parent, \"layer_max_length\"):" in cache_source
    assert "return int(self.parent.layer_max_length(self.layer_idx))" in cache_source
    assert "def layer_max_length(self, layer_idx: int) -> int:" in kv_cache_source
    assert "if hasattr(layer, \"max_length\"):" in kv_cache_source
    assert "runtime/csrc/backend/bitnet/bitnet_frontend.cu" in setup_source
    assert "bitnet_quantize_input_int8_codes_nospin_kernel" in bitnet_frontend_source
    assert "bitnet_quantize_gated_activation_int8_codes_nospin_kernel" in bitnet_frontend_source
    assert "CudaBitNetQuantizeActivationInt8CodesForward(" in bitnet_frontend_source
    assert "CudaBitNetQuantizeGatedActivationInt8CodesForward(" in bitnet_frontend_source


def test_quantized_wrappers_accept_float_checkpoint_keys_on_load() -> None:
    quant_source = _read("compress/quantization.py")
    assert "weight_key = prefix + \"weight\"" in quant_source
    assert "if weight_key in state_dict:" in quant_source
    assert "self._assign_float_state(weight, bias)" in quant_source
    assert "qweight_key = prefix + \"qweight\"" in quant_source
    assert "qweight_packed_key = prefix + \"qweight_packed\"" in quant_source
    assert "weight_scale_key = prefix + \"weight_scale\"" in quant_source
    assert "fp8_weight_key = prefix + \"weight_fp8\"" in quant_source


def test_native_int4_kernel_is_registered_in_source() -> None:
    native_source = _read("runtime/csrc/model_stack_native.cpp")
    cuda_source = _read("runtime/csrc/backend/cuda_int4_linear.cu")
    cuda_int8_source = _read("runtime/csrc/backend/cuda_int8_linear.cu")
    cuda_int8_attention_source = _read("runtime/csrc/backend/cuda_int8_attention.cu")
    cublaslt_source = _read("runtime/csrc/backend/cublaslt_linear.cu")
    cuda_arch_source = _read("runtime/csrc/backend/cuda_device_arch.cuh")
    cuda_hopper_source = _read("runtime/csrc/backend/cuda_hopper_advanced.cuh")
    decode_source = _read("runtime/csrc/backend/attention/cuda_attention_decode.cuh")
    prefill_source = _read("runtime/csrc/backend/attention/cuda_attention_prefill.cuh")
    setup_source = _read("setup.py")
    native_py_source = _read("runtime/native.py")
    assert "{\"bitnet_linear\", true}" in native_source
    assert "\"bitnet_linear_compute_packed\"" in native_source
    assert "{\"pack_bitnet_weight\", true}" in native_source
    assert "{\"bitnet_runtime_row_quantize\", true}" in native_source
    assert "{\"bitnet_qkv_packed_heads_projection\", true}" in native_source
    assert "info[\"bitnet_linear_dtypes\"]" in native_source
    assert "m.def(\"bitnet_linear_forward\"" in native_source
    assert "m.def(\"pack_bitnet_weight_forward\"" in native_source
    assert "bitnet_runtime_row_quantize_forward" in native_source
    assert "m.def(\"bitnet_qkv_packed_heads_projection_forward\"" in native_source
    assert "{\"int4_linear\", true}" in native_source
    assert "info[\"int4_linear_dtypes\"]" in native_source
    assert "m.def(\"int4_linear_forward\"" in native_source
    assert "CudaInt4LinearForward(" in native_source
    assert "HasCudaInt4LinearKernel()" in native_source
    assert "{\"int8_linear\", true}" in native_source
    assert "info[\"int8_linear_dtypes\"]" in native_source
    assert "\"int8_linear_forward\"" in native_source
    assert "CudaInt8LinearForward(" in native_source
    assert "HasCudaInt8LinearKernel()" in native_source
    assert "{\"int8_attention\", true}" in native_source
    assert "info[\"int8_attention_dtypes\"]" in native_source
    assert "info[\"int8_attention_kernel_family\"]" in native_source
    assert "info[\"int8_attention_tensorcore_tile\"]" in native_source
    assert "info[\"int8_attention_specializations\"]" in native_source
    assert "info[\"int8_attention_decode_specialized_env\"]" in native_source
    assert "row_rescale" in cuda_int8_attention_source
    assert "m.def(\"int8_attention_forward\"" in native_source
    assert "CudaInt8AttentionForward(" in native_source
    assert "HasCudaInt8AttentionKernel()" in native_source
    assert "int4_linear_forward_kernel" in cuda_source
    assert "int4_linear_forward_tiled_kernel" in cuda_source
    assert "int4_linear_forward_sm90_vectorized_kernel" in cuda_source
    assert "int4_linear_forward_sm90_imma_kernel" in cuda_source
    assert "MODEL_STACK_ENABLE_INT4_IMMA_ACT_QUANT" in cuda_source
    assert "wmma::experimental::precision::s4" in cuda_source
    assert "int8_linear_forward_kernel" in cuda_int8_source
    assert "int8_linear_forward_row1_kernel" in cuda_int8_source
    assert "int8_linear_forward_tiled_kernel" in cuda_int8_source
    assert "int8_linear_forward_sm90a_wgmma_kernel" in cuda_int8_source
    assert "int8_linear_forward_sm90_tensorcore_kernel" in cuda_int8_source
    assert "CublasLtInt8LinearForward" in cuda_int8_source
    assert "MODEL_STACK_INT8_LINEAR_CUBLASLT_MIN_OPS" in cuda_int8_source
    assert "MODEL_STACK_ENABLE_INT8_LINEAR_WGMMA" in cuda_int8_source
    assert "MODEL_STACK_DISABLE_INT8_LINEAR_WGMMA" in cuda_int8_source
    assert "MODEL_STACK_INT8_LINEAR_WGMMA_MIN_OPS" in cuda_int8_source
    assert "MODEL_STACK_DISABLE_INT8_LINEAR_WMMA" in cuda_int8_source
    assert "#include <mma.h>" in cuda_int8_source
    assert "wmma::mma_sync" in cuda_int8_source
    assert "WgmmaM64N8K32S32U8S8" in cuda_int8_source
    assert "weight_correction" in cuda_int8_source
    assert "MakeWgmmaSmemDesc" in cuda_int8_source
    assert "AsyncProxyFenceSharedCta" in cuda_int8_source
    assert "RunCublasLtInt8LinearAccum" in cublaslt_source
    assert "CublasLtInt8LinearForward" in cublaslt_source
    assert "MODEL_STACK_DISABLE_INT8_LINEAR_CUBLASLT" in cublaslt_source
    assert "MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL" in setup_source
    assert "maybe_enable_sm90a_target" in setup_source
    assert "9.0a" in setup_source
    assert "BuildRequestsSm90aExperimental" in cuda_hopper_source
    assert "ArchHasSm90aFeatures" in cuda_hopper_source
    assert "WgmmaFenceSyncAligned" in cuda_hopper_source
    assert "WgmmaCommitGroupSyncAligned" in cuda_hopper_source
    assert "WgmmaWaitGroupSyncAligned" in cuda_hopper_source
    assert "WgmmaDescriptor" in cuda_hopper_source
    assert "MakeWgmmaSmemDesc" in cuda_hopper_source
    assert "AsyncProxyFenceSharedCta" in cuda_hopper_source
    assert "WgmmaM64N8K32S32U8S8" in cuda_hopper_source
    assert "int8_attention_forward_generic_kernel" in cuda_int8_attention_source
    assert "int8_attention_forward_tensorcore_kernel" in cuda_int8_attention_source
    assert "int8_attention_forward_sm90_pipeline_kernel" in cuda_int8_attention_source
    assert "launch_int8_attention_sm90_pipeline_if_supported" in cuda_int8_attention_source
    assert "#include <cuda/barrier>" in cuda_int8_attention_source
    assert "#include <mma.h>" in cuda_int8_attention_source
    assert "#include <sm_61_intrinsics.h>" in cuda_int8_attention_source
    assert "wmma::mma_sync" in cuda_int8_attention_source
    assert "MODEL_STACK_DISABLE_INT8_ATTENTION_WMMA" in cuda_int8_attention_source
    assert "MODEL_STACK_DISABLE_INT8_ATTENTION_SM90_PIPELINE" in cuda_int8_attention_source
    assert "MODEL_STACK_DISABLE_INT8_ATTENTION_SM90_BULK_ASYNC" in cuda_int8_attention_source
    assert "MODEL_STACK_ENABLE_INT8_ATTENTION_DECODE_SPECIALIZED" in cuda_int8_attention_source
    assert "MODEL_STACK_ENABLE_INT8_ATTENTION_WGMMA" in cuda_int8_attention_source
    assert "MODEL_STACK_DISABLE_INT8_ATTENTION_WGMMA" in cuda_int8_attention_source
    assert "MODEL_STACK_INT8_ATTENTION_WGMMA_MIN_WORK" in cuda_int8_attention_source
    assert "MODEL_STACK_ENABLE_INT8_ATTENTION_PERSISTENT" in cuda_int8_attention_source
    assert "MODEL_STACK_DISABLE_INT8_ATTENTION_PERSISTENT" in cuda_int8_attention_source
    assert "MODEL_STACK_INT8_ATTENTION_PERSISTENT_WAVES" in cuda_int8_attention_source
    assert "MODEL_STACK_INT8_ATTENTION_OPTIMIZED_MIN_WORK" in cuda_int8_attention_source
    assert "MODEL_STACK_INT8_ATTENTION_OPTIMIZED_SMALL_SEQ_MIN_HEAD_DIM" in cuda_int8_attention_source
    assert "MODEL_STACK_DISABLE_INT8_ATTENTION_OPTIMIZED" in cuda_int8_attention_source
    assert "SupportsInt8AttentionWgmmaPath" in cuda_int8_attention_source
    assert "SupportsInt8AttentionPersistentPath" in cuda_int8_attention_source
    assert "PersistentAttentionBlockCount" in cuda_int8_attention_source
    assert "__pipeline_memcpy_async" in cuda_int8_attention_source
    assert "Sm90CpAsyncBulkGlobalToShared" in cuda_int8_attention_source
    assert "cp_async_bulk_global_to_shared" in cuda_hopper_source
    assert "Sm90BarrierArriveTx" in cuda_int8_attention_source
    assert "barrier_arrive_tx" in cuda_hopper_source
    assert "Sm90FenceProxyAsyncSharedCta" in cuda_int8_attention_source
    assert "fence_proxy_async_shared_cta" in cuda_hopper_source
    assert "int8_attention_forward_sm90a_wgmma_kernel" in cuda_int8_attention_source
    assert "compute_int8_score_tile_wgmma" in cuda_int8_attention_source
    assert "launch_int8_attention_sm90a_wgmma_if_supported" in cuda_int8_attention_source
    assert "launch_int8_attention_tensorcore_specialized" in cuda_int8_attention_source
    assert "launch_int8_attention_decode_if_supported" in cuda_int8_attention_source
    assert "int8_attention_decode_nomask_kernel" in cuda_int8_attention_source
    assert "if constexpr (HasBoolMask)" in cuda_int8_attention_source
    assert "if constexpr (HasAdditiveMask)" in cuda_int8_attention_source
    assert "if constexpr (IsCausal)" in cuda_int8_attention_source
    assert "DeviceIsSm90OrLater" in cuda_arch_source
    assert "decode_attention_q1_hdim_sm90_forward_kernel" in decode_source
    assert "prefill_attention_hdim_sm90_forward_kernel" in prefill_source
    assert "sm90_specialized_ops" in native_source
    assert "sm90a_advanced_ops" in native_source
    assert "attention_arches" in native_source
    assert "int4_linear_arches" in native_source
    assert "int4_linear_kernel_family" in native_source
    assert "int4_linear_sm90_tile" in native_source
    assert "int4_linear_imma_tile" in native_source
    assert "int4_linear_imma_requires" in native_source
    assert "int8_linear_arches" in native_source
    assert "int8_linear_kernel_family" in native_source
    assert "int8_linear_tensorcore_tile" in native_source
    assert "int8_linear_tensorcore_arches" in native_source
    assert "int8_linear_wgmma_tile" in native_source
    assert "int8_linear_wgmma_requires" in native_source
    assert "int8_linear_wgmma_env" in native_source
    assert "int8_linear_wgmma_min_ops_env" in native_source
    assert "int8_linear_wgmma_activation_strategy" in native_source
    assert "int8_linear_large_gemm_backend" in native_source
    assert "sm90a_experimental_build_requested" in native_source
    assert "int8_linear_wgmma_build_requested" in native_source
    assert "int8_attention_wgmma_tile" in native_source
    assert "int8_attention_wgmma_head_dims" in native_source
    assert "int8_attention_wgmma_requires" in native_source
    assert "int8_attention_wgmma_env" in native_source
    assert "int8_attention_wgmma_disable_env" in native_source
    assert "int8_attention_wgmma_min_work_env" in native_source
    assert "int8_attention_sm90_bulk_async" in native_source
    assert "int8_attention_sm90_bulk_async_requires" in native_source
    assert "int8_attention_scheduler" in native_source
    assert "int8_attention_persistent_env" in native_source
    assert "int8_attention_persistent_disable_env" in native_source
    assert "int8_attention_persistent_waves_env" in native_source
    assert "int8_attention_persistent_waves_default" in native_source
    assert "int8_attention_persistent_requires" in native_source
    assert "int8_attention_wgmma_build_requested" in native_source
    assert "int8_attention_sm90_pipeline_stages" in native_source
    assert "int8_attention_optimized_default" in native_source
    assert "int8_attention_optimized_min_work_default" in native_source
    assert "int8_attention_optimized_small_seq_min_head_dim_default" in native_source
    assert "int8_attention_optimized_min_work_env" in native_source
    assert "int8_attention_optimized_small_seq_min_head_dim_env" in native_source
    assert "runtime/csrc/backend/cuda_int4_linear.cu" in setup_source
    assert "runtime/csrc/backend/cuda_int8_attention.cu" in setup_source
    assert "runtime/csrc/backend/cuda_int8_linear.cu" in setup_source
    assert "\"bitnet_linear\"" in native_py_source
    assert "\"bitnet_linear_compute_packed\"" in native_py_source
    assert "\"bitnet_linear_from_float\"" in native_py_source
    assert "\"bitnet_int8_linear_from_float\"" in native_py_source
    assert "\"bitnet_int8_fused_qkv_packed_heads_projection\"" in native_py_source
    assert "\"bitnet_transform_input\"" in native_py_source
    assert "\"pack_bitnet_weight\"" in native_py_source
    assert "\"bitnet_runtime_row_quantize\"" in native_py_source
    assert "\"bitnet_qkv_packed_heads_projection\"" in native_py_source
    assert "\"bitnet_fused_qkv_packed_heads_projection\"" in native_py_source
    assert "\"int4_linear\"" in native_py_source
    assert "\"int8_linear\"" in native_py_source
    assert "\"int8_quantize_activation_transpose\"" in native_py_source
    assert "\"int8_linear_grad_weight_from_float\"" in native_py_source
    assert "\"int8_attention\"" in native_py_source


def test_native_bitnet_cuda_kernel_sources_are_registered_in_source() -> None:
    native_source = _read("runtime/csrc/model_stack_native.cpp")
    ops_source = _read("runtime/ops.py")
    compress_source = _read("compress/quantization.py")
    setup_source = _read("setup.py")
    pack_source = _read("runtime/csrc/backend/bitnet/bitnet_pack.cu")
    decode_source = _read("runtime/csrc/backend/bitnet/bitnet_linear_decode.cu")
    prefill_source = _read("runtime/csrc/backend/bitnet/bitnet_linear_prefill.cu")
    dispatch_source = _read("runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu")
    frontend_source = _read("runtime/csrc/backend/bitnet/bitnet_frontend.cu")
    bitnet_attn_decode_source = _read("runtime/csrc/backend/bitnet/bitnet_attention_decode_dispatch.cu")
    bitnet_attn_prefill_source = _read("runtime/csrc/backend/bitnet/bitnet_attention_prefill_dispatch.cu")
    bitnet_attn_dispatch_source = _read("runtime/csrc/backend/bitnet/bitnet_attention_dispatch.cu")
    bitnet_attn_common_source = _read("runtime/csrc/backend/bitnet/bitnet_attention_common.cuh")
    common_source = _read("runtime/csrc/backend/bitnet/bitnet_common.cuh")
    formats_source = _read("runtime/csrc/backend/bitnet/bitnet_formats.h")
    bench_linear_source = _read("tests/bench_bitnet_linear.py")
    bench_attention_source = _read("tests/bench_bitnet_attention.py")
    bench_decode_source = _read("tests/bench_bitnet_decode.py")

    assert "#include \"backend/bitnet/bitnet_formats.h\"" in native_source
    assert "t10::bitnet::CudaBitNetLinearForward(" in native_source
    assert "t10::bitnet::CudaPackBitNetWeightForward(weight)" in native_source
    assert "t10::bitnet::CudaBitNetRuntimeRowQuantizeForward(weight, eps)" in native_source
    assert "t10::bitnet::HasCudaBitNetLinearKernel()" in native_source
    assert "bitnet_kernel_family" in native_source
    assert "decode_persistent_prefill_tiled_splitk" in native_source
    assert "bitnet_attention_kernel_family" in native_source
    assert "bitnet_fused_qkv_packed_heads_projection" in native_source
    assert "bitnet_int8_fused_qkv_packed_heads_projection" in native_source
    assert "BitNetStateUsesInt8PackedPath" in native_source
    assert "AttentionUsesBitNetInt8AttentionCore" in native_source
    assert "ModuleUsesBitNetInt8PackedPath" in native_source
    assert "ModelUsesBitNetInt8PackedPath" in native_source
    assert "\"_compute_backend_weight\"" in native_source
    assert "\"_decode_backend_weight\"" in native_source
    assert "\"_int8_backend_weight\"" in native_source
    assert "if int(self.out_features) >= 32768:" in compress_source
    assert "CudaBitNetLinearForwardComputePacked(" in native_source
    assert "CudaBitNetRmsNormLinearForwardDecodeRows(" in native_source
    assert "CudaBitNetAddRmsNormLinearForwardDecodeRows(" in native_source
    assert "BitNetDecodeFusedNormRowsEnabled(" in native_source
    assert "MODEL_STACK_ENABLE_BITNET_DECODE_FUSED_NORM_ROWS" in native_source
    assert "MODEL_STACK_DISABLE_BITNET_DECODE_FUSED_NORM_ROWS" in native_source
    assert "one row-local scale per token row" in frontend_source
    assert frontend_source.count("CudaBitNetCalibrateInputScaleForward(") == 1
    assert "return (row_max.clamp_min(1.0e-8f) / qmax).to(torch::kFloat32);" in frontend_source
    assert "row_max.amax().reshape({1})" not in frontend_source
    assert "dynamic_int4" in dispatch_source
    assert "dynamic_int4" in frontend_source
    assert "compute_packed_words" in native_source
    assert "decode_nz_masks" in native_source
    assert "FusedRmsNormBitNetLinearStateForward(" in native_source
    assert "FusedAddRmsNormBitNetLinearStateForward(" in native_source
    assert "TryFusedRmsNormPreparedAttentionQkv(" in native_source
    assert "ExecutePreparedAttentionProjected(" in native_source
    assert "ExecutePreparedMlpHidden(" in native_source
    assert "LaunchBitNetDecodeKernelBitplaneRow1(" in dispatch_source
    assert "LaunchBitNetDecodeKernelBitplaneRow1RmsNorm(" in dispatch_source
    assert "LaunchBitNetDecodeKernelBitplaneRow1AddRmsNorm(" in dispatch_source
    assert "LaunchBitNetDecodeKernelComputePackedRmsNorm(" in dispatch_source
    assert "LaunchBitNetDecodeKernelComputePackedAddRmsNorm(" in dispatch_source
    assert "LaunchBitNetPrefillKernelComputePacked(" in dispatch_source
    assert "LaunchBitNetPrefillSplitKKernelComputePacked(" in dispatch_source
    assert "if (plan.kind != KernelKind::kDecodePersistent)" not in dispatch_source
    assert "BitNetInt8FusedQkvPackedHeadsProjectionForward(" in native_source
    assert "return PythonLinearModuleForward(merged, module, \"auto\");" in native_source
    assert "return BitNetLinearStateForward(merged, state);" in native_source
    assert "ModelUsesAttentionBiases" in native_source
    assert "SupportsNativeCausalRuntimeInputs" in native_source
    assert "if (ModelUsesBitNetInt8PackedPath(model)) {" not in native_source
    assert "cache.is_none() || x.size(1) != 1 || token_attention_mask.has_value() ||" in native_source
    assert "Int8AttentionFromFloatForward(" in native_source
    assert "bitnet_qkv_fused_int8" in ops_source
    assert "bitnet_int8_fused_qkv_packed_heads_projection_forward" in ops_source
    assert "bitnet_w2a8_int8" in ops_source
    assert "_pack_bitnet_compute_weight" in compress_source
    assert "_pack_bitnet_decode_backend_weight" in compress_source
    assert "def _compute_backend_weight(" in compress_source
    assert "def _decode_backend_weight(" in compress_source
    assert "bitnet_w2a8_int8" in compress_source
    assert "bitnet_decode_rows_buckets" in native_source
    assert "bitnet_decode_scheduler" in native_source
    assert "bitnet_prefill_scheduler" in native_source
    assert "bitnet_splitk_env" in native_source
    assert "bitnet_persistent_decode_env" in native_source
    assert "runtime/csrc/backend/bitnet/bitnet_pack.cu" in setup_source
    assert "runtime/csrc/backend/bitnet/bitnet_linear_decode.cu" in setup_source
    assert "runtime/csrc/backend/bitnet/bitnet_linear_prefill.cu" in setup_source
    assert "runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu" in setup_source
    assert "runtime/csrc/backend/bitnet/bitnet_attention_decode_dispatch.cu" in setup_source
    assert "runtime/csrc/backend/bitnet/bitnet_attention_prefill_dispatch.cu" in setup_source
    assert "runtime/csrc/backend/bitnet/bitnet_attention_dispatch.cu" in setup_source
    assert "bitnet_pack_weight_kernel" in pack_source
    assert "bitnet_runtime_row_quantize_kernel" in pack_source
    assert "bitnet_linear_decode_scalar_kernel" in decode_source
    assert "bitnet_linear_decode_persistent_kernel" in decode_source
    assert "bitnet_linear_prefill_tiled_kernel" in prefill_source
    assert "bitnet_linear_prefill_splitk_kernel" in prefill_source
    assert "LaunchBitNetAttentionDecodePackedQkv" in bitnet_attn_decode_source
    assert "LaunchBitNetAttentionPrefillPackedQkv" in bitnet_attn_prefill_source
    assert "CudaBitNetFusedQkvPackedHeadsProjectionForward" in bitnet_attn_dispatch_source
    assert "SplitBitNetFusedQkv" in bitnet_attn_common_source
    assert "CudaBitNetLinearForward(" in dispatch_source
    assert "ResolvePlan(" in dispatch_source
    assert "KernelKind::kPrefillSplitK" in dispatch_source
    assert "LaunchBitNetPrefillSplitKKernel" in dispatch_source
    assert "LaunchBitNetDecodeKernel" in common_source
    assert "LaunchBitNetPrefillKernel" in common_source
    assert "KernelKindName" in common_source
    assert "ResolvePlan(" in common_source
    assert "kPrefillSplitK" in common_source
    assert "MODEL_STACK_DISABLE_BITNET_SPLITK" in common_source
    assert "MODEL_STACK_DISABLE_BITNET_PERSISTENT_DECODE" in common_source
    assert "getCurrentCUDAStream" in pack_source
    assert "getCurrentCUDAStream" in decode_source
    assert "getCurrentCUDAStream" in prefill_source
    assert "getDefaultCUDAStream" not in pack_source
    assert "getDefaultCUDAStream" not in decode_source
    assert "getDefaultCUDAStream" not in prefill_source
    assert "struct LayoutInfo" in formats_source
    assert "def main() -> None:" in bench_linear_source
    assert "def main() -> None:" in bench_attention_source
    assert "def main() -> None:" in bench_decode_source
    assert "dense_bitnet_ref_ms" in bench_linear_source
    assert "bitnet_native_ms" in bench_linear_source
    assert "native_max_abs_err_vs_bitnet_ref" in bench_linear_source
    assert "dense_bitnet_ref_ms" in bench_attention_source
    assert "max_abs_err_vs_bitnet_ref" in bench_attention_source
    assert "\"--packed-backend\"" in bench_attention_source
    assert "native_dense_executor_kind" in bench_decode_source
    assert "dense_bitnet_ref_ms" in bench_decode_source
    assert "max_abs_err_vs_bitnet_ref" in bench_decode_source
    assert "\"--no-native-session\"" in bench_decode_source


def test_native_int8_attention_benchmark_script_is_registered_in_source() -> None:
    bench_source = _read("tests/bench_int8_attention.py")

    assert "def main() -> None:" in bench_source
    assert "runtime_info()" in bench_source
    assert "int8_matmul_qkv" in bench_source
    assert "MODEL_STACK_DISABLE_INT8_ATTENTION_OPTIMIZED" in bench_source
    assert "MODEL_STACK_ENABLE_INT8_ATTENTION_DECODE_SPECIALIZED" in bench_source
    assert "MODEL_STACK_ENABLE_INT8_ATTENTION_WGMMA" in bench_source
    assert "MODEL_STACK_DISABLE_INT8_ATTENTION_WGMMA" in bench_source
    assert "MODEL_STACK_ENABLE_INT8_ATTENTION_PERSISTENT" in bench_source
    assert "MODEL_STACK_DISABLE_INT8_ATTENTION_PERSISTENT" in bench_source
    assert "\"--wgmma-opt-in\"" in bench_source
    assert "\"--persistent-opt-in\"" in bench_source
    assert "int8_attention_scheduler" in bench_source
    assert "int8_attention_wgmma_tile" in bench_source


def test_native_int8_attention_vs_flash3_benchmark_script_is_registered_in_source() -> None:
    bench_source = _read("tests/bench_int8_attention_vs_flash3.py")

    assert "def main() -> None:" in bench_source
    assert "runtime_info()" in bench_source
    assert "int8_matmul_qkv" in bench_source
    assert "flash_attn_func" in bench_source
    assert "FLASH3_HOPPER" in bench_source
    assert "\"--native-wgmma\"" in bench_source
    assert "\"--native-persistent\"" in bench_source
    assert "native_speedup_vs_flash3" in bench_source


def test_native_int8_linear_benchmark_script_is_registered_in_source() -> None:
    bench_source = _read("tests/bench_int8_linear.py")

    assert "def main() -> None:" in bench_source
    assert "runtime_info()" in bench_source
    assert "int8_linear_from_quantized_activation" in bench_source
    assert "quantize_activation_int8_rowwise" in bench_source
    assert "MODEL_STACK_ENABLE_INT8_LINEAR_WGMMA" in bench_source
    assert "MODEL_STACK_DISABLE_INT8_LINEAR_WGMMA" in bench_source
    assert "MODEL_STACK_DISABLE_INT8_LINEAR_CUBLASLT" in bench_source
