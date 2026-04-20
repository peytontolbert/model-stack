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
    assert "def bitnet_linear_from_float(" in runtime_quant_source
    assert "prefer_hopper_library_attention" in runtime_quant_source
    assert "prefer_hopper_library_linear" not in runtime_quant_source
    assert "module.int8_linear_forward(" in runtime_quant_source
    assert "has_native_op(\"bitnet_linear_from_float\")" in runtime_quant_source
    assert "module.bitnet_linear_from_float_forward(" in runtime_quant_source
    assert "module.int8_attention_forward(" in runtime_quant_source
    assert "native_mask_ok" in runtime_quant_source
    assert "module.int4_linear_forward(x_cast, packed_cast, scale_cast, bias_cast)" in runtime_quant_source
    assert "\"nf4_linear\": \"runtime.quant\"" in runtime_init_source
    assert "\"bitnet_linear\": \"runtime.quant\"" in runtime_init_source
    assert "\"bitnet_linear_from_float\": \"runtime.quant\"" in runtime_init_source
    assert "\"int4_linear\": \"runtime.quant\"" in runtime_init_source
    assert "\"int8_linear\": \"runtime.quant\"" in runtime_init_source
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
    bitnet_frontend_source = _read("runtime/csrc/backend/bitnet/bitnet_frontend.cu")
    assert "def resolve_linear_module_tensors(" in ops_source
    assert "def linear_module(" in ops_source
    assert "def mlp_module(" in ops_source
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
    assert "def runtime_packed_linear_signature(self, backend: str):" in quant_source
    assert "def runtime_packed_linear_spec(" in quant_source
    assert "def _pre_scale_active_runtime(self) -> bool:" in quant_source
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
    cache_source = _read("runtime/cache.py")
    kv_cache_source = _read("runtime/kv_cache.py")
    setup_source = _read("setup.py")
    assert "bool PyCallableAttr(" in native_source
    assert "struct BitNetModuleState" in native_source
    assert "{\"bitnet_transform_input\", true}" in native_source
    assert "{\"bitnet_linear_from_float\", true}" in native_source
    assert "{\"bitnet_int8_linear_from_float\", true}" in native_source
    assert "{\"bitnet_int8_fused_qkv_packed_heads_projection\", true}" in native_source
    assert "bool HasCudaBitNetInputFrontendKernel()" in native_source
    assert "cuda_backend_ops.push_back(\"bitnet_transform_input\");" in native_source
    assert "cuda_backend_ops.push_back(\"bitnet_linear_from_float\");" in native_source
    assert "cuda_backend_ops.push_back(\"bitnet_int8_fused_qkv_packed_heads_projection\");" in native_source
    assert "m.def(\"bitnet_transform_input_forward\"" in native_source
    assert "m.def(\"bitnet_linear_from_float_forward\"" in native_source
    assert "bitnet_int8_linear_from_float_forward" in native_source
    assert "bitnet_int8_fused_qkv_packed_heads_projection_forward" in native_source
    assert "bool TryLoadBitNetModuleState(" in native_source
    assert "CudaBitNetTransformInputForward(" in native_source
    assert "CudaBitNetQuantizeActivationInt8CodesForward(" in native_source
    assert "torch::Tensor ApplyBitNetModuleInputTransforms(" in native_source
    assert "torch::Tensor BitNetLinearFromFloatForward(" in native_source
    assert "torch::Tensor BitNetLinearStateForward(" in native_source
    assert "torch::Tensor BitNetInt8LinearFromFloatForward(" in native_source
    assert "TryBitNetGatedInt8LinearStateForward(" in native_source
    assert "torch::Tensor BitNetLinearModuleForward(" in native_source
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
    assert "{\"pack_bitnet_weight\", true}" in native_source
    assert "{\"bitnet_qkv_packed_heads_projection\", true}" in native_source
    assert "info[\"bitnet_linear_dtypes\"]" in native_source
    assert "m.def(\"bitnet_linear_forward\"" in native_source
    assert "m.def(\"pack_bitnet_weight_forward\"" in native_source
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
    assert "\"bitnet_linear_from_float\"" in native_py_source
    assert "\"bitnet_int8_linear_from_float\"" in native_py_source
    assert "\"bitnet_int8_fused_qkv_packed_heads_projection\"" in native_py_source
    assert "\"bitnet_transform_input\"" in native_py_source
    assert "\"pack_bitnet_weight\"" in native_py_source
    assert "\"bitnet_qkv_packed_heads_projection\"" in native_py_source
    assert "\"bitnet_fused_qkv_packed_heads_projection\"" in native_py_source
    assert "\"int4_linear\"" in native_py_source
    assert "\"int8_linear\"" in native_py_source
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
    assert "\"_int8_backend_weight\"" in native_source
    assert "BitNetInt8FusedQkvPackedHeadsProjectionForward(" in native_source
    assert "return BitNetLinearStateForward(MergeHeadsForward(x), state);" in native_source
    assert "ModelUsesAttentionBiases" in native_source
    assert "SupportsNativeCausalRuntimeInputs" in native_source
    assert "if (ModelUsesBitNetInt8PackedPath(model)) {" not in native_source
    assert "cache.is_none() || x.size(1) != 1 || token_attention_mask.has_value() ||" in native_source
    assert "Int8AttentionFromFloatForward(" in native_source
    assert "bitnet_qkv_fused_int8" in ops_source
    assert "bitnet_int8_fused_qkv_packed_heads_projection_forward" in ops_source
    assert "bitnet_w2a8_int8" in ops_source
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
