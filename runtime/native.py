from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from runtime.hardware import current_cuda_hardware_info


NATIVE_MODULE_NAME = "_model_stack_native"
DISABLE_ENV = "MODEL_STACK_DISABLE_NATIVE"
_FALLBACK_NATIVE_OPS = [
    "activation",
    "gated_activation",
    "embedding",
    "linear",
    "linear_module",
    "bitnet_transform_input",
    "bitnet_linear",
    "bitnet_linear_compute_packed",
    "bitnet_linear_from_float",
    "bitnet_int8_linear_from_float",
    "bitnet_int8_fused_qkv_packed_heads_projection",
    "int4_linear",
    "nf4_linear",
    "fp8_linear",
    "int8_quantize_activation",
    "int8_quantize_activation_transpose",
    "int8_quantize_relu2_activation",
    "int8_quantize_leaky_relu_half2_activation",
    "int8_linear",
    "int8_linear_from_float",
    "int8_linear_grad_weight_from_float",
    "int8_attention",
    "int8_attention_from_float",
    "pack_bitnet_weight",
    "bitnet_runtime_row_quantize",
    "pack_linear_weight",
    "mlp",
    "qkv_projection",
    "pack_qkv_weights",
    "qkv_packed_heads_projection",
    "bitnet_qkv_packed_heads_projection",
    "bitnet_fused_qkv_packed_heads_projection",
    "qkv_heads_projection",
    "split_heads",
    "merge_heads",
    "head_output_projection",
    "prepare_attention_mask",
    "resolve_position_ids",
    "create_causal_mask",
    "resolve_rotary_embedding",
    "token_counts",
    "append_tokens",
    "decode_positions",
    "rms_norm",
    "add_rms_norm",
    "residual_add",
    "layer_norm",
    "add_layer_norm",
    "rope",
    "kv_cache_append",
    "kv_cache_write",
    "kv_cache_gather",
    "paged_kv_assign_blocks",
    "paged_kv_reserve_pages",
    "paged_kv_read_range",
    "paged_kv_read_last",
    "paged_kv_append",
    "paged_kv_compact",
    "paged_kv_gather",
    "paged_kv_write",
    "int3_kv_pack",
    "int3_kv_dequantize",
    "paged_attention_decode",
    "attention_decode",
    "attention_prefill",
    "sampling",
    "beam_search_step",
    "incremental_beam_search",
]


@dataclass(frozen=True)
class NativeRuntimeStatus:
    available: bool
    module_name: str
    info: dict[str, Any]
    error: str | None = None


def _dedupe_str_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_runtime_info(info: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(info)
    native_ops = _dedupe_str_list(normalized.get("native_ops"))
    if not native_ops:
        native_ops = list(_FALLBACK_NATIVE_OPS)
    normalized["native_ops"] = native_ops

    planned_ops = _dedupe_str_list(normalized.get("planned_ops"))
    if not planned_ops:
        planned_ops = list(native_ops)
    normalized["planned_ops"] = planned_ops

    kernel_ops = _dedupe_str_list(normalized.get("cuda_kernel_ops") or normalized.get("cuda_backend_ops"))
    normalized["cuda_backend_ops"] = list(kernel_ops)
    normalized["cuda_kernel_ops"] = list(kernel_ops)

    compiled_with_cuda = bool(normalized.get("compiled_with_cuda", False))
    inference_ops = _dedupe_str_list(normalized.get("cuda_inference_ops"))
    if not inference_ops:
        inference_ops = list(native_ops) if compiled_with_cuda else []
    normalized["cuda_inference_ops"] = inference_ops

    composite_ops = _dedupe_str_list(normalized.get("cuda_composite_ops"))
    if not composite_ops:
        kernel_set = set(kernel_ops)
        composite_ops = [name for name in inference_ops if name not in kernel_set]
    normalized["cuda_composite_ops"] = composite_ops
    normalized["full_cuda_inference"] = bool(compiled_with_cuda and inference_ops)
    hardware = current_cuda_hardware_info()
    for key, value in hardware.items():
        normalized.setdefault(key, value)
    return normalized


@lru_cache(maxsize=1)
def runtime_status() -> NativeRuntimeStatus:
    if os.getenv(DISABLE_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}:
        return NativeRuntimeStatus(
            available=False,
            module_name=NATIVE_MODULE_NAME,
            info={"disabled_by_env": True},
            error=f"{DISABLE_ENV} is set",
        )

    try:
        import torch  # noqa: F401
        module = importlib.import_module(NATIVE_MODULE_NAME)
    except Exception as exc:
        return NativeRuntimeStatus(
            available=False,
            module_name=NATIVE_MODULE_NAME,
            info={"disabled_by_env": False},
            error=str(exc),
        )

    try:
        info = dict(module.runtime_info())
    except Exception as exc:
        return NativeRuntimeStatus(
            available=False,
            module_name=NATIVE_MODULE_NAME,
            info={},
            error=f"native module loaded but runtime_info failed: {exc}",
        )

    return NativeRuntimeStatus(
        available=True,
        module_name=NATIVE_MODULE_NAME,
        info=info,
        error=None,
    )


@lru_cache(maxsize=1)
def _native_module_cached():
    if not native_available():
        return None
    return importlib.import_module(NATIVE_MODULE_NAME)


def native_available() -> bool:
    return runtime_status().available


def runtime_info() -> dict[str, Any]:
    return _normalize_runtime_info(runtime_status().info)


def cuda_kernel_ops() -> list[str]:
    return list(runtime_info().get("cuda_kernel_ops", ()))


def cuda_inference_ops() -> list[str]:
    return list(runtime_info().get("cuda_inference_ops", ()))


def cuda_composite_ops() -> list[str]:
    return list(runtime_info().get("cuda_composite_ops", ()))


def full_cuda_inference_available() -> bool:
    status = runtime_status()
    return bool(status.available and runtime_info().get("full_cuda_inference", False))


def native_module():
    return _native_module_cached()


def has_native_op(name: str) -> bool:
    if not native_available():
        return False
    return str(name) in set(runtime_info().get("native_ops", ()))


def resolve_linear_backend(requested: str | None = None) -> str:
    module = native_module()
    candidate = "auto" if requested is None else str(requested)
    if module is not None and hasattr(module, "resolve_linear_backend"):
        try:
            return str(module.resolve_linear_backend(candidate))
        except Exception:
            pass
    if candidate.strip().lower() in {"", "auto"}:
        return "aten"
    return candidate.strip().lower()


def native_paged_kv_cache_available() -> bool:
    module = native_module()
    if module is None:
        return False
    return bool(hasattr(module, "PagedKvCacheState") or hasattr(module, "PagedKvLayerState"))


def create_native_paged_kv_cache_state(
    *,
    batch: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    pagesize: int,
    dtype,
    device,
) -> tuple[Any | None, list[Any] | None]:
    module = native_module()
    if module is None:
        return None, None

    import torch

    example = torch.empty(
        0,
        int(n_kv_heads),
        max(int(pagesize), 1),
        int(head_dim),
        dtype=dtype,
        device=device,
    )
    if hasattr(module, "PagedKvCacheState"):
        return (
            module.PagedKvCacheState(
                int(batch),
                int(n_layers),
                int(n_kv_heads),
                int(head_dim),
                max(int(pagesize), 1),
                example,
            ),
            None,
        )
    if hasattr(module, "PagedKvLayerState"):
        return (
            None,
            [
                module.PagedKvLayerState(
                    int(batch),
                    int(n_kv_heads),
                    int(head_dim),
                    max(int(pagesize), 1),
                    example,
                )
                for _ in range(int(n_layers))
            ],
        )
    return None, None


def native_model_session_available() -> bool:
    module = native_module()
    return bool(module is not None and hasattr(module, "NativeModelSession"))


def create_native_model_session(model, seq, attention_mask=None, cache=None, trace: bool = False):
    module = native_module()
    if module is None or not hasattr(module, "NativeModelSession"):
        return None
    return module.NativeModelSession(model, seq, attention_mask, cache, bool(trace))
