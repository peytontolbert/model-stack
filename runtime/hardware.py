from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - handled gracefully when torch is unavailable
    torch = None  # type: ignore[assignment]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _device_index(device: Any | None) -> int | None:
    if torch is None or not hasattr(torch, "cuda"):
        return None
    try:
        if device is None:
            if not torch.cuda.is_available():
                return None
            return int(torch.cuda.current_device())
        ref = getattr(device, "device", device)
        dev = ref if isinstance(ref, torch.device) else torch.device(ref)
        if dev.type != "cuda":
            return None
        if dev.index is not None:
            return int(dev.index)
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.current_device())
    except Exception:
        return None


def cuda_device_capability(device: Any | None = None) -> tuple[int, int] | None:
    if torch is None or not hasattr(torch, "cuda"):
        return None
    try:
        index = _device_index(device)
        if index is None:
            return None
        major, minor = torch.cuda.get_device_capability(index)
        return int(major), int(minor)
    except Exception:
        return None


def cuda_device_name(device: Any | None = None) -> str | None:
    if torch is None or not hasattr(torch, "cuda"):
        return None
    try:
        index = _device_index(device)
        if index is None:
            return None
        return str(torch.cuda.get_device_name(index))
    except Exception:
        return None


def cuda_arch_name(device: Any | None = None) -> str | None:
    capability = cuda_device_capability(device)
    if capability is None:
        return None
    major, minor = capability
    return f"sm{major}{minor}"


def cuda_arch_family(device: Any | None = None) -> str | None:
    capability = cuda_device_capability(device)
    if capability is None:
        return None
    major, minor = capability
    if major >= 10:
        return "blackwell"
    if major >= 9:
        return "hopper"
    if major == 8 and minor == 9:
        return "ada"
    if major == 8:
        return "ampere"
    if major == 7:
        return "turing_volta"
    return "legacy"


def is_hopper_device(device: Any | None = None) -> bool:
    capability = cuda_device_capability(device)
    return bool(capability is not None and capability[0] >= 9)


def hopper_optimizations_enabled() -> bool:
    return _env_flag("MODEL_STACK_ENABLE_HOPPER_OPT", "1")


def prefer_hopper_library_attention(
    *,
    device: Any | None,
    dtype,
    q_seq: int,
    kv_seq: int,
    forced_backend: str | None = None,
) -> bool:
    if not hopper_optimizations_enabled():
        return False
    if forced_backend is not None and str(forced_backend).strip().lower() not in {"", "auto"}:
        return False
    if not is_hopper_device(device):
        return False
    if torch is not None and dtype not in (torch.float16, torch.bfloat16):
        return False
    min_seq = _env_int("MODEL_STACK_HOPPER_ATTN_MIN_SEQ", 64)
    return max(int(q_seq), int(kv_seq)) >= max(min_seq, 1)


def prefer_torch_library_attention(
    *,
    device: Any | None,
    dtype,
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    forced_backend: str | None = None,
) -> bool:
    if torch is None:
        return False
    if forced_backend is not None and str(forced_backend).strip().lower() not in {"", "auto"}:
        return False
    index = _device_index(device)
    if index is None:
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    if int(q_seq) <= 1:
        return False
    if is_hopper_device(device):
        return prefer_hopper_library_attention(
            device=device,
            dtype=dtype,
            q_seq=int(q_seq),
            kv_seq=int(kv_seq),
            forced_backend=forced_backend,
        )
    if not _env_flag("MODEL_STACK_ENABLE_TORCH_LIBRARY_ATTENTION", "1"):
        return False
    min_seq = _env_int("MODEL_STACK_TORCH_LIBRARY_ATTN_MIN_SEQ", 16)
    max_head_dim = _env_int("MODEL_STACK_TORCH_LIBRARY_ATTN_MAX_HEAD_DIM", 256)
    return max(int(q_seq), int(kv_seq)) >= max(min_seq, 1) and int(head_dim) <= max(max_head_dim, 1)


def prefer_hopper_library_linear(
    *,
    device: Any | None,
    rows: int,
    out_features: int,
    in_features: int,
) -> bool:
    if not hopper_optimizations_enabled():
        return False
    if not is_hopper_device(device):
        return False
    min_ops = _env_int("MODEL_STACK_HOPPER_LINEAR_MIN_OPS", 2_000_000)
    total_ops = int(rows) * int(out_features) * int(in_features)
    return total_ops >= max(min_ops, 1)


@lru_cache(maxsize=1)
def current_cuda_hardware_info() -> dict[str, Any]:
    return {
        "current_cuda_arch": cuda_arch_name(),
        "current_cuda_arch_family": cuda_arch_family(),
        "current_cuda_device_name": cuda_device_name(),
        "current_cuda_is_hopper": is_hopper_device(),
        "hopper_optimizations_enabled": hopper_optimizations_enabled(),
        "sm90a_experimental_requested": _env_flag("MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL", "0"),
    }


__all__ = [
    "current_cuda_hardware_info",
    "cuda_arch_family",
    "cuda_arch_name",
    "cuda_device_capability",
    "cuda_device_name",
    "hopper_optimizations_enabled",
    "is_hopper_device",
    "prefer_hopper_library_attention",
    "prefer_torch_library_attention",
    "prefer_hopper_library_linear",
]
