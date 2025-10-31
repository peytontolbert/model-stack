import torch


def cast_for_softmax(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def cast_for_norm(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def restore_dtype(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=like.dtype)


def to_dtype_like(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=like.dtype, device=like.device)


def is_fp16(x: torch.Tensor) -> bool:
    return x.dtype == torch.float16


def is_bf16(x: torch.Tensor) -> bool:
    return x.dtype == torch.bfloat16


def is_int8(x: torch.Tensor) -> bool:
    return x.dtype == torch.int8


def is_fp8(x: torch.Tensor) -> bool:
    # Placeholder; PyTorch FP8 dtypes vary by extension
    return str(x.dtype).endswith("fp8")


def cast_logits_for_loss(logits: torch.Tensor) -> torch.Tensor:
    return logits.float()


def set_matmul_precision(level: str = "high"):
    try:
        torch.set_float32_matmul_precision(level)  # type: ignore[attr-defined]
    except Exception:
        pass


class maybe_autocast:
    def __init__(self, enabled: bool = True, dtype: torch.dtype | None = None):
        self.enabled = enabled
        self.dtype = dtype
        self.ctx = None

    def __enter__(self):
        if self.enabled:
            self.ctx = torch.cuda.amp.autocast(dtype=self.dtype)  # type: ignore[attr-defined]
            return self.ctx.__enter__()
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ctx is not None:
            return self.ctx.__exit__(exc_type, exc_val, exc_tb)
        return False


class with_logits_precision:
    def __init__(self, fp32: bool = True):
        self.fp32 = fp32
        self.prev = None

    def __enter__(self):
        # No global state switch in PyTorch; users should call cast on tensors
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# FP8 scale helpers (no kernels)
class FP8AmaxTracker:
    def __init__(self, momentum: float = 0.0):
        self.amax = 0.0
        self.momentum = float(momentum)

    def update(self, x: torch.Tensor) -> float:
        a = float(x.detach().abs().max().item())
        self.amax = self.momentum * self.amax + (1 - self.momentum) * a
        return self.amax


def fp8_scale_from_amax(amax: float, fmt: str = "e4m3") -> float:
    # Simplified: map to scale so dynamic range roughly fits; constants rough
    if amax <= 0:
        return 1.0
    max_code = 240.0 if fmt == "e4m3" else 448.0
    return float(amax / max_code)


def expect_dtype(x: torch.Tensor, allowed: set[torch.dtype] | tuple[torch.dtype, ...]):
    allowed_set = set(allowed)
    if x.dtype not in allowed_set:
        raise TypeError(f"Expected dtype in {allowed_set}, got {x.dtype}")
    return True


def promote_mixed(a: torch.Tensor, b: torch.Tensor, policy: str = "bf16_wins") -> tuple[torch.Tensor, torch.Tensor]:
    if a.dtype == b.dtype:
        return a, b
    if policy == "bf16_wins":
        target = torch.bfloat16 if (a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16) else torch.float16
    elif policy == "fp32":
        target = torch.float32
    else:
        # default to the wider type
        target = torch.promote_types(a.dtype, b.dtype)
    return a.to(dtype=target), b.to(dtype=target)


def amp_policy_for_op(op_name: str, dtype: torch.dtype, fp8: bool = False) -> dict:
    """Return policy dict for autocast behavior.

    Keys: {"autocast": bool, "target_dtype": torch.dtype}
    """
    numerics_sensitive = {"softmax", "log_softmax", "exp", "log", "sum"}
    if fp8:
        return {"autocast": True, "target_dtype": torch.float16}
    if op_name in numerics_sensitive and dtype in (torch.float16, torch.bfloat16):
        return {"autocast": True, "target_dtype": torch.float32}
    return {"autocast": True, "target_dtype": dtype}


def fp8_dynamic_scale_update(amax: float, history: float, momentum: float = 0.0) -> float:
    return momentum * float(history) + (1.0 - float(momentum)) * float(amax)


