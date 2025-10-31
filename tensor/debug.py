import warnings
import torch


def install_nan_guard(module: torch.nn.Module, *, throw: bool = True, sample_tensor_names: tuple[str, ...] = ()):  # type: ignore[name-defined]
    """Install a forward hook that checks for NaN/Inf in inputs/outputs.

    If `throw` is True, raises RuntimeError on detection; otherwise warns.
    """

    def _check(t: torch.Tensor, where: str):
        if not isinstance(t, torch.Tensor):
            return
        if not torch.isfinite(t).all():
            msg = f"NaN/Inf detected in {where}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
            if throw:
                raise RuntimeError(msg)
            warnings.warn(msg)

    def _hook(_mod, inputs, output):
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                _check(inp, f"input[{i}]")
        if isinstance(output, torch.Tensor):
            _check(output, "output")
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    _check(out, f"output[{i}]")

    return module.register_forward_hook(_hook)


def detect_fp16_overflow(tensors: list[torch.Tensor]) -> bool:
    """Return True if any half/bfloat16 tensor contains inf/nan."""
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.dtype in (torch.float16, torch.bfloat16):
            if not torch.isfinite(t).all():
                return True
    return False


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
        return False
    if a.dtype.is_floating_point:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a, b)


def bitwise_equal_forward(module: torch.nn.Module, inputs: tuple, kwargs: dict | None = None) -> bool:
    kwargs = kwargs or {}
    with torch.no_grad():
        out1 = module(*inputs, **kwargs)
        out2 = module(*inputs, **kwargs)
    if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
        return _bitwise_equal(out1, out2)
    if isinstance(out1, (list, tuple)) and isinstance(out2, (list, tuple)):
        if len(out1) != len(out2):
            return False
        return all(_bitwise_equal(a, b) for a, b in zip(out1, out2) if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor))
    return False


def assert_reproducible_step(step_fn, seed: int = 1234, *args, **kwargs):
    import torch
    torch.manual_seed(seed)
    out1 = step_fn(*args, **kwargs)
    torch.manual_seed(seed)
    out2 = step_fn(*args, **kwargs)
    if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
        if not torch.allclose(out1, out2):
            raise AssertionError("Step not reproducible: outputs differ")
        return True
    return True


def numeric_grad_check(fn, x: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference gradient check for scalar fn(x). Returns (grad_numeric, grad_autograd)."""
    x = x.detach().clone().requires_grad_(True)
    y = fn(x)
    y.backward()
    grad_auto = x.grad.detach().clone()
    grad_num = torch.zeros_like(x)
    flat = x.detach().view(-1)
    flat_grad = grad_num.view(-1)
    for i in range(flat.numel()):
        orig = float(flat[i].item())
        flat[i] = orig + eps
        y_pos = float(fn(x.detach()).item())
        flat[i] = orig - eps
        y_neg = float(fn(x.detach()).item())
        flat[i] = orig
        flat_grad[i] = (y_pos - y_neg) / (2 * eps)
    return grad_num, grad_auto


def gradcheck_stateless(fn, x: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference gradient check for stateless scalar fn(x).

    Detaches inputs before numerical probing.
    """
    return numeric_grad_check(fn, x.detach(), eps=eps)


def nan_window_scan(tensors: list[torch.Tensor], window: int = 1) -> int | None:
    """Return first index where any tensor in window has NaN/Inf; None if clean."""
    n = len(tensors)
    for i in range(0, n - window + 1):
        chunk = tensors[i:i + window]
        for t in chunk:
            if isinstance(t, torch.Tensor) and (not torch.isfinite(t).all()):
                return i
    return None


def assert_same_rng_scope(fn, *args, **kwargs) -> bool:
    import torch
    s_cpu = torch.random.get_rng_state()
    out1 = fn(*args, **kwargs)
    s_cpu2 = torch.random.get_rng_state()
    out2 = fn(*args, **kwargs)
    s_cpu3 = torch.random.get_rng_state()
    if not torch.equal(s_cpu2, s_cpu3):
        raise AssertionError("RNG state changed across identical calls; not scope-invariant")
    if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
        if not torch.allclose(out1, out2):
            raise AssertionError("Outputs differ under same scope")
    return True


