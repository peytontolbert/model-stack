import contextlib
import torch


def pin_if_cpu(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor) and (not x.is_cuda):
        try:
            return x.pin_memory()
        except Exception:
            return x
    return x


def async_to_device(x: torch.Tensor, device: torch.device | str, stream=None) -> torch.Tensor:
    """Non-blocking device transfer. If `stream` is provided, perform copy on that stream and record it."""
    dev = torch.device(device)
    if not isinstance(x, torch.Tensor):
        return x
    if x.device == dev:
        return x
    y = x
    if stream is not None and hasattr(torch.cuda, "stream"):
        try:
            with torch.cuda.stream(stream):  # type: ignore[attr-defined]
                y = x.to(device=dev, non_blocking=True)
        except Exception:
            y = x.to(device=dev, non_blocking=True)
    else:
        y = x.to(device=dev, non_blocking=True)
    # Best-effort record stream for lifetime correctness
    try:
        from .compile import record_stream_guard
        with record_stream_guard([y], stream):
            pass
    except Exception:
        pass
    return y


