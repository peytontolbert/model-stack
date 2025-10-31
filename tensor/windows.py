from tensor.shape import window_partition, window_merge
import torch

__all__ = ["window_partition", "window_merge", "ring_buffer_indices"]


def ring_buffer_indices(T: int, window: int) -> torch.Tensor:
    """Return (T, window) indices selecting the last `window` positions for each time step (clamped at start)."""
    idx = torch.arange(T).view(T, 1)
    offs = torch.arange(window).view(1, window)
    base = idx - (window - 1 - offs)
    base = base.clamp_min(0)
    return base


