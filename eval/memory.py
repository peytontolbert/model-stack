from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MemoryReport:
    params_mb: float
    buffers_mb: float
    peak_cuda_mb: Optional[float]


def model_memory_footprint_mb(model: torch.nn.Module) -> tuple[float, float]:
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return params / (1024**2), buffers / (1024**2)


@torch.inference_mode()
def measure_peak_cuda_mb(model: torch.nn.Module, fn, *args, **kwargs) -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    device = next(model.parameters()).device
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    _ = fn(*args, **kwargs)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return peak / (1024**2)


def report_memory(model: torch.nn.Module, *, run_forward=None) -> MemoryReport:
    p_mb, b_mb = model_memory_footprint_mb(model)
    peak = None
    if run_forward is not None:
        peak = measure_peak_cuda_mb(model, run_forward)
    return MemoryReport(params_mb=p_mb, buffers_mb=b_mb, peak_cuda_mb=peak)


