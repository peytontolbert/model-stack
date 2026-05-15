from __future__ import annotations

import torch


def closure_residual(
    teacher_vector_field: torch.Tensor,
    student_base_vector_field: torch.Tensor,
    *,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    teacher = teacher_vector_field
    if projection is not None:
        teacher = teacher_vector_field @ projection.to(device=teacher_vector_field.device, dtype=teacher_vector_field.dtype)
    if teacher.shape != student_base_vector_field.shape:
        raise ValueError(f"projected teacher and student fields must match, got {tuple(teacher.shape)} and {tuple(student_base_vector_field.shape)}")
    return teacher - student_base_vector_field


def closure_residual_metrics(residual: torch.Tensor) -> dict[str, float]:
    flat = residual.float().flatten(1) if residual.ndim > 1 else residual.float().view(1, -1)
    rms = flat.pow(2).mean(dim=1).sqrt()
    return {
        "rms_mean": float(rms.mean().item()),
        "rms_max": float(rms.max().item()),
        "l2_mean": float(flat.norm(dim=1).mean().item()),
        "max_abs": float(flat.abs().max().item()),
    }


def closure_target(
    teacher_vector_field: torch.Tensor,
    student_base_vector_field: torch.Tensor,
    *,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    return closure_residual(teacher_vector_field, student_base_vector_field, projection=projection)
