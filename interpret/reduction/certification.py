from __future__ import annotations

import torch


def reduced_order_certification(
    *,
    balanced_energy_retained: float,
    rg_error_cached: float,
    rg_error_fresh: float,
    closure_residual_rms: float,
    endpoint_error: float,
    prompt_delta_error: float | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    th = {
        "balanced_energy_retained": 0.95,
        "rg_error_cached": 0.05,
        "rg_error_fresh": 0.10,
        "closure_residual_rms": 0.05,
        "endpoint_error": 0.10,
        "prompt_delta_error": 0.10,
    }
    if thresholds:
        th.update(thresholds)
    checks = {
        "balanced_energy_retained": float(balanced_energy_retained) >= th["balanced_energy_retained"],
        "rg_error_cached": float(rg_error_cached) <= th["rg_error_cached"],
        "rg_error_fresh": float(rg_error_fresh) <= th["rg_error_fresh"],
        "closure_residual_rms": float(closure_residual_rms) <= th["closure_residual_rms"],
        "endpoint_error": float(endpoint_error) <= th["endpoint_error"],
    }
    if prompt_delta_error is not None:
        checks["prompt_delta_error"] = float(prompt_delta_error) <= th["prompt_delta_error"]
    return {"passed": all(checks.values()), "checks": checks, "thresholds": th}


def offpolicy_error_growth(cached_errors: torch.Tensor, fresh_errors: torch.Tensor) -> dict[str, float]:
    cached = cached_errors.float()
    fresh = fresh_errors.float()
    return {
        "cached_mean": float(cached.mean().item()) if cached.numel() else 0.0,
        "fresh_mean": float(fresh.mean().item()) if fresh.numel() else 0.0,
        "growth_ratio": float((fresh.mean() / cached.mean().clamp_min(1e-12)).item()) if cached.numel() and fresh.numel() else 0.0,
    }
