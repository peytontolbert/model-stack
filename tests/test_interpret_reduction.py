from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch.nn")

from interpret import (
    balanced_energy_retained,
    balanced_hankel_singular_values,
    balanced_projection,
    block_grouping_dynamic_program,
    closure_residual,
    closure_residual_metrics,
    empirical_controllability_gramian,
    empirical_observability_gramian,
    offpolicy_error_growth,
    project_onto_basis,
    reconstruct_from_basis,
    reduced_order_certification,
    rg_commutation_error,
)


def test_balanced_projection_and_energy() -> None:
    torch.manual_seed(0)
    deltas = torch.randn(32, 6)
    jac = torch.randn(16, 6)
    c = empirical_controllability_gramian(deltas)
    o = empirical_observability_gramian(jac)
    hsv = balanced_hankel_singular_values(c, o)
    basis, kept = balanced_projection(c, o, rank=3)
    assert hsv.shape == (6,)
    assert basis.shape == (6, 3)
    assert kept.shape == (3,)
    assert 0.0 <= balanced_energy_retained(hsv, 3).item() <= 1.0
    coords = project_onto_basis(deltas, basis)
    recon = reconstruct_from_basis(coords, basis)
    assert coords.shape[-1] == 3
    assert recon.shape == deltas.shape


def test_closure_commutation_and_certification() -> None:
    teacher = torch.randn(4, 6)
    projection = torch.randn(6, 3)
    student = torch.randn(4, 3)
    residual = closure_residual(teacher, student, projection=projection)
    assert residual.shape == student.shape
    assert closure_residual_metrics(residual)["rms_mean"] >= 0.0
    assert rg_commutation_error(teacher, student, projection_after=projection).item() >= 0.0
    cert = reduced_order_certification(
        balanced_energy_retained=0.99,
        rg_error_cached=0.01,
        rg_error_fresh=0.02,
        closure_residual_rms=0.01,
        endpoint_error=0.01,
    )
    assert cert["passed"] is True
    growth = offpolicy_error_growth(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
    assert growth["growth_ratio"] > 1.0


def test_block_grouping_dynamic_program() -> None:
    cost = torch.full((4, 4), float("inf"))
    for i in range(4):
        for j in range(i, 4):
            cost[i, j] = float(j - i + 1)
    total, groups = block_grouping_dynamic_program(cost, 2)
    assert total == 4.0
    assert groups[0][0] == 0
    assert groups[-1][1] == 4
