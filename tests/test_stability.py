"""Tests for PSI calculation behavior."""

from __future__ import annotations

import numpy as np

from credit_risk.stability import population_stability_index


def test_identical_distributions_have_zero_psi() -> None:
    values = np.linspace(0.01, 0.99, 1_000)
    psi, details = population_stability_index(values, values)
    assert abs(psi) < 1e-12
    assert abs(details["psi_contribution"].sum()) < 1e-12


def test_shifted_distribution_has_positive_psi() -> None:
    reference = np.linspace(0.01, 0.60, 1_000)
    comparison = np.linspace(0.20, 0.90, 1_000)
    psi, _ = population_stability_index(reference, comparison)
    assert psi > 0.10
