"""Tests for probability calibration and threshold policy utilities."""

from __future__ import annotations

import numpy as np

from credit_risk.calibration import cross_fitted_calibration
from credit_risk.decision import theoretical_cost_threshold, threshold_metrics


def test_cross_fitted_platt_probabilities_are_valid() -> None:
    y_true = np.array([0, 1] * 50)
    probability = np.linspace(0.05, 0.95, len(y_true))
    calibrated = cross_fitted_calibration(
        "platt", probability, y_true, n_splits=5, random_state=42
    )
    assert calibrated.shape == probability.shape
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()


def test_cost_threshold_and_counts() -> None:
    assert theoretical_cost_threshold(5.0) == 1.0 / 6.0
    result = threshold_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.2, 0.9]),
        threshold=0.5,
        false_negative_cost_ratio=5.0,
    )
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_negative"] == 1
