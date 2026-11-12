"""Evaluation utilities focused on ranking and probability quality."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, *, bins: int = 10
) -> float:
    """Compute equal-width expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bin_index == index
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(
            float(np.mean(y_true[mask])) - float(np.mean(probability[mask]))
        )
    return error


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return discrimination and calibration metrics without choosing a threshold."""
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability)),
        "ece_10_bin": expected_calibration_error(y_true, probability, bins=10),
    }
