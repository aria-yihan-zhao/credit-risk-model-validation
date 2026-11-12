"""Transparent cost-sensitive decision policy utilities."""

from __future__ import annotations

import numpy as np


def theoretical_cost_threshold(false_negative_cost_ratio: float) -> float:
    """Return the Bayes threshold when false-positive cost is normalized to one."""
    if false_negative_cost_ratio <= 0:
        raise ValueError("False-negative cost ratio must be positive")
    return 1.0 / (1.0 + false_negative_cost_ratio)


def threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    false_negative_cost_ratio: float,
) -> dict[str, float | int]:
    """Summarize a review policy at a specified probability threshold."""
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(probability) >= threshold
    positive = y_true == 1
    negative = ~positive
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & negative))
    true_negative = int(np.sum((~predicted) & negative))
    false_negative = int(np.sum((~predicted) & positive))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    normalized_cost = (
        false_positive + false_negative_cost_ratio * false_negative
    ) / len(y_true)
    return {
        "threshold": float(threshold),
        "false_negative_cost_ratio": float(false_negative_cost_ratio),
        "flag_rate": float(predicted.mean()),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "normalized_cost_per_account": float(normalized_cost),
    }
