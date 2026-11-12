"""Cross-fitted probability calibration utilities."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    """Convert probabilities to finite logits for Platt scaling."""
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_calibrator(method: str, probability: np.ndarray, y_true: np.ndarray):
    """Fit a one-dimensional Platt or isotonic calibrator."""
    if method == "platt":
        calibrator = LogisticRegression(C=1e6, max_iter=1_000, solver="lbfgs")
        calibrator.fit(probability_to_logit(probability), y_true)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probability, y_true)
        return calibrator
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator, probability: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to probability estimates."""
    if method == "platt":
        return calibrator.predict_proba(probability_to_logit(probability))[:, 1]
    if method == "isotonic":
        return calibrator.predict(probability)
    raise ValueError(f"Unknown calibration method: {method}")


def cross_fitted_calibration(
    method: str,
    probability: np.ndarray,
    y_true: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 7301,
) -> np.ndarray:
    """Calibrate OOF scores without fitting and evaluating a map on the same row."""
    probability = np.asarray(probability, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    calibrated = np.full_like(probability, np.nan, dtype=float)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for train_index, validation_index in splitter.split(probability, y_true):
        calibrator = fit_calibrator(
            method, probability[train_index], y_true[train_index]
        )
        calibrated[validation_index] = apply_calibrator(
            method, calibrator, probability[validation_index]
        )
    if np.isnan(calibrated).any():
        raise RuntimeError("Cross-fitted calibration left missing predictions")
    return np.clip(calibrated, 0.0, 1.0)
