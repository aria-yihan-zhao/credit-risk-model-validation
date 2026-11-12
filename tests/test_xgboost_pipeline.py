"""Smoke test for the XGBoost challenger pipeline."""

from __future__ import annotations

import pandas as pd

from credit_risk.data import EXPECTED_FEATURES
from credit_risk.modeling import build_xgboost_pipeline


def test_xgboost_pipeline_returns_probabilities() -> None:
    rows = 20
    frame = pd.DataFrame({feature: range(rows) for feature in EXPECTED_FEATURES})
    frame["SEX"] = [1, 2] * 10
    frame["EDUCATION"] = [1, 2, 3, 4] * 5
    frame["MARRIAGE"] = [1, 2] * 10
    target = pd.Series([0, 1] * 10)

    pipeline = build_xgboost_pipeline()
    pipeline.set_params(model__n_estimators=2)
    pipeline.fit(frame, target)
    probability = pipeline.predict_proba(frame)[:, 1]

    assert probability.shape == (rows,)
    assert ((probability >= 0.0) & (probability <= 1.0)).all()
