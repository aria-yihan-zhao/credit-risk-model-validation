"""Feature policy and model pipelines for credit-default prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.data import EXPECTED_FEATURES, TARGET

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_FEATURES = [
    feature for feature in EXPECTED_FEATURES if feature not in CATEGORICAL_FEATURES
]


def apply_documented_category_policy(frame: pd.DataFrame) -> pd.DataFrame:
    """Consolidate undocumented category codes instead of silently dropping rows.

    The UCI workbook contains education values 0, 5, and 6 and marital-status
    value 0, although those codes are not separately defined in the data notes.
    They are mapped to the respective 'other/unknown' bucket and retained.
    """
    clean = frame.copy()
    clean["EDUCATION"] = clean["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    clean["MARRIAGE"] = clean["MARRIAGE"].replace({0: 3})
    return clean


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target, excluding the account identifier."""
    clean = apply_documented_category_policy(frame)
    return clean[EXPECTED_FEATURES].copy(), clean[TARGET].astype(int).copy()


def build_logistic_pipeline() -> Pipeline:
    """Build an unweighted, regularized logistic-regression baseline.

    Class weights are intentionally not applied because the output is intended
    to represent default probability. Decision costs will be handled later via
    threshold selection rather than by distorting the training prevalence.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=2_000,
                    solver="lbfgs",
                    random_state=5420,
                ),
            ),
        ]
    )
