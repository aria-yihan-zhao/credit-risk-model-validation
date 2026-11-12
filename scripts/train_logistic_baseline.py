"""Train and evaluate the interpretable baseline using development data only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.data import IDENTIFIER, TARGET, load_raw_data, validate_data_contract  # noqa: E402
from credit_risk.evaluation import probability_metrics  # noqa: E402
from credit_risk.modeling import (  # noqa: E402
    build_logistic_pipeline,
    split_features_target,
)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "tables" / "logistic_cv_metrics.json"
FOLDS_PATH = PROJECT_ROOT / "reports" / "tables" / "logistic_cv_fold_metrics.csv"
COEFFICIENTS_PATH = PROJECT_ROOT / "reports" / "tables" / "logistic_coefficients.csv"
OOF_PATH = PROJECT_ROOT / "data" / "processed" / "logistic_oof_predictions.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "logistic_baseline_validation.png"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_baseline.joblib"


def main() -> None:
    frame = load_raw_data(RAW_PATH)
    validate_data_contract(frame)
    assignments = pd.read_csv(SPLIT_PATH)
    merged = frame.merge(assignments, on=IDENTIFIER, validate="one_to_one")
    development = merged.loc[merged["split"] == "development"].copy()
    if len(development) != 24_000:
        raise RuntimeError(f"Expected 24,000 development rows, found {len(development):,}")

    x, y = split_features_target(development)
    y_array = y.to_numpy()
    oof_probability = np.full(len(development), np.nan, dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=5420)

    for fold, (train_index, validation_index) in enumerate(
        cross_validator.split(x, y), start=1
    ):
        candidate = build_logistic_pipeline()
        candidate.fit(x.iloc[train_index], y.iloc[train_index])
        fold_probability = candidate.predict_proba(x.iloc[validation_index])[:, 1]
        oof_probability[validation_index] = fold_probability
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(len(validation_index)),
                "default_rate": float(y.iloc[validation_index].mean()),
                **probability_metrics(y_array[validation_index], fold_probability),
            }
        )

    if np.isnan(oof_probability).any():
        raise RuntimeError("Some development rows did not receive an OOF prediction")

    overall_metrics = {
        "evaluation_sample": "development_out_of_fold_only",
        "rows": int(len(development)),
        "default_rate": float(y.mean()),
        "cross_validation": "5-fold stratified, shuffled, random_state=5420",
        "final_test_accessed": False,
        **probability_metrics(y_array, oof_probability),
    }

    final_model = build_logistic_pipeline()
    final_model.fit(x, y)
    feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = final_model.named_steps["model"].coef_[0]
    coefficient_table = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "odds_ratio": np.exp(coefficients),
            }
        )
        .assign(abs_coefficient=lambda table: table["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(overall_metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(FOLDS_PATH, index=False)
    coefficient_table.to_csv(COEFFICIENTS_PATH, index=False)
    pd.DataFrame(
        {
            IDENTIFIER: development[IDENTIFIER].to_numpy(),
            TARGET: y_array,
            "logistic_oof_probability": oof_probability,
        }
    ).sort_values(IDENTIFIER).to_csv(OOF_PATH, index=False)
    joblib.dump(final_model, MODEL_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    RocCurveDisplay.from_predictions(y_array, oof_probability, ax=axes[0], name="Logistic")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC curve (development OOF)")
    PrecisionRecallDisplay.from_predictions(
        y_array, oof_probability, ax=axes[1], name="Logistic"
    )
    axes[1].axhline(y.mean(), linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("Precision-recall curve (development OOF)")
    CalibrationDisplay.from_predictions(
        y_array,
        oof_probability,
        n_bins=10,
        strategy="quantile",
        ax=axes[2],
        name="Logistic",
    )
    axes[2].set_title("Calibration (development OOF)")
    fig.suptitle("Logistic-regression baseline validation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(overall_metrics, indent=2))
    print(f"\nSaved metrics: {METRICS_PATH}")
    print(f"Saved validation figure: {FIGURE_PATH}")
    print("The locked final test set was not evaluated.")


if __name__ == "__main__":
    main()
