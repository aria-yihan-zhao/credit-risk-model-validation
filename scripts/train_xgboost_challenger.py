"""Train XGBoost and compare it fairly with the logistic baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.data import IDENTIFIER, TARGET, load_raw_data, validate_data_contract  # noqa: E402
from credit_risk.evaluation import probability_metrics  # noqa: E402
from credit_risk.modeling import build_xgboost_pipeline, split_features_target  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"
LOGISTIC_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "logistic_oof_predictions.csv"
XGB_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "xgboost_oof_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "tables" / "xgboost_cv_metrics.json"
FOLDS_PATH = PROJECT_ROOT / "reports" / "tables" / "xgboost_cv_fold_metrics.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "tables" / "xgboost_feature_importance.csv"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "tables" / "model_comparison.csv"
BOOTSTRAP_PATH = PROJECT_ROOT / "reports" / "tables" / "paired_bootstrap_comparison.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "model_comparison_validation.png"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_challenger.joblib"


def paired_bootstrap(
    y_true: np.ndarray,
    logistic_probability: np.ndarray,
    xgb_probability: np.ndarray,
    *,
    repetitions: int = 500,
    random_state: int = 5420,
) -> dict[str, dict[str, float]]:
    """Estimate paired confidence intervals for metric improvements."""
    generator = np.random.default_rng(random_state)
    deltas = {"roc_auc_gain": [], "pr_auc_gain": [], "brier_reduction": []}
    n_rows = len(y_true)
    for _ in range(repetitions):
        sample = generator.integers(0, n_rows, size=n_rows)
        sample_y = y_true[sample]
        if np.unique(sample_y).size < 2:
            continue
        logistic_metrics = probability_metrics(sample_y, logistic_probability[sample])
        xgb_metrics = probability_metrics(sample_y, xgb_probability[sample])
        deltas["roc_auc_gain"].append(
            xgb_metrics["roc_auc"] - logistic_metrics["roc_auc"]
        )
        deltas["pr_auc_gain"].append(
            xgb_metrics["pr_auc"] - logistic_metrics["pr_auc"]
        )
        deltas["brier_reduction"].append(
            logistic_metrics["brier_score"] - xgb_metrics["brier_score"]
        )

    result: dict[str, dict[str, float]] = {}
    for metric, values in deltas.items():
        array = np.asarray(values)
        result[metric] = {
            "mean": float(np.mean(array)),
            "ci_2_5": float(np.quantile(array, 0.025)),
            "ci_97_5": float(np.quantile(array, 0.975)),
        }
    return result


def main() -> None:
    if not LOGISTIC_OOF_PATH.exists():
        raise FileNotFoundError("Run `python scripts/train_logistic_baseline.py` first.")

    frame = load_raw_data(RAW_PATH)
    validate_data_contract(frame)
    assignments = pd.read_csv(SPLIT_PATH)
    development = (
        frame.merge(assignments, on=IDENTIFIER, validate="one_to_one")
        .loc[lambda table: table["split"] == "development"]
        .copy()
    )
    x, y = split_features_target(development)
    y_array = y.to_numpy()
    oof_probability = np.full(len(development), np.nan, dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=5420)

    for fold, (train_index, validation_index) in enumerate(
        cross_validator.split(x, y), start=1
    ):
        challenger = build_xgboost_pipeline()
        challenger.fit(x.iloc[train_index], y.iloc[train_index])
        fold_probability = challenger.predict_proba(x.iloc[validation_index])[:, 1]
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

    final_model = build_xgboost_pipeline()
    final_model.fit(x, y)
    feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": final_model.named_steps["model"].feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    xgb_oof = pd.DataFrame(
        {
            IDENTIFIER: development[IDENTIFIER].to_numpy(),
            TARGET: y_array,
            "xgboost_oof_probability": oof_probability,
        }
    ).sort_values(IDENTIFIER)
    logistic_oof = pd.read_csv(LOGISTIC_OOF_PATH)
    paired = logistic_oof.merge(xgb_oof, on=[IDENTIFIER, TARGET], validate="one_to_one")
    logistic_probability = paired["logistic_oof_probability"].to_numpy()
    xgb_paired_probability = paired["xgboost_oof_probability"].to_numpy()
    paired_y = paired[TARGET].to_numpy()
    logistic_metrics = probability_metrics(paired_y, logistic_probability)
    xgb_metrics = probability_metrics(paired_y, xgb_paired_probability)
    comparison = pd.DataFrame(
        [
            {"model": "Logistic regression", **logistic_metrics},
            {"model": "XGBoost", **xgb_metrics},
        ]
    )
    bootstrap = {
        "definition": {
            "roc_auc_gain": "XGBoost minus logistic; positive favors XGBoost",
            "pr_auc_gain": "XGBoost minus logistic; positive favors XGBoost",
            "brier_reduction": "logistic minus XGBoost; positive favors XGBoost",
        },
        "repetitions": 500,
        "results": paired_bootstrap(
            paired_y, logistic_probability, xgb_paired_probability, repetitions=500
        ),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(overall_metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(FOLDS_PATH, index=False)
    feature_importance.to_csv(IMPORTANCE_PATH, index=False)
    xgb_oof.to_csv(XGB_OOF_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    BOOTSTRAP_PATH.write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
    joblib.dump(final_model, MODEL_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, probability in [
        ("Logistic", logistic_probability),
        ("XGBoost", xgb_paired_probability),
    ]:
        RocCurveDisplay.from_predictions(paired_y, probability, ax=axes[0], name=label)
        PrecisionRecallDisplay.from_predictions(paired_y, probability, ax=axes[1], name=label)
        CalibrationDisplay.from_predictions(
            paired_y,
            probability,
            n_bins=10,
            strategy="quantile",
            ax=axes[2],
            name=label,
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC curve (same development OOF rows)")
    axes[1].axhline(y.mean(), linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("Precision-recall curve")
    axes[2].set_title("Probability calibration")
    fig.suptitle("Fair challenger comparison: Logistic vs XGBoost", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(comparison.to_string(index=False))
    print("\nPaired bootstrap comparison:")
    print(json.dumps(bootstrap["results"], indent=2))
    print("\nThe locked final test set was not evaluated.")


if __name__ == "__main__":
    main()
