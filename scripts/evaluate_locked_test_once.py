"""Perform the single governed evaluation of the locked final test sample."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.calibration import apply_calibrator  # noqa: E402
from credit_risk.data import IDENTIFIER, TARGET, load_raw_data, validate_data_contract  # noqa: E402
from credit_risk.decision import threshold_metrics  # noqa: E402
from credit_risk.evaluation import probability_metrics  # noqa: E402
from credit_risk.modeling import split_features_target  # noqa: E402
from credit_risk.stability import population_stability_index  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_challenger.joblib"
CALIBRATOR_PATH = PROJECT_ROOT / "models" / "selected_calibrator.joblib"
DEVELOPMENT_OOF_PATH = (
    PROJECT_ROOT / "data" / "processed" / "xgboost_calibrated_oof_predictions.csv"
)
TEST_PREDICTION_PATH = PROJECT_ROOT / "data" / "processed" / "final_test_predictions.csv"
RESULT_PATH = PROJECT_ROOT / "reports" / "tables" / "final_test_evaluation.json"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "final_test_summary.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "final_test_validation.png"
PRIMARY_THRESHOLD = 1.0 / 6.0
FALSE_NEGATIVE_COST_RATIO = 5.0


def bootstrap_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    repetitions: int = 1_000,
    random_state: int = 5420,
) -> dict[str, dict[str, float]]:
    """Return percentile intervals for final-test probability metrics."""
    generator = np.random.default_rng(random_state)
    rows: list[dict[str, float]] = []
    for _ in range(repetitions):
        sample = generator.integers(0, len(y_true), size=len(y_true))
        if np.unique(y_true[sample]).size < 2:
            continue
        rows.append(probability_metrics(y_true[sample], probability[sample]))
    result: dict[str, dict[str, float]] = {}
    for metric in ["roc_auc", "pr_auc", "brier_score", "log_loss"]:
        values = np.array([row[metric] for row in rows])
        result[metric] = {
            "ci_2_5": float(np.quantile(values, 0.025)),
            "ci_97_5": float(np.quantile(values, 0.975)),
        }
    return result


def main() -> None:
    if RESULT_PATH.exists():
        raise RuntimeError(
            "Final test evaluation already exists. Governance rule: do not rerun or retune."
        )

    frame = load_raw_data(RAW_PATH)
    validate_data_contract(frame)
    assignments = pd.read_csv(SPLIT_PATH)
    test = (
        frame.merge(assignments, on=IDENTIFIER, validate="one_to_one")
        .loc[lambda table: table["split"] == "test_locked"]
        .copy()
    )
    if len(test) != 6_000:
        raise RuntimeError(f"Expected 6,000 locked test rows, found {len(test):,}")

    features, y = split_features_target(test)
    y_true = y.to_numpy()
    model = joblib.load(MODEL_PATH)
    calibration_artifact = joblib.load(CALIBRATOR_PATH)
    raw_probability = model.predict_proba(features)[:, 1]
    calibration_method = calibration_artifact["method"]
    if calibration_method == "raw":
        final_probability = raw_probability
    else:
        final_probability = apply_calibrator(
            calibration_method,
            calibration_artifact["calibrator"],
            raw_probability,
        )

    metrics = probability_metrics(y_true, final_probability)
    decisions = threshold_metrics(
        y_true,
        final_probability,
        threshold=PRIMARY_THRESHOLD,
        false_negative_cost_ratio=FALSE_NEGATIVE_COST_RATIO,
    )
    development_oof = pd.read_csv(DEVELOPMENT_OOF_PATH)
    score_psi, _ = population_stability_index(
        development_oof["selected_probability"].to_numpy(), final_probability
    )
    intervals = bootstrap_intervals(y_true, final_probability)
    evaluated_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()

    result = {
        "governance": {
            "evaluation_type": "single locked final test evaluation",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluated_commit": evaluated_commit,
            "rows": int(len(test)),
            "calibration_method": calibration_method,
            "threshold": PRIMARY_THRESHOLD,
            "false_negative_cost_ratio": FALSE_NEGATIVE_COST_RATIO,
            "retuning_after_test": False,
        },
        "probability_metrics": metrics,
        "bootstrap_95_percent_intervals": intervals,
        "decision_metrics": decisions,
        "development_to_test_score_psi": score_psi,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "sample": "locked_test",
                "rows": len(test),
                "default_rate": float(y.mean()),
                **metrics,
                "threshold": PRIMARY_THRESHOLD,
                "flag_rate": decisions["flag_rate"],
                "precision": decisions["precision"],
                "recall": decisions["recall"],
                "specificity": decisions["specificity"],
                "score_psi_vs_development": score_psi,
            }
        ]
    ).to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(
        {
            IDENTIFIER: test[IDENTIFIER].to_numpy(),
            TARGET: y_true,
            "final_probability": final_probability,
            "enhanced_review_flag": final_probability >= PRIMARY_THRESHOLD,
        }
    ).sort_values(IDENTIFIER).to_csv(TEST_PREDICTION_PATH, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    RocCurveDisplay.from_predictions(y_true, final_probability, ax=axes[0, 0], name="XGBoost")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0, 0].set_title("ROC curve - locked test")
    PrecisionRecallDisplay.from_predictions(
        y_true, final_probability, ax=axes[0, 1], name="XGBoost"
    )
    axes[0, 1].axhline(y.mean(), linestyle="--", color="gray", linewidth=1)
    axes[0, 1].set_title("Precision-recall curve - locked test")
    CalibrationDisplay.from_predictions(
        y_true,
        final_probability,
        n_bins=10,
        strategy="quantile",
        ax=axes[1, 0],
        name="XGBoost",
    )
    axes[1, 0].set_title("Calibration - locked test")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        final_probability >= PRIMARY_THRESHOLD,
        ax=axes[1, 1],
        colorbar=False,
    )
    axes[1, 1].set_title("Enhanced-review policy at threshold 0.167")
    fig.suptitle("Single locked final-test evaluation", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(result, indent=2))
    print("\nFinal test has now been accessed once. Do not retune or rerun.")


if __name__ == "__main__":
    main()
