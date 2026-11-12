"""Cross-fit calibration and define transparent cost-sensitive review thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.calibration import (  # noqa: E402
    cross_fitted_calibration,
    fit_calibrator,
)
from credit_risk.data import IDENTIFIER, TARGET  # noqa: E402
from credit_risk.decision import (  # noqa: E402
    theoretical_cost_threshold,
    threshold_metrics,
)
from credit_risk.evaluation import probability_metrics  # noqa: E402

XGB_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "xgboost_oof_predictions.csv"
CALIBRATED_OOF_PATH = (
    PROJECT_ROOT / "data" / "processed" / "xgboost_calibrated_oof_predictions.csv"
)
CALIBRATION_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "calibration_comparison.csv"
)
THRESHOLD_PATH = PROJECT_ROOT / "reports" / "tables" / "threshold_policy.csv"
POLICY_PATH = PROJECT_ROOT / "reports" / "tables" / "decision_policy.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "calibration_and_cost_policy.png"
CALIBRATOR_PATH = PROJECT_ROOT / "models" / "selected_calibrator.joblib"


def main() -> None:
    if not XGB_OOF_PATH.exists():
        raise FileNotFoundError("Run `python scripts/train_xgboost_challenger.py` first.")

    oof = pd.read_csv(XGB_OOF_PATH).sort_values(IDENTIFIER).reset_index(drop=True)
    y_true = oof[TARGET].to_numpy(dtype=int)
    raw_probability = oof["xgboost_oof_probability"].to_numpy(dtype=float)
    probability_by_method = {
        "raw": raw_probability,
        "platt": cross_fitted_calibration("platt", raw_probability, y_true),
        "isotonic": cross_fitted_calibration("isotonic", raw_probability, y_true),
    }
    comparison = pd.DataFrame(
        [
            {"method": method, **probability_metrics(y_true, probability)}
            for method, probability in probability_by_method.items()
        ]
    ).sort_values(["brier_score", "log_loss"], ascending=True)
    selected_method = str(comparison.iloc[0]["method"])
    selected_probability = probability_by_method[selected_method]

    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    if selected_method == "raw":
        selected_calibrator = None
    else:
        selected_calibrator = fit_calibrator(
            selected_method, raw_probability, y_true
        )
    joblib.dump(
        {"method": selected_method, "calibrator": selected_calibrator},
        CALIBRATOR_PATH,
    )

    calibrated_oof = pd.DataFrame(
        {
            IDENTIFIER: oof[IDENTIFIER],
            TARGET: y_true,
            "raw_xgboost_probability": raw_probability,
            "selected_method": selected_method,
            "selected_probability": selected_probability,
        }
    )
    calibrated_oof.to_csv(CALIBRATED_OOF_PATH, index=False)

    cost_ratios = [2.0, 5.0, 10.0]
    policy_rows = []
    for ratio in cost_ratios:
        threshold = theoretical_cost_threshold(ratio)
        policy_rows.append(
            threshold_metrics(
                y_true,
                selected_probability,
                threshold=threshold,
                false_negative_cost_ratio=ratio,
            )
        )
    policy_table = pd.DataFrame(policy_rows)

    policy = {
        "evaluation_sample": "development_cross_fitted_oof_only",
        "selected_calibration_method": selected_method,
        "selection_rule": "lowest cross-fitted Brier score; log loss used as tie-breaker",
        "primary_cost_scenario": {
            "false_positive_cost": 1.0,
            "false_negative_cost": 5.0,
            "interpretation": (
                "Illustrative enhanced-review policy only; not an observed bank cost. "
                "A false negative is assumed five times as costly as a false positive."
            ),
            "threshold": theoretical_cost_threshold(5.0),
        },
        "final_test_accessed": False,
    }

    CALIBRATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CALIBRATION_METRICS_PATH, index=False)
    policy_table.to_csv(THRESHOLD_PATH, index=False)
    POLICY_PATH.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method, probability in probability_by_method.items():
        CalibrationDisplay.from_predictions(
            y_true,
            probability,
            n_bins=10,
            strategy="quantile",
            ax=axes[0],
            name=method.title(),
        )
    axes[0].set_title("Cross-fitted calibration comparison")

    thresholds = np.linspace(0.01, 0.60, 150)
    for ratio in cost_ratios:
        costs = [
            threshold_metrics(
                y_true,
                selected_probability,
                threshold=float(threshold),
                false_negative_cost_ratio=ratio,
            )["normalized_cost_per_account"]
            for threshold in thresholds
        ]
        axes[1].plot(thresholds, costs, label=f"FN cost = {ratio:g}x FP")
        axes[1].axvline(
            theoretical_cost_threshold(ratio),
            linestyle="--",
            linewidth=1,
            alpha=0.55,
        )
    axes[1].set_xlabel("Probability threshold for enhanced review")
    axes[1].set_ylabel("Normalized expected cost per account")
    axes[1].set_title("Illustrative cost-sensitive threshold scenarios")
    axes[1].legend()
    fig.suptitle("Probability quality and decision policy", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Calibration comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSelected method: {selected_method}")
    print("\nCost-sensitive review scenarios:")
    print(policy_table.to_string(index=False))
    print("\nThe locked final test set was not evaluated.")


if __name__ == "__main__":
    main()
