"""Run PSI controls and transparent synthetic stress scenarios on development data."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.data import IDENTIFIER, TARGET, load_raw_data, validate_data_contract  # noqa: E402
from credit_risk.modeling import split_features_target  # noqa: E402
from credit_risk.stability import (  # noqa: E402
    descriptive_psi_band,
    population_stability_index,
)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_challenger.joblib"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "stability_stress_summary.csv"
DETAIL_PATH = PROJECT_ROOT / "reports" / "tables" / "stability_psi_bins.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "stability_stress_test.png"

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAYMENT_AMOUNT = [f"PAY_AMT{index}" for index in range(1, 7)]


def score(model, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python scripts/train_xgboost_challenger.py` first.")

    frame = load_raw_data(RAW_PATH)
    validate_data_contract(frame)
    assignments = pd.read_csv(SPLIT_PATH)
    development = (
        frame.merge(assignments, on=IDENTIFIER, validate="one_to_one")
        .loc[lambda table: table["split"] == "development"]
        .copy()
    )
    features, _ = split_features_target(development)
    model = joblib.load(MODEL_PATH)
    baseline_probability = score(model, features)

    delayed = features.copy()
    for column in PAY_STATUS:
        delayed[column] = np.where(
            delayed[column] >= 0,
            np.minimum(delayed[column] + 1, 8),
            delayed[column],
        )

    reduced_payment = features.copy()
    reduced_payment[PAYMENT_AMOUNT] = reduced_payment[PAYMENT_AMOUNT] * 0.70

    combined = delayed.copy()
    combined[PAYMENT_AMOUNT] = combined[PAYMENT_AMOUNT] * 0.70

    scenario_probability = {
        "repayment_delay_plus_one": score(model, delayed),
        "payment_amount_minus_30pct": score(model, reduced_payment),
        "combined_stress": score(model, combined),
    }

    generator = np.random.default_rng(5420)
    permutation = generator.permutation(len(baseline_probability))
    midpoint = len(permutation) // 2
    control_reference = baseline_probability[permutation[:midpoint]]
    control_comparison = baseline_probability[permutation[midpoint:]]

    summary_rows: list[dict[str, float | str | int]] = []
    detail_frames: list[pd.DataFrame] = []

    control_psi, control_details = population_stability_index(
        control_reference, control_comparison
    )
    summary_rows.append(
        {
            "scenario": "same_distribution_random_half_control",
            "rows": len(control_comparison),
            "reference_mean_risk": float(control_reference.mean()),
            "comparison_mean_risk": float(control_comparison.mean()),
            "psi": control_psi,
            "psi_band_heuristic": descriptive_psi_band(control_psi),
        }
    )
    control_details.insert(0, "scenario", "same_distribution_random_half_control")
    detail_frames.append(control_details)

    for scenario, probability in scenario_probability.items():
        psi, details = population_stability_index(baseline_probability, probability)
        summary_rows.append(
            {
                "scenario": scenario,
                "rows": len(probability),
                "reference_mean_risk": float(baseline_probability.mean()),
                "comparison_mean_risk": float(probability.mean()),
                "psi": psi,
                "psi_band_heuristic": descriptive_psi_band(psi),
            }
        )
        details.insert(0, "scenario", scenario)
        detail_frames.append(details)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    details.to_csv(DETAIL_PATH, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bins = np.linspace(0.0, 1.0, 41)
    axes[0].hist(
        baseline_probability,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Baseline",
    )
    for scenario, probability in scenario_probability.items():
        axes[0].hist(
            probability,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=scenario.replace("_", " "),
        )
    axes[0].set_xlabel("Predicted default probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score distribution under synthetic stress")
    axes[0].legend(fontsize=8)

    plot_summary = summary.set_index("scenario")
    axes[1].barh(
        [name.replace("_", " ") for name in plot_summary.index],
        plot_summary["psi"],
    )
    axes[1].axvline(0.10, color="orange", linestyle="--", linewidth=1, label="0.10 heuristic")
    axes[1].axvline(0.25, color="red", linestyle="--", linewidth=1, label="0.25 heuristic")
    axes[1].set_xlabel("Population Stability Index")
    axes[1].set_title("PSI control and stress scenarios")
    axes[1].legend(fontsize=8)
    fig.suptitle("Development-only stability stress test", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print("\nThese are synthetic development-data stresses, not temporal validation.")
    print("The locked final test set was not evaluated.")


if __name__ == "__main__":
    main()
