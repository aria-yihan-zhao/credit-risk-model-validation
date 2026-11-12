"""Audit development OOF predictions across demographic subgroups."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.data import IDENTIFIER, TARGET, load_raw_data, validate_data_contract  # noqa: E402
from credit_risk.evaluation import subgroup_metrics  # noqa: E402
from credit_risk.modeling import apply_documented_category_policy  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
XGB_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "xgboost_oof_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "tables" / "subgroup_audit.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "subgroup_audit.png"
PRIMARY_THRESHOLD = 1.0 / 6.0


def main() -> None:
    frame = apply_documented_category_policy(load_raw_data(RAW_PATH))
    validate_data_contract(frame)
    oof = pd.read_csv(XGB_OOF_PATH)
    audit = frame[
        [IDENTIFIER, "SEX", "EDUCATION", "MARRIAGE", "AGE", TARGET]
    ].merge(oof, on=[IDENTIFIER, TARGET], validate="one_to_one")
    audit["sex_group"] = (
        audit["SEX"]
        .map({1: "male (code 1)", 2: "female (code 2)"})
        .fillna("unknown")
    )
    audit["education_group"] = (
        audit["EDUCATION"]
        .map({1: "graduate school", 2: "university", 3: "high school", 4: "other/unknown"})
        .fillna("unknown")
    )
    audit["marriage_group"] = (
        audit["MARRIAGE"]
        .map({1: "married", 2: "single", 3: "other/unknown"})
        .fillna("unknown")
    )
    audit["age_group"] = pd.cut(
        audit["AGE"],
        bins=[0, 29, 39, 49, 59, float("inf")],
        labels=["under 30", "30-39", "40-49", "50-59", "60+"],
    ).astype(str)

    group_columns = [
        "sex_group",
        "education_group",
        "marriage_group",
        "age_group",
    ]
    rows: list[dict[str, str | float | int]] = []
    for group_column in group_columns:
        for group_value, group in audit.groupby(group_column, observed=True):
            if len(group) < 100 or group[TARGET].nunique() < 2:
                continue
            rows.append(
                {
                    "audit_dimension": group_column.removesuffix("_group"),
                    "group": str(group_value),
                    **subgroup_metrics(
                        group[TARGET].to_numpy(),
                        group["xgboost_oof_probability"].to_numpy(),
                        threshold=PRIMARY_THRESHOLD,
                    ),
                }
            )
    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    dimensions = ["sex", "education", "marriage", "age"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, dimension in zip(axes.ravel(), dimensions, strict=True):
        subset = result.loc[result["audit_dimension"] == dimension].copy()
        positions = range(len(subset))
        width = 0.38
        axis.bar(
            [position - width / 2 for position in positions],
            subset["recall"],
            width=width,
            label="Recall",
        )
        axis.bar(
            [position + width / 2 for position in positions],
            subset["false_positive_rate"],
            width=width,
            label="False-positive rate",
        )
        axis.set_xticks(list(positions), subset["group"], rotation=25, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(dimension.title())
        axis.set_ylabel("Rate at 0.167 review threshold")
        axis.legend(fontsize=8)
    fig.suptitle(
        "Descriptive subgroup audit (demographics excluded from model)",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(result.to_string(index=False))
    print("\nThis is a descriptive development-data audit, not a legal fairness certification.")
    print("The locked final test set was not evaluated.")


if __name__ == "__main__":
    main()
