"""Validate the raw source and create a locked split manifest."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.data import (  # noqa: E402
    IDENTIFIER,
    TARGET,
    load_raw_data,
    make_locked_split,
    validate_data_contract,
)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"
AUDIT_PATH = PROJECT_ROOT / "reports" / "tables" / "data_audit.json"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError("Run `python scripts/download_data.py` first.")

    frame = load_raw_data(RAW_PATH)
    audit = validate_data_contract(frame)
    assignments = make_locked_split(frame)

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(SPLIT_PATH, index=False)

    split_summary = (
        frame[[IDENTIFIER, TARGET]]
        .merge(assignments, on=IDENTIFIER, validate="one_to_one")
        .groupby("split")[TARGET]
        .agg(rows="size", defaults="sum", default_rate="mean")
        .reset_index()
        .to_dict(orient="records")
    )
    report = {**asdict(audit), "split_summary": split_summary}
    AUDIT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved split manifest: {SPLIT_PATH}")
    print(f"Saved audit report: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
