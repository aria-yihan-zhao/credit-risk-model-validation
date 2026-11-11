"""Data loading, contract validation, and reproducible split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "default_next_month"
RAW_TARGET = "default payment next month"
IDENTIFIER = "ID"

EXPECTED_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


@dataclass(frozen=True)
class DataAudit:
    rows: int
    columns: int
    duplicate_ids: int
    missing_cells: int
    default_count: int
    default_rate: float


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the official UCI workbook and normalize the target name."""
    frame = pd.read_excel(path, header=1, engine="xlrd")
    return frame.rename(columns={RAW_TARGET: TARGET})


def validate_data_contract(frame: pd.DataFrame) -> DataAudit:
    """Fail loudly if the source no longer matches the expected schema."""
    expected = [IDENTIFIER, *EXPECTED_FEATURES, TARGET]
    missing_columns = sorted(set(expected) - set(frame.columns))
    unexpected_columns = sorted(set(frame.columns) - set(expected))
    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Schema mismatch. Missing={missing_columns}; unexpected={unexpected_columns}"
        )
    if len(frame) != 30_000:
        raise ValueError(f"Expected 30,000 rows, found {len(frame):,}")
    if set(frame[TARGET].dropna().unique()) != {0, 1}:
        raise ValueError("Target must be binary with values {0, 1}")

    duplicate_ids = int(frame[IDENTIFIER].duplicated().sum())
    missing_cells = int(frame.isna().sum().sum())
    if duplicate_ids:
        raise ValueError(f"Found {duplicate_ids} duplicate IDs")

    return DataAudit(
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        duplicate_ids=duplicate_ids,
        missing_cells=missing_cells,
        default_count=int(frame[TARGET].sum()),
        default_rate=float(frame[TARGET].mean()),
    )


def make_locked_split(
    frame: pd.DataFrame, *, test_size: float = 0.20, random_state: int = 5420
) -> pd.DataFrame:
    """Create an ID-only assignment table for development and final test sets."""
    development_ids, test_ids = train_test_split(
        frame[IDENTIFIER],
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET],
    )
    assignments = pd.DataFrame(
        {
            IDENTIFIER: frame[IDENTIFIER],
            "split": "development",
        }
    )
    assignments.loc[assignments[IDENTIFIER].isin(test_ids), "split"] = "test_locked"
    assert len(development_ids) + len(test_ids) == len(frame)
    return assignments.sort_values(IDENTIFIER).reset_index(drop=True)
