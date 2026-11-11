"""Fast tests for schema and split behavior using synthetic records."""

from __future__ import annotations

import pandas as pd

from credit_risk.data import IDENTIFIER, TARGET, make_locked_split


def test_locked_split_is_reproducible_and_disjoint() -> None:
    frame = pd.DataFrame(
        {
            IDENTIFIER: range(1, 101),
            TARGET: [0] * 80 + [1] * 20,
        }
    )
    first = make_locked_split(frame, random_state=42)
    second = make_locked_split(frame, random_state=42)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["split"]) == {"development", "test_locked"}
    assert first[IDENTIFIER].is_unique
    assert len(first) == len(frame)
    assert (first["split"] == "test_locked").sum() == 20
