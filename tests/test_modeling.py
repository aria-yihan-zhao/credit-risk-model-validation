"""Tests for the documented feature policy."""

from __future__ import annotations

import pandas as pd

from credit_risk.modeling import apply_documented_category_policy


def test_undocumented_categories_map_to_other_bucket() -> None:
    frame = pd.DataFrame(
        {
            "EDUCATION": [0, 1, 5, 6],
            "MARRIAGE": [0, 1, 2, 3],
        }
    )
    clean = apply_documented_category_policy(frame)

    assert clean["EDUCATION"].tolist() == [4, 1, 4, 4]
    assert clean["MARRIAGE"].tolist() == [3, 1, 2, 3]
