"""Population Stability Index utilities for score-distribution monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> tuple[float, pd.DataFrame]:
    """Compute PSI using quantile bins derived only from the reference sample."""
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    if reference.size == 0 or comparison.size == 0:
        raise ValueError("PSI inputs must be non-empty")

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        raise ValueError("Reference distribution has too few distinct values for PSI")
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    comparison_counts, _ = np.histogram(comparison, bins=edges)
    reference_share = np.maximum(reference_counts / reference.size, epsilon)
    comparison_share = np.maximum(comparison_counts / comparison.size, epsilon)
    contribution = (comparison_share - reference_share) * np.log(
        comparison_share / reference_share
    )
    details = pd.DataFrame(
        {
            "bin_lower": edges[:-1],
            "bin_upper": edges[1:],
            "reference_count": reference_counts,
            "comparison_count": comparison_counts,
            "reference_share": reference_share,
            "comparison_share": comparison_share,
            "psi_contribution": contribution,
        }
    )
    return float(contribution.sum()), details


def descriptive_psi_band(value: float) -> str:
    """Return a transparent heuristic band, not a regulatory conclusion."""
    if value < 0.10:
        return "low shift"
    if value < 0.25:
        return "moderate shift"
    return "high shift"
