"""Multiple-comparison corrections returning adjusted p-values.

Adjusted p-values can be compared directly to the original alpha, which is
easier to report than per-hypothesis thresholds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate(p_values: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be 1-D")
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("p_values must lie in [0, 1]")
    return p


def holm(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm step-down adjustment; controls the family-wise error rate.

    Uniformly more powerful than Bonferroni with the same guarantee.
    """
    p = _validate(p_values)
    m = p.size
    order = np.argsort(p)
    # The k-th smallest p-value (0-based) is multiplied by (m - k).
    scaled = p[order] * (m - np.arange(m))
    # Step-down: an adjusted p-value can never be smaller than the one before it.
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg step-up adjustment; controls the false discovery rate."""
    p = _validate(p_values)
    m = p.size
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    # Step-up: walk from the largest p-value down, carrying the running minimum.
    adjusted_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted
