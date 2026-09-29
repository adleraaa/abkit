"""Sample ratio mismatch (SRM) check.

If users were meant to be split 50/50 but the observed counts deviate more than
chance allows, the assignment or logging pipeline is broken and the metric
comparisons cannot be trusted. A chi-square goodness-of-fit test flags this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    observed: list[int]
    expected: list[float]
    statistic: float
    p_value: float
    threshold: float

    @property
    def mismatch(self) -> bool:
        return self.p_value < self.threshold


def srm_check(
    counts: Sequence[int],
    expected_ratios: Sequence[float] | None = None,
    threshold: float = 0.001,
) -> SRMResult:
    """Chi-square test of observed arm sizes against the intended split.

    The default threshold is 0.001 rather than 0.05: the check runs on every
    experiment, and a strict threshold keeps false alarms rare while real
    assignment bugs with large samples still produce tiny p-values.
    """
    observed = np.asarray(counts, dtype=float)
    if observed.ndim != 1 or observed.size < 2:
        raise ValueError("need counts for at least 2 arms")
    if np.any(observed < 0):
        raise ValueError("counts must be non-negative")
    total = observed.sum()
    if total == 0:
        raise ValueError("no users in any arm")

    if expected_ratios is None:
        ratios = np.full(observed.size, 1 / observed.size)
    else:
        ratios = np.asarray(expected_ratios, dtype=float)
        if ratios.shape != observed.shape or np.any(ratios <= 0):
            raise ValueError("expected_ratios must be positive, one per arm")
        ratios = ratios / ratios.sum()

    expected = ratios * total
    statistic, p_value = stats.chisquare(observed, expected)
    return SRMResult(
        observed=[int(c) for c in observed],
        expected=[float(e) for e in expected],
        statistic=float(statistic),
        p_value=float(p_value),
        threshold=threshold,
    )
