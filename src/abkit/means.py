"""Welch's two-sample t-test for a difference in means."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from abkit.result import TestResult


def welch_from_stats(
    mean_a: float,
    var_a: float,
    n_a: int,
    mean_b: float,
    var_b: float,
    n_b: int,
    alpha: float = 0.05,
    method: str = "welch_t",
) -> TestResult:
    """Welch t-test from summary statistics (sample variances use ddof=1).

    Working from summary statistics lets the sequential simulation re-test
    cumulative data at every look without re-scanning the raw arrays.
    """
    if n_a < 2 or n_b < 2:
        raise ValueError("each arm needs at least 2 observations")
    se2_a = var_a / n_a
    se2_b = var_b / n_b
    se = math.sqrt(se2_a + se2_b)
    if se == 0:
        raise ValueError("both arms have zero variance; the test is undefined")

    # Welch-Satterthwaite degrees of freedom: does not assume equal variances,
    # which rarely hold between control and treatment in practice.
    df = (se2_a + se2_b) ** 2 / (se2_a**2 / (n_a - 1) + se2_b**2 / (n_b - 1))
    diff = mean_b - mean_a
    t_stat = diff / se
    p_value = float(2 * stats.t.sf(abs(t_stat), df))
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return TestResult(
        method=method,
        control_value=mean_a,
        treatment_value=mean_b,
        estimate=diff,
        std_error=se,
        statistic=t_stat,
        p_value=p_value,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        alpha=alpha,
    )


def welch_ttest(control: ArrayLike, treatment: ArrayLike, alpha: float = 0.05) -> TestResult:
    """Welch t-test and (1 - alpha) CI for ``mean(treatment) - mean(control)``."""
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    return welch_from_stats(
        float(a.mean()),
        float(a.var(ddof=1)),
        a.size,
        float(b.mean()),
        float(b.var(ddof=1)),
        b.size,
        alpha=alpha,
    )
