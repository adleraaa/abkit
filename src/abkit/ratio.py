"""Ratio metrics (e.g. revenue per session) with the delta method.

When users are randomized but the metric is defined per session, sessions from
the same user are correlated. Treating each session as an independent
observation underestimates the variance. The fix is to aggregate to the
randomization unit (user): each user contributes a numerator ``y_i`` (revenue)
and a denominator ``x_i`` (sessions), and the metric is ``sum(y) / sum(x)``.
Its variance follows from a first-order Taylor expansion (the delta method):

    Var(Y/X) ~= (Var(y) - 2 R Cov(y, x) + R^2 Var(x)) / (n * mean(x)^2),  R = mean(y)/mean(x)

Reference: Deng, Knoblich, Lu (2018), "Applying the Delta Method in Metric
Analytics: A Practical Guide with Novel Ideas", KDD.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from abkit.result import TestResult


def ratio_estimate(numerator: ArrayLike, denominator: ArrayLike) -> tuple[float, float]:
    """Return ``(ratio, variance of the ratio estimate)`` for one arm.

    ``numerator`` and ``denominator`` are per-user totals, aligned by user.
    """
    y = np.asarray(numerator, dtype=float)
    x = np.asarray(denominator, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("numerator and denominator must be 1-D arrays of equal length")
    n = y.size
    if n < 2:
        raise ValueError("need at least 2 users per arm")
    mean_x = float(x.mean())
    if mean_x == 0:
        raise ValueError("denominator sums to zero")
    mean_y = float(y.mean())
    ratio = mean_y / mean_x

    cov = np.cov(y, x, ddof=1)
    var_y, var_x, cov_yx = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    var_ratio = (var_y - 2 * ratio * cov_yx + ratio**2 * var_x) / (n * mean_x**2)
    # The expression is a variance of y - R x, so it cannot be negative except
    # through floating-point cancellation.
    return ratio, max(var_ratio, 0.0)


def delta_ratio_test(
    numerator_a: ArrayLike,
    denominator_a: ArrayLike,
    numerator_b: ArrayLike,
    denominator_b: ArrayLike,
    alpha: float = 0.05,
) -> TestResult:
    """z-test and CI for the difference of two ratio metrics (B - A)."""
    r_a, v_a = ratio_estimate(numerator_a, denominator_a)
    r_b, v_b = ratio_estimate(numerator_b, denominator_b)
    se = math.sqrt(v_a + v_b)
    if se == 0:
        raise ValueError("ratio metric has zero variance in both arms")
    diff = r_b - r_a
    z = diff / se
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return TestResult(
        method="delta_ratio_z",
        control_value=r_a,
        treatment_value=r_b,
        estimate=diff,
        std_error=se,
        statistic=z,
        p_value=float(2 * stats.norm.sf(abs(z))),
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        alpha=alpha,
    )
