"""Two-proportion z-test for conversion-style (0/1) metrics."""

from __future__ import annotations

import math

from scipy import stats

from abkit.result import TestResult


def two_proportion_ztest(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> TestResult:
    """Compare two conversion rates.

    The test statistic uses the *pooled* standard error, which is the correct
    variance under H0 (both arms share one rate). The confidence interval uses
    the *unpooled* (Wald) standard error, because away from H0 the arms have
    different rates. The two can disagree right at the significance boundary;
    this is the same convention as most experimentation platforms.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("sample sizes must be positive")
    if not (0 <= successes_a <= n_a and 0 <= successes_b <= n_b):
        raise ValueError("successes must be between 0 and n")

    p_a = successes_a / n_a
    p_b = successes_b / n_b
    diff = p_b - p_a

    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se_pool == 0:
        raise ValueError("all observations are 0 or all are 1; the test is undefined")
    z = diff / se_pool
    p_value = float(2 * stats.norm.sf(abs(z)))

    se_unpooled = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return TestResult(
        method="two_proportion_z",
        control_value=p_a,
        treatment_value=p_b,
        estimate=diff,
        std_error=se_unpooled,
        statistic=z,
        p_value=p_value,
        ci_low=diff - crit * se_unpooled,
        ci_high=diff + crit * se_unpooled,
        alpha=alpha,
    )
