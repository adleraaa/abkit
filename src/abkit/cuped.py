"""CUPED: variance reduction with a pre-experiment covariate.

Each user's metric ``y`` is replaced by ``y - theta * (x - mean(x))`` where ``x``
is the same user's pre-period value. Because ``x`` was measured before
randomization it is independent of treatment, so the adjustment does not bias
the difference in means; it only removes the part of ``y`` that ``x`` predicts.
With ``theta = Cov(y, x) / Var(x)`` the variance shrinks by a factor of
``1 - corr(y, x)^2``.

Reference: Deng, Xu, Kohavi, Walker (2013), "Improving the Sensitivity of
Online Controlled Experiments by Utilizing Pre-Experiment Data", WSDM.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from abkit.means import welch_ttest
from abkit.result import TestResult


def cuped_theta(y: ArrayLike, x: ArrayLike) -> float:
    """OLS slope of ``y`` on ``x``: the variance-minimizing adjustment coefficient."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    var_x = float(x_arr.var(ddof=1))
    if var_x == 0:
        raise ValueError("covariate is constant, so it cannot explain any variance")
    return float(np.cov(y_arr, x_arr, ddof=1)[0, 1]) / var_x


def cuped_test(
    y_a: ArrayLike,
    x_a: ArrayLike,
    y_b: ArrayLike,
    x_b: ArrayLike,
    alpha: float = 0.05,
) -> TestResult:
    """Welch t-test on CUPED-adjusted metrics (B - A)."""
    ya, xa = np.asarray(y_a, dtype=float), np.asarray(x_a, dtype=float)
    yb, xb = np.asarray(y_b, dtype=float), np.asarray(x_b, dtype=float)
    if ya.shape != xa.shape or yb.shape != xb.shape:
        raise ValueError("metric and covariate arrays must be aligned per arm")

    # theta and the centering mean are estimated on both arms pooled, as in
    # Deng et al. (2013). This is the simple standard choice and is
    # asymptotically equivalent to ANCOVA with a common slope. Per-arm slopes
    # around the pooled mean (Lin 2013) are a valid alternative that can help when
    # treatment changes the slope; they are not implemented here.
    y_all = np.concatenate([ya, yb])
    x_all = np.concatenate([xa, xb])
    theta = cuped_theta(y_all, x_all)
    x_mean = float(x_all.mean())

    def adjust(y: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
        return y - theta * (x - x_mean)

    # Treating theta as known ignores one estimated parameter; with thousands of
    # users per arm this is negligible (see the A/A results in the README).
    res = welch_ttest(adjust(ya, xa), adjust(yb, xb), alpha=alpha)
    return TestResult(**{**res.__dict__, "method": "cuped_welch_t"})
