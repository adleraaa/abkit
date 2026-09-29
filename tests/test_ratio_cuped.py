import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from abkit import cuped_test, cuped_theta, delta_ratio_test, ratio_estimate, welch_ttest
from abkit.simulate import session_data


def test_ratio_with_unit_denominators_is_a_mean() -> None:
    """With one session per user the ratio is a plain mean and Var = s^2 / n."""
    rng = np.random.default_rng(1)
    y = rng.exponential(2.0, 500)
    ratio, var = ratio_estimate(y, np.ones_like(y))
    assert ratio == pytest.approx(y.mean())
    assert var == pytest.approx(y.var(ddof=1) / y.size)


def test_delta_variance_matches_bootstrap() -> None:
    """The delta-method variance should agree with a user-level bootstrap."""
    rng = np.random.default_rng(2)
    rev, ses, _ = session_data(rng, 2000, heterogeneity=1.0)
    _, var_delta = ratio_estimate(rev, ses)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, rev.size, rev.size)
        boot.append(rev[idx].sum() / ses[idx].sum())
    assert var_delta == pytest.approx(np.var(boot, ddof=1), rel=0.1)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), scale=st.floats(0.01, 100))
def test_ratio_scales_with_numerator(seed: int, scale: float) -> None:
    rng = np.random.default_rng(seed)
    y = rng.exponential(1.0, 50)
    x = rng.integers(1, 5, 50).astype(float)
    r1, v1 = ratio_estimate(y, x)
    r2, v2 = ratio_estimate(scale * y, x)
    assert r2 == pytest.approx(scale * r1)
    assert v2 == pytest.approx(scale**2 * v1, rel=1e-9, abs=1e-12)


def test_delta_ratio_test_direction_and_validation() -> None:
    rng = np.random.default_rng(3)
    rev_a, ses_a, _ = session_data(rng, 3000, 0.5)
    rev_b, ses_b, _ = session_data(rng, 3000, 0.5, lift=0.3)
    res = delta_ratio_test(rev_a, ses_a, rev_b, ses_b)
    assert res.estimate > 0 and res.significant
    with pytest.raises(ValueError):
        ratio_estimate([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        ratio_estimate([1.0, 2.0], [0.0, 0.0])


def test_cuped_theta_is_ols_slope() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=400)
    y = 3.0 * x + rng.normal(size=400)
    assert cuped_theta(y, x) == pytest.approx(np.polyfit(x, y, 1)[0])


def test_cuped_estimate_formula_and_variance_reduction() -> None:
    rng = np.random.default_rng(5)
    xa, xb = rng.normal(size=2000), rng.normal(size=2000)
    ya = xa + 0.3 * rng.normal(size=2000)
    yb = xb + 0.3 * rng.normal(size=2000) + 0.05
    res = cuped_test(ya, xa, yb, xb)
    theta = cuped_theta(np.concatenate([ya, yb]), np.concatenate([xa, xb]))
    expected = (yb.mean() - ya.mean()) - theta * (xb.mean() - xa.mean())
    assert res.estimate == pytest.approx(expected)
    # corr(x, y) ~ 0.96, so the standard error should shrink by roughly 3x or more.
    assert res.std_error < welch_ttest(ya, yb).std_error / 3


def test_cuped_rejects_constant_covariate() -> None:
    with pytest.raises(ValueError):
        cuped_test([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [1.0, 1.0, 1.0])
