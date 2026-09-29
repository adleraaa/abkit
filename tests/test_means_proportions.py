import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from abkit import two_proportion_ztest, welch_from_stats, welch_ttest


def test_welch_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 300)
    b = rng.normal(10.4, 3, 150)  # unequal n and variances: where Welch matters
    ours = welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert ours.statistic == pytest.approx(ref.statistic)
    assert ours.p_value == pytest.approx(ref.pvalue)
    assert ours.ci_low == pytest.approx(ci.low)
    assert ours.ci_high == pytest.approx(ci.high)
    assert ours.estimate == pytest.approx(b.mean() - a.mean())


def test_welch_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        welch_ttest([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        welch_from_stats(1.0, 0.0, 10, 1.0, 0.0, 10)


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    shift=st.floats(-1, 1),
    alpha=st.sampled_from([0.01, 0.05, 0.1]),
)
def test_welch_ci_and_pvalue_agree(seed: int, shift: float, alpha: float) -> None:
    """Test/CI duality: 0 lies outside the (1-alpha) CI exactly when p < alpha."""
    rng = np.random.default_rng(seed)
    res = welch_ttest(rng.normal(0, 1, 40), rng.normal(shift, 2, 25), alpha=alpha)
    zero_outside = not (res.ci_low <= 0 <= res.ci_high)
    # Skip the measure-zero boundary where rounding decides.
    if abs(res.p_value - alpha) > 1e-9:
        assert zero_outside == (res.p_value < alpha)


def test_ztest_matches_statsmodels() -> None:
    ours = two_proportion_ztest(120, 1000, 150, 1000)
    z, p = proportions_ztest([150, 120], [1000, 1000])
    assert ours.statistic == pytest.approx(z)
    assert ours.p_value == pytest.approx(p)
    lo, hi = confint_proportions_2indep(150, 1000, 120, 1000, method="wald", compare="diff")
    assert (ours.ci_low, ours.ci_high) == pytest.approx((lo, hi))
    assert ours.relative_lift == pytest.approx(0.25)


@pytest.mark.parametrize(
    "args",
    [(5, 0, 3, 10), (11, 10, 3, 10), (-1, 10, 3, 10), (0, 10, 0, 10), (10, 10, 10, 10)],
)
def test_ztest_invalid_input(args: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        two_proportion_ztest(*args)
