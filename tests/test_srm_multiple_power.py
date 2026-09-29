import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower

from abkit import (
    benjamini_hochberg,
    holm,
    minimum_detectable_effect,
    power_per_arm,
    sample_size_means,
    sample_size_per_arm,
    sample_size_proportions,
    srm_check,
)


def test_srm_matches_scipy_and_flags_imbalance() -> None:
    res = srm_check([5000, 5400])
    ref = stats.chisquare([5000, 5400])
    assert res.p_value == pytest.approx(ref.pvalue)
    assert res.mismatch
    assert not srm_check([5000, 5040]).mismatch


def test_srm_uses_expected_ratios() -> None:
    # A 90/10 split that is exactly as designed is not a mismatch.
    res = srm_check([9000, 1000], expected_ratios=[9, 1])
    assert res.expected == pytest.approx([9000, 1000])
    assert res.p_value == pytest.approx(1.0)
    assert srm_check([9000, 1000]).mismatch


@pytest.mark.parametrize(
    "counts, ratios", [([10], None), ([10, -1], None), ([0, 0], None), ([5, 5], [1, 0])]
)
def test_srm_invalid(counts: list[int], ratios: list[float] | None) -> None:
    with pytest.raises(ValueError):
        srm_check(counts, ratios)


p_lists = st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=30)


@settings(max_examples=200)
@given(p=p_lists)
def test_holm_matches_statsmodels(p: list[float]) -> None:
    ref = multipletests(p, method="holm")[1]
    np.testing.assert_allclose(holm(p), ref, atol=1e-12)


@settings(max_examples=200)
@given(p=p_lists)
def test_bh_matches_statsmodels(p: list[float]) -> None:
    ref = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(benjamini_hochberg(p), ref, atol=1e-12)


def test_corrections_reject_bad_pvalues() -> None:
    for fn in (holm, benjamini_hochberg):
        with pytest.raises(ValueError):
            fn([0.1, 1.5])
        with pytest.raises(ValueError):
            fn([[0.1, 0.2]])


def test_sample_size_matches_statsmodels() -> None:
    # Equal variances: our formula equals statsmodels' with effect size mde/sd.
    ref = NormalIndPower().solve_power(effect_size=0.1 / 2.0, alpha=0.05, power=0.8, ratio=1.0)
    assert sample_size_means(sd=2.0, mde=0.1) == int(np.ceil(ref))


def test_power_matches_statsmodels() -> None:
    ref = NormalIndPower().power(effect_size=0.2, nobs1=300, alpha=0.05, ratio=1.0)
    assert power_per_arm(1.0, 1.0, 300, 0.2) == pytest.approx(ref)


@settings(max_examples=100)
@given(
    baseline=st.floats(0.01, 0.6),
    rel_mde=st.floats(0.02, 0.5),
    power=st.floats(0.5, 0.95),
)
def test_sample_size_reaches_target_power(baseline: float, rel_mde: float, power: float) -> None:
    mde = baseline * rel_mde
    n = sample_size_proportions(baseline, mde, power=power)
    var_a, var_b = baseline * (1 - baseline), (baseline + mde) * (1 - baseline - mde)
    # n is the smallest integer that reaches the target (the formula ignores the
    # negligible opposite tail, so allow a tiny slack at n - 1).
    assert power_per_arm(var_a, var_b, n, mde) >= power - 1e-9
    if n > 1:
        assert power_per_arm(var_a, var_b, n - 1, mde) < power + 1e-3


def test_power_at_zero_effect_is_alpha() -> None:
    assert power_per_arm(1.0, 1.0, 500, 0.0, alpha=0.05) == pytest.approx(0.05)
    assert power_per_arm(0.09, 0.09, 4000, 0.0, alpha=0.01) == pytest.approx(0.01)


def test_mde_inverts_sample_size() -> None:
    mde = minimum_detectable_effect(1.0, 1.0, 5600)
    # 5600 users per arm detect this effect with 80% power (plus the ~1e-6 chance
    # of rejecting in the wrong direction, which the sample-size formula ignores).
    assert power_per_arm(1.0, 1.0, 5600, mde) == pytest.approx(0.8, abs=1e-5)
    assert sample_size_per_arm(1.0, 1.0, mde) in (5600, 5601)


def test_power_input_validation() -> None:
    with pytest.raises(ValueError):
        sample_size_per_arm(1, 1, mde=0)
    with pytest.raises(ValueError):
        power_per_arm(1, 1, 0, 0.1)
    with pytest.raises(ValueError):
        power_per_arm(0, 0, 100, 0.1)
    with pytest.raises(ValueError):
        sample_size_proportions(0.95, 0.1)
    with pytest.raises(ValueError):
        sample_size_means(sd=-1, mde=0.1)
