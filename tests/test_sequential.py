import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from abkit import always_valid_ci, msprt_log_lambda, msprt_monitor
from abkit.simulate import peeking


def test_lambda_below_one_at_zero_and_symmetric() -> None:
    assert msprt_log_lambda(0.0, 1.0, 1.0) < 0
    assert msprt_log_lambda(0.3, 0.01, 0.04) == pytest.approx(msprt_log_lambda(-0.3, 0.01, 0.04))
    assert msprt_log_lambda(0.5, 0.01, 0.04) > msprt_log_lambda(0.3, 0.01, 0.04)


@settings(max_examples=100)
@given(
    diff=st.floats(-5, 5),
    var=st.floats(1e-4, 10),
    tau2=st.floats(1e-4, 10),
    alpha=st.floats(0.001, 0.2),
)
def test_ci_boundary_is_where_lambda_hits_one_over_alpha(
    diff: float, var: float, tau2: float, alpha: float
) -> None:
    lo, hi = always_valid_ci(diff, var, tau2, alpha)
    assert lo <= diff <= hi
    # At the boundary theta0 the test of H0: effect = theta0 is exactly at threshold.
    assert msprt_log_lambda(diff - lo, var, tau2) == pytest.approx(math.log(1 / alpha), abs=1e-6)
    assert msprt_log_lambda(diff - hi, var, tau2) == pytest.approx(math.log(1 / alpha), abs=1e-6)


@settings(max_examples=50)
@given(
    diffs=st.lists(st.floats(-1, 1), min_size=1, max_size=30),
    tau2=st.floats(1e-3, 1),
)
def test_monitor_pvalues_nonincreasing_and_cis_nested(diffs: list[float], tau2: float) -> None:
    variances = [1.0 / (k + 1) for k in range(len(diffs))]
    res = msprt_monitor(diffs, variances, tau2)
    assert all(b <= a for a, b in zip(res.p_values, res.p_values[1:], strict=False))
    assert all(b >= a for a, b in zip(res.ci_low, res.ci_low[1:], strict=False))
    assert all(b <= a for a, b in zip(res.ci_high, res.ci_high[1:], strict=False))
    assert all(0 < p <= 1 for p in res.p_values)


def test_stopped_at() -> None:
    res = msprt_monitor([0.0, 0.0, 2.0], [0.01, 0.01, 0.01], tau2=1.0)
    assert res.stopped_at == 2
    assert msprt_monitor([0.0], [0.01], tau2=1.0).stopped_at is None


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        msprt_log_lambda(0.1, 0.0, 1.0)
    with pytest.raises(ValueError):
        msprt_monitor([0.1, 0.2], [1.0], 1.0)


def test_peeking_simulation_controls_error_only_for_msprt() -> None:
    """Small seeded A/A run: naive daily peeking inflates errors, mSPRT does not."""
    out = peeking(reps_null=300, reps_alt=10, seed=7, days=14, users_per_day=100)
    null = out["null"]
    naive = null["naive_reject_by_day"][-1]
    assert naive > 0.12
    for row in null["msprt"]:
        assert row["reject_rate"] <= 0.06
    # A single test at the final day stays near 5%.
    assert null["fixed_horizon_reject"] < 0.09
