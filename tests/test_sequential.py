import math

import numpy as np
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


@settings(max_examples=100)
@given(
    diffs=st.lists(st.floats(-5, 5), min_size=1, max_size=30),
    var_scale=st.floats(1e-4, 1),
    tau2=st.floats(1e-3, 1),
)
def test_monitor_never_returns_an_inverted_ci(
    diffs: list[float], var_scale: float, tau2: float
) -> None:
    """Each running CI is a real interval (lo <= hi) or NaN after it became empty."""
    variances = [var_scale / (k + 1) for k in range(len(diffs))]
    res = msprt_monitor(diffs, variances, tau2)
    empty_at = res.ci_empty_at
    for k, (lo, hi) in enumerate(zip(res.ci_low, res.ci_high, strict=True)):
        if empty_at is not None and k >= empty_at:
            assert math.isnan(lo) and math.isnan(hi)
        else:
            assert lo <= hi
    # Huge likelihood ratios underflow 1/Lambda to exactly 0, which is fine.
    assert all(0 <= p <= 1 for p in res.p_values)


def test_contradictory_looks_give_an_empty_ci() -> None:
    res = msprt_monitor([5.0, -5.0, 0.0], [0.01, 0.01, 0.01], tau2=1.0)
    assert res.ci_empty_at == 1
    assert res.ci_low[0] < res.ci_high[0]
    assert all(math.isnan(x) for x in res.ci_low[1:] + res.ci_high[1:])


def test_always_valid_ci_covers_truth_across_all_looks() -> None:
    """Seeded simulation: the running CI must contain the true effect at every look
    in at least ~95% of runs, even though we look 20 times."""
    rng = np.random.default_rng(12)
    effect, looks, per_look, runs = 0.3, 20, 50, 400
    n = per_look * np.arange(1, looks + 1)
    covered = 0
    for _ in range(runs):
        a = rng.standard_normal(looks * per_look)
        b = rng.standard_normal(looks * per_look) + effect
        diffs = [float(b[:k].mean() - a[:k].mean()) for k in n]
        var_diffs = [float((a[:k].var(ddof=1) + b[:k].var(ddof=1)) / k) for k in n]
        res = msprt_monitor(diffs, var_diffs, tau2=0.1)
        covered += all(lo <= effect <= hi for lo, hi in zip(res.ci_low, res.ci_high, strict=True))
    # Monte Carlo s.e. at 95% with 400 runs is about 1.1 points.
    assert covered / runs >= 0.93


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
