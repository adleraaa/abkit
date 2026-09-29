"""Smoke tests for the simulation studies (tiny sizes; full runs live in experiments/)."""

import numpy as np
import pytest

from abkit import simulate


def test_true_ratio_formula_matches_large_sample() -> None:
    rng = np.random.default_rng(0)
    rev, ses, _ = simulate.session_data(rng, 200_000, heterogeneity=0.75, lift=0.1)
    assert rev.sum() / ses.sum() == pytest.approx(
        simulate.true_revenue_per_session(0.75, 0.1), rel=0.01
    )


def test_correlated_pair_has_requested_correlation() -> None:
    x, y = simulate.correlated_pair(np.random.default_rng(1), 50_000, 0.6)
    assert np.corrcoef(x, y)[0, 1] == pytest.approx(0.6, abs=0.01)


def test_studies_run_and_are_reproducible() -> None:
    a = simulate.aa_type1(reps=5, seed=3, n_per_arm=100)
    b = simulate.aa_type1(reps=5, seed=3, n_per_arm=100)
    assert a == b
    assert all(0 <= row["rate"] <= 1 for row in a["rows"])

    pc = simulate.power_curves(reps=3, seed=1)
    assert len(pc["proportions"]["rows"]) == 9
    cv = simulate.cuped_variance(reps=3, seed=1, n_per_arm=50)
    assert cv["rows"][0]["theory_1_minus_rho2"] == 1.0
    rc = simulate.ratio_coverage(reps=3, seed=1, n_users=50)
    assert {"delta_coverage", "naive_coverage"} <= set(rc["rows"][0])
