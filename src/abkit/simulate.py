"""Monte Carlo studies of the error rates of abkit's methods.

Every study takes a seed and a number of replications and returns a
JSON-serializable dict. ``experiments/run_study.py`` runs them at full size and
saves the output under ``results/``; the test suite runs them with tiny sizes.
The studies call the public library functions, so they test the code that
users run rather than a re-implementation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from abkit.cuped import cuped_test
from abkit.means import welch_from_stats, welch_ttest
from abkit.multiple import benjamini_hochberg, holm
from abkit.power import power_per_arm
from abkit.proportions import two_proportion_ztest
from abkit.ratio import delta_ratio_test
from abkit.sequential import msprt_monitor
from abkit.srm import srm_check

Array = NDArray[np.float64]

# Session-level noise around each user's typical spend (log scale).
SESSION_LOG_SD = 0.5


def _rate_row(
    method: str, scenario: str, rejections: int, reps: int, target: float
) -> dict[str, Any]:
    rate = rejections / reps
    return {
        "method": method,
        "scenario": scenario,
        "reps": reps,
        "rejections": rejections,
        "rate": rate,
        # Monte Carlo standard error of a proportion estimated from `reps` draws.
        "mc_se": math.sqrt(rate * (1 - rate) / reps),
        "target": target,
    }


def correlated_pair(rng: np.random.Generator, n: int, rho: float) -> tuple[Array, Array]:
    """Standard-normal (pre-period x, in-experiment y) pairs with correlation rho."""
    x = rng.standard_normal(n)
    y = rho * x + math.sqrt(1 - rho**2) * rng.standard_normal(n)
    return x, y


def session_data(
    rng: np.random.Generator, n_users: int, heterogeneity: float, lift: float = 0.0
) -> tuple[Array, Array, Array]:
    """Simulate revenue-per-session data with user-level randomization.

    Each user has a session count (1 + Poisson with a Gamma-distributed rate)
    and a personal spend level ``u ~ LogNormal(0, heterogeneity)``. Every
    session's revenue is ``u * LogNormal(0, SESSION_LOG_SD) * (1 + lift)``.
    ``heterogeneity`` controls how strongly sessions of one user are correlated
    (0 means sessions are i.i.d.).

    Returns ``(revenue per user, sessions per user, revenue per session)``.
    """
    sessions = 1 + rng.poisson(rng.gamma(shape=2.0, scale=2.0, size=n_users))
    user_level = rng.lognormal(0.0, heterogeneity, size=n_users)
    owner = np.repeat(np.arange(n_users), sessions)
    per_session = user_level[owner] * rng.lognormal(0.0, SESSION_LOG_SD, size=owner.size)
    per_session = per_session * (1 + lift)
    per_user = np.bincount(owner, weights=per_session, minlength=n_users)
    return per_user, sessions.astype(float), per_session


def true_revenue_per_session(heterogeneity: float, lift: float = 0.0) -> float:
    """Population ratio E[revenue] / E[sessions] for :func:`session_data`.

    Spend level is independent of session count, so the ratio is simply
    E[u] * E[session noise] * (1 + lift), both lognormal means.
    """
    return math.exp(heterogeneity**2 / 2) * math.exp(SESSION_LOG_SD**2 / 2) * (1 + lift)


def aa_type1(reps: int, seed: int, n_per_arm: int = 1000, alpha: float = 0.05) -> dict[str, Any]:
    """A/A tests: how often each method rejects when there is no effect."""
    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(
        ["welch", "ztest", "cuped", "delta", "naive_ratio", "srm", "raw10", "holm10", "bh10"], 0
    )
    n_metrics = 10
    for _ in range(reps):
        a, b = rng.lognormal(0, 1, n_per_arm), rng.lognormal(0, 1, n_per_arm)
        counts["welch"] += welch_ttest(a, b, alpha).significant

        conv_a, conv_b = rng.binomial(n_per_arm, 0.1), rng.binomial(n_per_arm, 0.1)
        counts["ztest"] += two_proportion_ztest(
            conv_a, n_per_arm, conv_b, n_per_arm, alpha
        ).significant

        xa, ya = correlated_pair(rng, n_per_arm, 0.7)
        xb, yb = correlated_pair(rng, n_per_arm, 0.7)
        counts["cuped"] += cuped_test(ya, xa, yb, xb, alpha).significant

        rev_a, ses_a, per_ses_a = session_data(rng, n_per_arm, heterogeneity=1.0)
        rev_b, ses_b, per_ses_b = session_data(rng, n_per_arm, heterogeneity=1.0)
        counts["delta"] += delta_ratio_test(rev_a, ses_a, rev_b, ses_b, alpha).significant
        counts["naive_ratio"] += welch_ttest(per_ses_a, per_ses_b, alpha).significant

        arm_a = int(rng.binomial(2 * n_per_arm, 0.5))
        counts["srm"] += srm_check([arm_a, 2 * n_per_arm - arm_a]).mismatch

        # A family of 10 independent A/A metrics, the familywise error rate.
        p = np.array(
            [
                welch_ttest(rng.standard_normal(200), rng.standard_normal(200)).p_value
                for _ in range(n_metrics)
            ]
        )
        counts["raw10"] += bool(np.any(p < alpha))
        counts["holm10"] += bool(np.any(holm(p) < alpha))
        counts["bh10"] += bool(np.any(benjamini_hochberg(p) < alpha))

    n = n_per_arm
    rows = [
        _rate_row("welch_t", f"lognormal(0,1) revenue, n={n}/arm", counts["welch"], reps, alpha),
        _rate_row("two_proportion_z", f"conversion 10%, n={n}/arm", counts["ztest"], reps, alpha),
        _rate_row(
            "cuped_welch_t", f"normal, pre-period corr 0.7, n={n}/arm", counts["cuped"], reps, alpha
        ),
        _rate_row(
            "delta_ratio_z",
            f"revenue/session, user heterogeneity 1.0, {n} users/arm",
            counts["delta"],
            reps,
            alpha,
        ),
        _rate_row(
            "naive_session_t",
            "same data, sessions treated as iid",
            counts["naive_ratio"],
            reps,
            alpha,
        ),
        _rate_row(
            "srm_chi2 (p<0.001)", f"true 50/50 split, {2 * n} users", counts["srm"], reps, 0.001
        ),
        _rate_row(
            "10 metrics, no correction",
            "familywise error, 10 independent A/A metrics",
            counts["raw10"],
            reps,
            alpha,
        ),
        _rate_row(
            "10 metrics, Holm",
            "familywise error, 10 independent A/A metrics",
            counts["holm10"],
            reps,
            alpha,
        ),
        _rate_row(
            "10 metrics, Benjamini-Hochberg",
            "familywise error, 10 independent A/A metrics",
            counts["bh10"],
            reps,
            alpha,
        ),
    ]
    return {"study": "aa_type1", "seed": seed, "reps": reps, "alpha": alpha, "rows": rows}


def power_curves(reps: int, seed: int, alpha: float = 0.05) -> dict[str, Any]:
    """Empirical power versus the analytic formula in :mod:`abkit.power`."""
    rng = np.random.default_rng(seed)

    baseline, n_prop = 0.10, 4000
    prop_rows = []
    for effect in np.linspace(0, 0.02, 9):
        p_b = baseline + float(effect)
        hits = sum(
            two_proportion_ztest(
                int(rng.binomial(n_prop, baseline)),
                n_prop,
                int(rng.binomial(n_prop, p_b)),
                n_prop,
                alpha,
            ).significant
            for _ in range(reps)
        )
        analytic = (
            power_per_arm(baseline * (1 - baseline), p_b * (1 - p_b), n_prop, float(effect), alpha)
            if effect > 0
            else alpha
        )
        prop_rows.append({"effect": float(effect), "empirical": hits / reps, "analytic": analytic})

    rho, n_mean = 0.7, 500
    mean_rows = []
    for effect in np.linspace(0, 0.25, 11):
        welch_hits = cuped_hits = 0
        for _ in range(reps):
            xa, ya = correlated_pair(rng, n_mean, rho)
            xb, yb = correlated_pair(rng, n_mean, rho)
            yb = yb + effect
            welch_hits += welch_ttest(ya, yb, alpha).significant
            cuped_hits += cuped_test(ya, xa, yb, xb, alpha).significant
        eff = float(effect)
        residual = 1 - rho**2
        mean_rows.append(
            {
                "effect": eff,
                "welch_empirical": welch_hits / reps,
                "cuped_empirical": cuped_hits / reps,
                "welch_analytic": power_per_arm(1, 1, n_mean, eff, alpha) if eff > 0 else alpha,
                "cuped_analytic": power_per_arm(residual, residual, n_mean, eff, alpha)
                if eff > 0
                else alpha,
            }
        )
    return {
        "study": "power_curves",
        "seed": seed,
        "reps": reps,
        "alpha": alpha,
        "proportions": {"baseline": baseline, "n_per_arm": n_prop, "rows": prop_rows},
        "means": {"sd": 1.0, "rho": rho, "n_per_arm": n_mean, "rows": mean_rows},
    }


def cuped_variance(reps: int, seed: int, n_per_arm: int = 1000) -> dict[str, Any]:
    """Variance of the CUPED estimate relative to the plain difference in means."""
    rng = np.random.default_rng(seed)
    rows = []
    for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        plain, adjusted, width_plain, width_cuped = [], [], [], []
        for _ in range(reps):
            xa, ya = correlated_pair(rng, n_per_arm, rho)
            xb, yb = correlated_pair(rng, n_per_arm, rho)
            r_plain = welch_ttest(ya, yb)
            r_cuped = cuped_test(ya, xa, yb, xb)
            plain.append(r_plain.estimate)
            adjusted.append(r_cuped.estimate)
            width_plain.append(r_plain.ci_high - r_plain.ci_low)
            width_cuped.append(r_cuped.ci_high - r_cuped.ci_low)
        rows.append(
            {
                "rho": rho,
                "variance_ratio": float(np.var(adjusted, ddof=1) / np.var(plain, ddof=1)),
                "theory_1_minus_rho2": 1 - rho**2,
                "mean_ci_width_ratio": float(np.mean(width_cuped) / np.mean(width_plain)),
            }
        )
    return {
        "study": "cuped_variance",
        "seed": seed,
        "reps": reps,
        "n_per_arm": n_per_arm,
        "rows": rows,
    }


def _peeking_run(
    rng: np.random.Generator,
    reps: int,
    effect: float,
    days: int,
    users_per_day: int,
    taus: list[float],
    alpha: float,
) -> dict[str, Any]:
    """Return first-rejection days (1-based, 0 = never) for naive peeking and each tau."""
    naive_day = np.zeros(reps, dtype=int)
    fixed_reject = np.zeros(reps, dtype=bool)
    msprt_day = {tau: np.zeros(reps, dtype=int) for tau in taus}
    n = users_per_day * np.arange(1, days + 1)
    for r in range(reps):
        a = rng.standard_normal((days, users_per_day))
        b = rng.standard_normal((days, users_per_day)) + effect
        # Cumulative mean and variance at the end of each day.
        stats_ab = []
        for arm in (a, b):
            s = np.cumsum(arm.sum(axis=1))
            ss = np.cumsum((arm**2).sum(axis=1))
            mean = s / n
            var = (ss - n * mean**2) / (n - 1)
            stats_ab.append((mean, var))
        (mean_a, var_a), (mean_b, var_b) = stats_ab

        for d in range(days):
            res = welch_from_stats(mean_a[d], var_a[d], n[d], mean_b[d], var_b[d], n[d], alpha)
            if res.significant and naive_day[r] == 0:
                naive_day[r] = d + 1
            if d == days - 1:
                fixed_reject[r] = res.significant

        diffs = (mean_b - mean_a).tolist()
        var_diffs = ((var_a + var_b) / n).tolist()
        for tau in taus:
            stop = msprt_monitor(diffs, var_diffs, tau**2, alpha).stopped_at
            msprt_day[tau][r] = 0 if stop is None else stop + 1

    def reject_by_day(first_day: NDArray[np.int_]) -> list[float]:
        return [float(np.mean((first_day > 0) & (first_day <= k))) for k in range(1, days + 1)]

    out: dict[str, Any] = {
        "effect": effect,
        "reps": reps,
        "fixed_horizon_reject": float(fixed_reject.mean()),
        "naive_reject_by_day": reject_by_day(naive_day),
        "msprt": [],
    }
    for tau in taus:
        first = msprt_day[tau]
        stopped = first[first > 0]
        out["msprt"].append(
            {
                "tau": tau,
                "reject_by_day": reject_by_day(first),
                "reject_rate": float(np.mean(first > 0)),
                "mean_stop_day_if_rejected": float(stopped.mean()) if stopped.size else None,
            }
        )
    return out


def peeking(
    reps_null: int,
    reps_alt: int,
    seed: int,
    days: int = 28,
    users_per_day: int = 200,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Daily peeking with a fixed-horizon t-test versus the always-valid mSPRT.

    The effect under the alternative is the one a single test at day ``days``
    detects with 80% power; tau (the prior sd of the mSPRT) is set relative to it.
    """
    rng = np.random.default_rng(seed)
    n_final = days * users_per_day
    # Effect with 80% power for the fixed-horizon z-test with unit variance.
    mde = (1.959963984540054 + 0.8416212335729143) * math.sqrt(2 / n_final)
    taus = [mde / 2, mde, 2 * mde, 4 * mde]
    return {
        "study": "peeking",
        "seed": seed,
        "alpha": alpha,
        "days": days,
        "users_per_day_per_arm": users_per_day,
        "mde": mde,
        "null": _peeking_run(rng, reps_null, 0.0, days, users_per_day, taus, alpha),
        "alternative": _peeking_run(rng, reps_alt, mde, days, users_per_day, taus, alpha),
    }


def ratio_coverage(
    reps: int, seed: int, n_users: int = 1000, lift: float = 0.05, alpha: float = 0.05
) -> dict[str, Any]:
    """Coverage of 95% CIs for a revenue-per-session difference.

    Compares the delta method on per-user aggregates with a naive Welch CI that
    treats every session as an independent observation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for h in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        truth = true_revenue_per_session(h, lift) - true_revenue_per_session(h)
        covered_delta = covered_naive = 0
        width_delta, width_naive = [], []
        for _ in range(reps):
            rev_a, ses_a, per_a = session_data(rng, n_users, h)
            rev_b, ses_b, per_b = session_data(rng, n_users, h, lift)
            d = delta_ratio_test(rev_a, ses_a, rev_b, ses_b, alpha)
            nv = welch_ttest(per_a, per_b, alpha)
            covered_delta += d.ci_low <= truth <= d.ci_high
            covered_naive += nv.ci_low <= truth <= nv.ci_high
            width_delta.append(d.ci_high - d.ci_low)
            width_naive.append(nv.ci_high - nv.ci_low)
        rows.append(
            {
                "heterogeneity": h,
                "true_diff": truth,
                "delta_coverage": covered_delta / reps,
                "naive_coverage": covered_naive / reps,
                "mean_width_delta": float(np.mean(width_delta)),
                "mean_width_naive": float(np.mean(width_naive)),
            }
        )
    return {
        "study": "ratio_coverage",
        "seed": seed,
        "reps": reps,
        "n_users_per_arm": n_users,
        "lift": lift,
        "nominal": 1 - alpha,
        "rows": rows,
    }
