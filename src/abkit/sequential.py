"""Always-valid sequential testing with the mixture SPRT (mSPRT).

Re-running a fixed-horizon test every day and stopping at the first p < 0.05
inflates the false-positive rate far above 5%. The mSPRT instead tracks a
likelihood ratio that is averaged over a normal prior on the effect,
N(0, tau^2). Under H0 this ratio is a non-negative martingale with mean 1, so by
Ville's inequality P(it ever exceeds 1/alpha) <= alpha: the experimenter may
look after every observation and stop at any time.

For a difference-in-means estimate ``d`` with variance ``V`` the mixture
likelihood ratio has the closed form

    Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 * d^2 / (2 V (V + tau^2)))

Reference: Johari, Koomen, Pekelis, Walsh (2017), "Peeking at A/B Tests: Why it
matters, and what to do about it", KDD. As in that paper's practical version we
plug in the estimated variance at each look, so the guarantee is asymptotic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def msprt_log_lambda(diff: float, var_diff: float, tau2: float) -> float:
    """Log of the mixture likelihood ratio for H0: true difference = 0."""
    if var_diff <= 0 or tau2 <= 0:
        raise ValueError("var_diff and tau2 must be positive")
    total = var_diff + tau2
    return 0.5 * math.log(var_diff / total) + tau2 * diff**2 / (2 * var_diff * total)


def always_valid_ci(
    diff: float, var_diff: float, tau2: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Confidence interval obtained by inverting the mSPRT at one look.

    It is the set of effects theta0 whose ``Lambda(diff - theta0) < 1/alpha``.
    """
    if var_diff <= 0 or tau2 <= 0:
        raise ValueError("var_diff and tau2 must be positive")
    total = var_diff + tau2
    half = math.sqrt(
        var_diff * total / tau2 * (2 * math.log(1 / alpha) + math.log(total / var_diff))
    )
    return diff - half, diff + half


@dataclass(frozen=True)
class SequentialResult:
    """Running always-valid quantities after each look."""

    p_values: list[float]
    ci_low: list[float]
    ci_high: list[float]
    alpha: float

    @property
    def stopped_at(self) -> int | None:
        """0-based index of the first look where H0 is rejected, or None."""
        for i, p in enumerate(self.p_values):
            if p < self.alpha:
                return i
        return None

    @property
    def ci_empty_at(self) -> int | None:
        """0-based index of the first look where the running CI became empty, or None."""
        for i, lo in enumerate(self.ci_low):
            if math.isnan(lo):
                return i
        return None


def msprt_monitor(
    diffs: Sequence[float],
    var_diffs: Sequence[float],
    tau2: float,
    alpha: float = 0.05,
) -> SequentialResult:
    """Always-valid p-values and CIs for a sequence of looks at cumulative data.

    The p-value is the running minimum of 1/Lambda and the CI is the running
    intersection of per-look intervals. Both are valid simultaneously over all
    looks, which is exactly what makes "stop whenever p < alpha" safe.

    If the per-look intervals stop overlapping, the intersection is empty. That
    means the data contradict the model (a constant effect with the plug-in
    variance), so the CI bounds are reported as NaN from that look on instead of
    as an inverted interval; ``ci_empty_at`` gives the first such look.
    """
    if len(diffs) != len(var_diffs):
        raise ValueError("diffs and var_diffs must have the same length")
    p_running = 1.0
    lo_running, hi_running = -math.inf, math.inf
    p_values, ci_low, ci_high = [], [], []
    for d, v in zip(diffs, var_diffs, strict=True):
        p_running = min(p_running, math.exp(-max(msprt_log_lambda(d, v, tau2), 0.0)))
        lo, hi = always_valid_ci(d, v, tau2, alpha)
        # Once empty, stay empty. Python's max/min do not propagate NaN reliably,
        # so the empty state is checked explicitly.
        if not math.isnan(lo_running):
            lo_running, hi_running = max(lo_running, lo), min(hi_running, hi)
            if lo_running > hi_running:
                lo_running = hi_running = math.nan
        p_values.append(p_running)
        ci_low.append(lo_running)
        ci_high.append(hi_running)
    return SequentialResult(p_values, ci_low, ci_high, alpha)
