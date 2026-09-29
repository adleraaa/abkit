"""Power and sample-size calculations for two-arm tests (normal approximation).

All functions assume equal allocation (the same n in each arm) and a two-sided
test. ``mde`` is the minimum detectable effect as an absolute difference.
"""

from __future__ import annotations

import math

from scipy import stats


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def sample_size_per_arm(
    var_control: float,
    var_treatment: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Users per arm needed to detect ``mde`` with the given power.

    n = (z_{1-alpha/2} + z_{power})^2 * (var_control + var_treatment) / mde^2
    """
    _check_alpha(alpha)
    if mde == 0:
        raise ValueError("mde must be non-zero")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    n = (z_alpha + z_power) ** 2 * (var_control + var_treatment) / mde**2
    return math.ceil(n)


def power_per_arm(
    var_control: float,
    var_treatment: float,
    n_per_arm: int,
    mde: float,
    alpha: float = 0.05,
) -> float:
    """Probability that a two-sided z-test rejects when the true effect is ``mde``.

    ``mde = 0`` is allowed and gives the size of the test, ``alpha``.
    """
    _check_alpha(alpha)
    if n_per_arm < 1:
        raise ValueError("n_per_arm must be at least 1")
    if var_control + var_treatment <= 0:
        raise ValueError("the variances must sum to a positive number")
    se = math.sqrt((var_control + var_treatment) / n_per_arm)
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    shift = abs(mde) / se
    # Include the (tiny) chance of rejecting in the wrong direction so that the
    # function returns alpha when the effect is 0.
    return float(stats.norm.sf(z_alpha - shift) + stats.norm.cdf(-z_alpha - shift))


def minimum_detectable_effect(
    var_control: float,
    var_treatment: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest absolute effect detected with the given power at ``n_per_arm`` users.

    The inverse of :func:`sample_size_per_arm` (without the rounding up).
    """
    _check_alpha(alpha)
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if n_per_arm < 1:
        raise ValueError("n_per_arm must be at least 1")
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_power = float(stats.norm.ppf(power))
    return (z_alpha + z_power) * math.sqrt((var_control + var_treatment) / n_per_arm)


def sample_size_proportions(
    baseline: float, mde: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Users per arm for a conversion rate moving from ``baseline`` to ``baseline + mde``."""
    treated = baseline + mde
    if not (0 < baseline < 1 and 0 < treated < 1):
        raise ValueError("baseline and baseline + mde must be in (0, 1)")
    return sample_size_per_arm(
        baseline * (1 - baseline), treated * (1 - treated), mde, alpha, power
    )


def sample_size_means(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per arm for a mean metric with standard deviation ``sd`` in both arms."""
    if sd <= 0:
        raise ValueError("sd must be positive")
    return sample_size_per_arm(sd**2, sd**2, mde, alpha, power)
