"""Analyze a per-user experiment table: SRM check plus one test per metric and arm.

This is the layer the CLI uses. It picks the right test for each metric and
applies a multiple-comparison correction across all (metric, treatment) pairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from abkit.cuped import cuped_test
from abkit.means import welch_ttest
from abkit.multiple import benjamini_hochberg, holm
from abkit.proportions import two_proportion_ztest
from abkit.ratio import delta_ratio_test
from abkit.result import TestResult
from abkit.srm import SRMResult, srm_check

Kind = Literal["auto", "mean", "proportion", "ratio"]
Correction = Literal["none", "holm", "bh"]


@dataclass(frozen=True)
class Comparison:
    metric: str
    control: str
    treatment: str
    result: TestResult
    adjusted_p_value: float


@dataclass(frozen=True)
class Report:
    variant_col: str
    arm_sizes: dict[str, int]
    srm: SRMResult
    correction: Correction
    comparisons: list[Comparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_col": self.variant_col,
            "arm_sizes": self.arm_sizes,
            "srm": {
                "observed": self.srm.observed,
                "expected": self.srm.expected,
                "statistic": self.srm.statistic,
                "p_value": self.srm.p_value,
                "threshold": self.srm.threshold,
                "mismatch": self.srm.mismatch,
            },
            "correction": self.correction,
            "comparisons": [
                {
                    "metric": c.metric,
                    "control": c.control,
                    "treatment": c.treatment,
                    "adjusted_p_value": c.adjusted_p_value,
                    **c.result.to_dict(),
                }
                for c in self.comparisons
            ],
        }


def _is_binary(values: pd.Series) -> bool:
    unique = pd.unique(values.dropna())
    return len(unique) <= 2 and set(np.asarray(unique, dtype=float)) <= {0.0, 1.0}


def resolve_kind(values: pd.Series, kind: Kind, denominator: str | None) -> Kind:
    """Choose the test family for a metric column when ``kind='auto'``."""
    if kind != "auto":
        return kind
    if denominator is not None:
        return "ratio"
    return "proportion" if _is_binary(values) else "mean"


def compare(
    control: pd.DataFrame,
    treatment: pd.DataFrame,
    metric: str,
    kind: Kind,
    denominator: str | None = None,
    covariate: str | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """Run the test that matches ``kind`` for one metric between two arms."""
    if kind == "proportion":
        if covariate is not None:
            raise ValueError("CUPED is implemented for mean metrics only")
        a = control[metric].astype(float)
        b = treatment[metric].astype(float)
        return two_proportion_ztest(int(a.sum()), a.size, int(b.sum()), b.size, alpha)
    if kind == "ratio":
        if denominator is None:
            raise ValueError("ratio metrics need a denominator column")
        return delta_ratio_test(
            control[metric], control[denominator], treatment[metric], treatment[denominator], alpha
        )
    if kind == "mean":
        if covariate is not None:
            return cuped_test(
                control[metric], control[covariate], treatment[metric], treatment[covariate], alpha
            )
        return welch_ttest(control[metric], treatment[metric], alpha)
    raise ValueError(f"unknown metric kind: {kind}")


def analyze(
    df: pd.DataFrame,
    variant_col: str,
    metrics: Sequence[str],
    control: str | None = None,
    kind: Kind = "auto",
    denominator: str | None = None,
    covariate: str | None = None,
    correction: Correction = "holm",
    expected_split: Sequence[float] | None = None,
    alpha: float = 0.05,
) -> Report:
    """Compare every treatment arm against the control on every metric."""
    missing = [c for c in [variant_col, *metrics, denominator, covariate] if c and c not in df]
    if missing:
        raise ValueError(f"columns not found: {', '.join(missing)}")

    arms = sorted(df[variant_col].astype(str).unique())
    if len(arms) < 2:
        raise ValueError("need at least 2 arms in the variant column")
    control = arms[0] if control is None else control
    if control not in arms:
        raise ValueError(f"control arm {control!r} not in {arms}")
    treatments = [a for a in arms if a != control]
    ordered = [control, *treatments]

    labels = df[variant_col].astype(str)
    groups = {arm: df[labels == arm] for arm in ordered}
    sizes = {arm: len(groups[arm]) for arm in ordered}
    srm = srm_check([sizes[a] for a in ordered], expected_split)

    results: list[tuple[str, str, TestResult]] = []
    for metric in metrics:
        metric_kind = resolve_kind(df[metric], kind, denominator)
        for arm in treatments:
            res = compare(
                groups[control], groups[arm], metric, metric_kind, denominator, covariate, alpha
            )
            results.append((metric, arm, res))

    raw = np.array([r.p_value for _, _, r in results])
    if correction == "holm":
        adjusted = holm(raw)
    elif correction == "bh":
        adjusted = benjamini_hochberg(raw)
    else:
        adjusted = raw
    comparisons = [
        Comparison(metric, control, arm, res, float(adj))
        for (metric, arm, res), adj in zip(results, adjusted, strict=True)
    ]
    return Report(variant_col, sizes, srm, correction, comparisons)
