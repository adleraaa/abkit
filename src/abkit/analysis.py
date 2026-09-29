"""Analyze a per-user experiment table: SRM check plus one test per metric and arm.

This is the layer the CLI uses. It picks the test for each metric, handles
missing values explicitly, and applies a multiple-comparison correction across
all (metric, treatment) pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    # Users actually used in the test, and users dropped because the metric
    # (or its denominator / covariate) was missing.
    n_control: int
    n_treatment: int
    dropped_control: int
    dropped_treatment: int


@dataclass(frozen=True)
class Report:
    variant_col: str
    arm_sizes: dict[str, int]
    srm: SRMResult
    correction: Correction
    comparisons: list[Comparison] = field(default_factory=list)
    unlabeled_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_col": self.variant_col,
            "arm_sizes": self.arm_sizes,
            "unlabeled_rows": self.unlabeled_rows,
            "srm": {
                "arms": list(self.arm_sizes),
                "observed": self.srm.observed,
                "expected": self.srm.expected,
                "statistic": self.srm.statistic,
                "p_value": self.srm.p_value,
                "threshold": self.srm.threshold,
                "mismatch": self.srm.mismatch,
            },
            "correction": self.correction,
            "notes": self.notes,
            "comparisons": [
                {
                    "metric": c.metric,
                    "control": c.control,
                    "treatment": c.treatment,
                    "n_control": c.n_control,
                    "n_treatment": c.n_treatment,
                    "dropped_control": c.dropped_control,
                    "dropped_treatment": c.dropped_treatment,
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
    """Run the test that matches ``kind`` for one metric between two arms.

    The inputs must not contain missing values; :func:`analyze` drops them first.
    """
    if covariate is not None and kind != "mean":
        raise ValueError(f"CUPED is implemented for mean metrics only, not {kind} metrics")
    if kind == "proportion":
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
    expected_split: Mapping[str, float] | None = None,
    alpha: float = 0.05,
) -> Report:
    """Compare every treatment arm against the control on every metric.

    ``denominator`` and ``covariate`` apply to all ``metrics``: with a denominator
    every metric becomes a ratio metric (``kind='auto'``), and the covariate is
    used for CUPED on the mean metrics only (a note in the report says which
    metrics it was skipped for). ``expected_split`` maps arm label to weight.

    Missing values are dropped per metric and arm (a row missing the metric, its
    denominator or the covariate is left out of that metric's test only) and the
    counts are reported. Rows with no arm label are dropped and counted too.
    """
    missing = [c for c in [variant_col, *metrics, denominator, covariate] if c and c not in df]
    if missing:
        raise ValueError(f"columns not found: {', '.join(missing)}")

    labeled = df[df[variant_col].notna()]
    unlabeled = len(df) - len(labeled)
    labels = labeled[variant_col].astype(str)
    arms = sorted(labels.unique())
    if len(arms) < 2:
        raise ValueError("need at least 2 arms in the variant column")
    control = arms[0] if control is None else control
    if control not in arms:
        raise ValueError(f"control arm {control!r} not in {arms}")
    treatments = [a for a in arms if a != control]
    ordered = [control, *treatments]

    groups = {arm: labeled[labels == arm] for arm in ordered}
    sizes = {arm: len(groups[arm]) for arm in ordered}

    ratios = None
    if expected_split is not None:
        if set(expected_split) != set(ordered):
            raise ValueError(
                f"expected split must name every arm exactly once: arms are {ordered}, "
                f"got {sorted(expected_split)}"
            )
        ratios = [expected_split[a] for a in ordered]
    srm = srm_check([sizes[a] for a in ordered], ratios)

    notes: list[str] = []
    if unlabeled:
        notes.append(f"{unlabeled} rows with an empty {variant_col!r} were dropped")

    results: list[tuple[str, str, TestResult, tuple[int, int, int, int]]] = []
    for metric in metrics:
        metric_kind = resolve_kind(labeled[metric], kind, denominator)
        # CUPED is only defined here for mean metrics; for the others the
        # covariate is skipped with a note instead of aborting the whole run.
        use_cov = covariate if metric_kind == "mean" else None
        if covariate is not None and use_cov is None:
            notes.append(f"covariate not applied to {metric!r} ({metric_kind} metric)")
        needed = [metric]
        if metric_kind == "ratio" and denominator is not None:
            needed.append(denominator)
        if use_cov is not None:
            needed.append(use_cov)

        complete = {arm: groups[arm].dropna(subset=needed) for arm in ordered}
        ctrl = complete[control]
        for arm in treatments:
            treat = complete[arm]
            res = compare(ctrl, treat, metric, metric_kind, denominator, use_cov, alpha)
            counts = (
                len(ctrl),
                len(treat),
                sizes[control] - len(ctrl),
                sizes[arm] - len(treat),
            )
            results.append((metric, arm, res, counts))
        dropped = sum(sizes[a] - len(complete[a]) for a in ordered)
        if dropped:
            notes.append(f"{metric!r}: {dropped} rows with missing values dropped")

    raw = np.array([r.p_value for _, _, r, _ in results])
    if correction == "holm":
        adjusted = holm(raw)
    elif correction == "bh":
        adjusted = benjamini_hochberg(raw)
    else:
        adjusted = raw
    comparisons = [
        Comparison(metric, control, arm, res, float(adj), *counts)
        for (metric, arm, res, counts), adj in zip(results, adjusted, strict=True)
    ]
    return Report(variant_col, sizes, srm, correction, comparisons, unlabeled, notes)
