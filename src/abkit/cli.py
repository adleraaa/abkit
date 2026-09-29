"""Command-line interface: ``abkit analyze`` and ``abkit power``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import pandas as pd

from abkit.analysis import Report, analyze
from abkit.power import sample_size_means, sample_size_proportions


def _parse_split(text: str | None) -> dict[str, float] | None:
    """Parse ``gate_30=50,gate_40=50`` into ``{"gate_30": 50.0, "gate_40": 50.0}``.

    Weights are named rather than positional so that they cannot be paired with
    the wrong arm (arm order depends on which arm is the control).
    """
    if text is None:
        return None
    split: dict[str, float] = {}
    for part in text.split(","):
        label, sep, weight = part.partition("=")
        if not sep or not label.strip():
            raise ValueError(f"--expected-split takes label=weight pairs, got {part!r}")
        split[label.strip()] = float(weight)
    return split


def format_report(report: Report, alpha: float) -> str:
    """Render an analysis report as a plain-text table."""
    lines = []
    sizes = ", ".join(f"{arm}={n}" for arm, n in report.arm_sizes.items())
    lines.append(f"Arms ({report.variant_col}): {sizes}")
    srm = report.srm
    expected = ", ".join(
        f"{arm}={e:.1f}" for arm, e in zip(report.arm_sizes, srm.expected, strict=True)
    )
    verdict = "MISMATCH - do not trust the results below" if srm.mismatch else "ok"
    lines.append(
        f"SRM check: expected {expected}; chi2={srm.statistic:.3f}, p={srm.p_value:.4g} "
        f"(threshold {srm.threshold}) -> {verdict}"
    )
    lines.append("")
    header = (
        f"{'metric':<18}{'treatment':<12}{'method':<17}{'n_c':>8}{'n_t':>8}{'control':>11}"
        f"{'treat':>11}{'diff':>11}{'lift':>8}{'95% CI':>25}{'p':>10}{'p_adj':>10}  sig"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for c in report.comparisons:
        r = c.result
        ci = f"[{r.ci_low:.4g}, {r.ci_high:.4g}]"
        lift = f"{100 * r.relative_lift:.2f}%"
        sig = "yes" if c.adjusted_p_value < alpha else "no"
        lines.append(
            f"{c.metric:<18}{c.treatment:<12}{r.method:<17}{c.n_control:>8}{c.n_treatment:>8}"
            f"{r.control_value:>11.4g}{r.treatment_value:>11.4g}{r.estimate:>11.4g}{lift:>8}"
            f"{ci:>25}{r.p_value:>10.4g}{c.adjusted_p_value:>10.4g}  {sig}"
        )
    lines.append("")
    lines.append(
        f"p_adj: {report.correction} correction across all rows; sig uses p_adj < {alpha}."
    )
    lines.extend(f"note: {note}" for note in report.notes)
    return "\n".join(lines)


def _cmd_analyze(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.csv)
    report = analyze(
        df,
        variant_col=args.variant_col,
        metrics=args.metric,
        control=args.control,
        kind=args.kind,
        denominator=args.denominator,
        covariate=args.covariate,
        correction=args.correction,
        expected_split=_parse_split(args.expected_split),
        alpha=args.alpha,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report, args.alpha))
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    if (args.baseline is None) == (args.sd is None):
        raise ValueError("give exactly one of --baseline (proportion) or --sd (mean)")
    if args.baseline is not None:
        n = sample_size_proportions(args.baseline, args.mde, args.alpha, args.power)
        what = f"conversion {args.baseline:g} -> {args.baseline + args.mde:g}"
    else:
        n = sample_size_means(args.sd, args.mde, args.alpha, args.power)
        what = f"mean shift {args.mde} with sd {args.sd}"
    print(f"{what}: {n} users per arm ({2 * n} total) for power={args.power}, alpha={args.alpha}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abkit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="analyze a per-user experiment CSV")
    p_an.add_argument("csv", help="one row per randomization unit (user)")
    p_an.add_argument("--variant-col", required=True, help="column with the arm label")
    p_an.add_argument("--metric", action="append", required=True, help="metric column (repeatable)")
    p_an.add_argument("--control", help="control arm label (default: first in sort order)")
    p_an.add_argument(
        "--kind",
        choices=["auto", "mean", "proportion", "ratio"],
        default="auto",
        help="auto: ratio if --denominator, proportion if 0/1, else mean",
    )
    p_an.add_argument(
        "--denominator",
        help="per-user denominator column; with --kind auto it makes every --metric a ratio metric",
    )
    p_an.add_argument(
        "--covariate",
        help="pre-period covariate column for CUPED; applied to mean metrics only",
    )
    p_an.add_argument("--correction", choices=["none", "holm", "bh"], default="holm")
    p_an.add_argument(
        "--expected-split",
        help="intended split as label=weight pairs, e.g. a=90,b=10 (default: equal)",
    )
    p_an.add_argument("--alpha", type=float, default=0.05)
    p_an.add_argument("--json", action="store_true", help="print JSON instead of a table")
    p_an.set_defaults(func=_cmd_analyze)

    p_pw = sub.add_parser("power", help="sample size per arm for a two-sided test")
    p_pw.add_argument("--baseline", type=float, help="control conversion rate")
    p_pw.add_argument("--sd", type=float, help="standard deviation of a mean metric")
    p_pw.add_argument("--mde", type=float, required=True, help="absolute effect to detect")
    p_pw.add_argument("--alpha", type=float, default=0.05)
    p_pw.add_argument("--power", type=float, default=0.8)
    p_pw.set_defaults(func=_cmd_power)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code: int = args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"abkit: error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
