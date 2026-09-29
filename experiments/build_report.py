"""Build results/summary.md and a static HTML report (site/) from results/*.json.

Standard library only, so the GitHub Pages workflow can run it without
installing anything. Every number on the page is read from the JSON files.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SITE = ROOT / "site"

Table = tuple[list[str], list[list[str]]]


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((RESULTS / f"{name}.json").read_text())
    return data


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def aa_table() -> Table:
    rows = [
        [r["method"], r["scenario"], pct(r["rate"], 2), pct(r["mc_se"], 2), pct(r["target"], 1)]
        for r in load("aa_type1")["rows"]
    ]
    return ["Method", "Scenario", "Rejection rate", "MC s.e.", "Target"], rows


def peeking_table() -> Table:
    d = load("peeking")
    null, alt, days = d["null"], d["alternative"], d["days"]
    rows = [
        [
            f"t-test once at day {days} (fixed horizon)",
            pct(null["fixed_horizon_reject"]),
            pct(alt["fixed_horizon_reject"]),
            f"{days} (fixed)",
        ],
        [
            "t-test every day, stop at first p < 0.05",
            pct(null["naive_reject_by_day"][-1]),
            pct(alt["naive_reject_by_day"][-1]),
            "-",
        ],
    ]
    for m in null["msprt"]:
        a = next(x for x in alt["msprt"] if x["tau"] == m["tau"])
        stop = a["mean_stop_day_if_rejected"]
        rows.append(
            [
                f"mSPRT every day, tau = {m['tau'] / d['mde']:g} x MDE",
                pct(m["reject_rate"]),
                pct(a["reject_rate"]),
                f"{stop:.1f}" if stop is not None else "-",
            ]
        )
    header = [
        f"Procedure ({days} daily looks)",
        f"False positive rate (A/A, {null['reps']:,} reps)",
        f"Power at effect = MDE ({alt['reps']:,} reps)",
        "Mean stopping day when it rejects",
    ]
    return header, rows


def ratio_table() -> Table:
    d = load("ratio_coverage")
    rows = [
        [
            f"{r['heterogeneity']:g}",
            pct(r["delta_coverage"]),
            pct(r["naive_coverage"]),
            f"{r['mean_width_delta'] / r['mean_width_naive']:.2f}",
        ]
        for r in d["rows"]
    ]
    return [
        "User heterogeneity",
        "Delta-method coverage",
        "Naive coverage",
        "CI width ratio (delta / naive)",
    ], rows


def cuped_table() -> Table:
    rows = [
        [
            f"{r['rho']:g}",
            f"{r['variance_ratio']:.3f}",
            f"{r['theory_1_minus_rho2']:.3f}",
            f"{r['mean_ci_width_ratio']:.3f}",
        ]
        for r in load("cuped_variance")["rows"]
    ]
    return [
        "corr(pre, metric)",
        "Var(CUPED)/Var(plain), simulated",
        "1 - rho^2",
        "CI width ratio",
    ], rows


def power_table() -> Table:
    d = load("power_curves")
    prop = d["proportions"]["rows"]
    means = d["means"]["rows"]
    gaps = {
        "z-test, conversion": max(abs(r["empirical"] - r["analytic"]) for r in prop),
        "Welch t-test, mean": max(abs(r["welch_empirical"] - r["welch_analytic"]) for r in means),
        "CUPED, mean": max(abs(r["cuped_empirical"] - r["cuped_analytic"]) for r in means),
    }
    rows = [[k, f"{v:.3f}"] for k, v in gaps.items()]
    return ["Test", f"Max abs(simulated - analytic) power ({d['reps']:,} reps per point)"], rows


def cookie_table() -> tuple[Table, dict[str, Any]]:
    d = load("cookie_cats")
    rows = []
    for c in d["main"]["comparisons"]:
        rows.append(
            [
                c["metric"],
                c["method"],
                f"{c['control_value']:.4g}",
                f"{c['treatment_value']:.4g}",
                f"{c['estimate']:+.4g} ({c['relative_lift'] * 100:+.2f}%)",
                f"[{c['ci_low']:.4g}, {c['ci_high']:.4g}]",
                f"{c['p_value']:.3g}",
                f"{c['adjusted_p_value']:.3g}",
            ]
        )
    for c in d["robustness_capped_gamerounds"]:
        rows.append(
            [
                c["metric"] + " (capped at p99.9)",
                c["method"],
                f"{c['control_value']:.4g}",
                f"{c['treatment_value']:.4g}",
                f"{c['estimate']:+.4g} ({c['relative_lift'] * 100:+.2f}%)",
                f"[{c['ci_low']:.4g}, {c['ci_high']:.4g}]",
                f"{c['p_value']:.3g}",
                "-",
            ]
        )
    header = ["Metric", "Test", "gate_30", "gate_40", "Diff (lift)", "95% CI", "p", "p (Holm)"]
    return (header, rows), d


def md_table(table: Table) -> str:
    header, rows = table
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def html_table(table: Table) -> str:
    header, rows = table
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def sections() -> list[tuple[str, str, Table, str | None]]:
    """(title, explanation, table, figure) for each study, in page order."""
    peek = load("peeking")
    return [
        (
            "A/A Type I error",
            "Each method is run on data with no true effect. A correct 5% test rejects about 5% "
            "of the time. The naive per-session t-test and uncorrected testing of 10 metrics do not.",
            aa_table(),
            "aa_type1.png",
        ),
        (
            "Power: simulation vs. formula",
            "Simulated power of the tests against the analytic power from abkit.power; CUPED "
            "power uses the residual variance (1 - rho^2).",
            power_table(),
            "power_curves.png",
        ),
        (
            "CUPED variance reduction",
            "Variance of the CUPED estimate divided by the variance of the plain difference in "
            "means, across replications, against the theoretical 1 - rho^2.",
            cuped_table(),
            "cuped_variance.png",
        ),
        (
            "Peeking: daily t-tests vs. mSPRT",
            f"{peek['days']} days, {peek['users_per_day_per_arm']} users per arm per day, normal "
            f"data. The MDE is the effect that the single day-{peek['days']} test detects with "
            "80% power.",
            peeking_table(),
            "peeking.png",
        ),
        (
            "Ratio metrics: delta method vs. naive CI",
            "Revenue per session with user-level randomization; the true treatment effect is "
            "+5%. Coverage is the share of 95% CIs that contain the true difference.",
            ratio_table(),
            "ratio_coverage.png",
        ),
    ]


def build_markdown() -> str:
    parts = [
        "# Simulation and case-study summary",
        "",
        "Generated by `experiments/build_report.py` from `results/*.json`.",
        "",
    ]
    for title, text, table, _ in sections():
        parts += [f"## {title}", "", text, "", md_table(table), ""]
    table, d = cookie_table()
    srm = d["main"]["srm"]
    parts += [
        "## Cookie Cats case study",
        "",
        f"{d['rows']:,} players. SRM check: arms {srm['observed']}, chi2 = "
        f"{srm['statistic']:.2f}, p = {srm['p_value']:.4f} (threshold {srm['threshold']}).",
        "",
        f"Source: {d['mirror']} (SHA-256 `{d['sha256']}`).",
        "",
        md_table(table),
        "",
    ]
    return "\n".join(parts)


CSS = """
:root { --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink-2:#52514e; --line:#e4e3de;
        --accent:#2a78d6; --code:#f3f2ee; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#1a1a19; --card:#232322; --ink:#ffffff; --ink-2:#c3c2b7; --line:#3a3a38;
  --accent:#3987e5; --code:#2c2c2a; } }
:root[data-theme="dark"] { --bg:#1a1a19; --card:#232322; --ink:#ffffff; --ink-2:#c3c2b7;
  --line:#3a3a38; --accent:#3987e5; --code:#2c2c2a; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 1.9rem; margin: 0 0 4px; }
h2 { font-size: 1.25rem; margin: 40px 0 8px; padding-top: 8px; border-top: 1px solid var(--line); }
p { color: var(--ink-2); margin: 6px 0 12px; }
a { color: var(--accent); }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; font-size: .9em; }
figure { margin: 16px 0; background: #fcfcfb; border: 1px solid var(--line); border-radius: 8px;
  padding: 8px; }
figure img { width: 100%; height: auto; display: block; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; }
.meta { font-size: .9rem; }
"""


def build_html() -> str:
    info = load("run_info")
    body = [
        "<h1>abkit simulation report</h1>",
        '<p class="meta">Error rates of the methods in '
        '<a href="https://github.com/adleraaa/abkit">abkit</a>, measured by seeded Monte Carlo '
        f"simulation (Python {html.escape(info['python'])}, numpy {html.escape(info['numpy'])}, "
        f"{info['cpu_count']} logical CPUs, {info['wall_seconds']} s wall time for all studies). "
        "Reproduce with <code>python experiments/run_study.py</code>.</p>",
    ]
    for title, text, table, fig in sections():
        body.append(f"<h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>")
        if fig:
            body.append(
                f'<figure><img src="figures/{fig}" alt="{html.escape(title)} chart"></figure>'
            )
        body.append(html_table(table))
    table, d = cookie_table()
    srm = d["main"]["srm"]
    body.append(
        "<h2>Case study: Cookie Cats</h2>"
        f"<p>{d['rows']:,} players randomized to the first gate at level 30 (control) or 40. "
        f"SRM check: arms {srm['observed']}, chi2 = {srm['statistic']:.2f}, "
        f"p = {srm['p_value']:.4f}, which passes the 0.001 threshold but would fail at 0.01. "
        'Data from DataCamp\'s project "Mobile Games A/B Testing with Cookie Cats" (data from '
        "Tactile Entertainment), downloaded by <code>scripts/download_cookie_cats.py</code> "
        f'from the public mirror <a href="{html.escape(d["mirror"])}">'
        f"{html.escape(d['mirror'])}</a> and not redistributed here. SHA-256 of the analyzed "
        f"file: <code>{d['sha256']}</code>. The mirror carries an MIT license from its uploader; "
        "we found no license from the original data owners.</p>"
    )
    body.append(html_table(table))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>abkit simulation report</title>"
        f"<style>{CSS}</style></head><body><main>{''.join(body)}</main></body></html>\n"
    )


def main() -> None:
    (RESULTS / "summary.md").write_text(build_markdown(), encoding="utf-8")
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(build_html(), encoding="utf-8")
    shutil.copytree(RESULTS / "figures", SITE / "figures", dirs_exist_ok=True)
    print(f"wrote {RESULTS / 'summary.md'} and {SITE / 'index.html'}")


if __name__ == "__main__":
    main()
