"""Draw the study figures from the JSON files in results/ (no re-simulation)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # file output only, never open a window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 9.5,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "savefig.dpi": 150,
    }
)


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((RESULTS / f"{name}.json").read_text())
    return data


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def fig_aa() -> None:
    data = load("aa_type1")
    rows = [r for r in data["rows"] if not r["method"].startswith("srm")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = [r["method"] for r in rows][::-1]
    rates = [r["rate"] for r in rows][::-1]
    errs = [1.96 * r["mc_se"] for r in rows][::-1]
    colors = [ORANGE if r > 0.07 else BLUE for r in rates]
    ax.axvline(data["alpha"], color=INK_2, linestyle="--", linewidth=1)
    ax.errorbar(rates, range(len(rows)), xerr=errs, fmt="none", ecolor=INK_2, elinewidth=1)
    ax.scatter(rates, range(len(rows)), s=40, c=colors, zorder=3, edgecolors=SURFACE)
    for y, r in enumerate(rates):
        ax.annotate(
            f"{100 * r:.1f}%",
            (r, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=8.5,
        )
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlabel("Rejection rate in A/A tests (dashed line: alpha = 5%)")
    ax.set_title(f"Type I error, {data['reps']:,} A/A replications per method")
    ax.set_xlim(0, max(rates) * 1.18)
    save(fig, "aa_type1")


def fig_power() -> None:
    data = load("power_curves")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    prop = data["proportions"]
    x = [100 * r["effect"] for r in prop["rows"]]
    ax1.plot(x, [r["analytic"] for r in prop["rows"]], color=BLUE, label="analytic (abkit.power)")
    ax1.scatter(
        x,
        [r["empirical"] for r in prop["rows"]],
        color=BLUE,
        s=36,
        zorder=3,
        edgecolors=SURFACE,
        label="simulated z-test",
    )
    ax1.set_title(f"Conversion {100 * prop['baseline']:.0f}%, n={prop['n_per_arm']:,}/arm")
    ax1.set_xlabel("Absolute lift (percentage points)")
    ax1.set_ylabel("Power")
    ax1.legend(loc="lower right")

    means = data["means"]
    x = [r["effect"] for r in means["rows"]]
    ax2.plot(x, [r["welch_analytic"] for r in means["rows"]], color=BLUE, label="Welch, analytic")
    ax2.scatter(
        x,
        [r["welch_empirical"] for r in means["rows"]],
        color=BLUE,
        s=36,
        zorder=3,
        edgecolors=SURFACE,
        label="Welch, simulated",
    )
    ax2.plot(x, [r["cuped_analytic"] for r in means["rows"]], color=ORANGE, label="CUPED, analytic")
    ax2.scatter(
        x,
        [r["cuped_empirical"] for r in means["rows"]],
        color=ORANGE,
        s=36,
        zorder=3,
        marker="s",
        edgecolors=SURFACE,
        label="CUPED, simulated",
    )
    ax2.set_title(f"Mean metric, pre-period corr {means['rho']}, n={means['n_per_arm']}/arm")
    ax2.set_xlabel("Effect (in standard deviations)")
    ax2.legend(loc="lower right")
    for ax in (ax1, ax2):
        ax.set_ylim(0, 1.03)
    save(fig, "power_curves")


def fig_cuped() -> None:
    data = load("cuped_variance")
    rows = data["rows"]
    rho = [r["rho"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(
        rho,
        [r["theory_1_minus_rho2"] for r in rows],
        color=INK_2,
        linestyle="--",
        linewidth=1.5,
        label="theory: 1 - rho^2",
    )
    ax.scatter(
        rho,
        [r["variance_ratio"] for r in rows],
        color=BLUE,
        s=40,
        zorder=3,
        edgecolors=SURFACE,
        label="simulated Var(CUPED) / Var(plain)",
    )
    ax.set_xlabel("Correlation between pre-period covariate and metric")
    ax.set_ylabel("Variance ratio")
    ax.set_title(f"CUPED variance reduction ({data['reps']:,} reps per point)")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left")
    save(fig, "cuped_variance")


def fig_peeking() -> None:
    data = load("peeking")
    days = list(range(1, data["days"] + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    null = data["null"]
    msprt_mde = next(m for m in null["msprt"] if abs(m["tau"] - data["mde"]) < 1e-12)
    ax1.plot(days, null["naive_reject_by_day"], color=ORANGE, label="t-test, peek daily")
    ax1.plot(days, msprt_mde["reject_by_day"], color=BLUE, label="mSPRT (tau = MDE)")
    ax1.axhline(data["alpha"], color=INK_2, linestyle="--", linewidth=1)
    ax1.set_title(f"A/A: false positives so far ({null['reps']:,} reps)")
    ax1.set_xlabel("Day (looks so far)")
    ax1.set_ylabel("Share of experiments stopped")
    ax1.legend(loc="upper left")

    alt = data["alternative"]
    for m, color in zip(alt["msprt"], [AQUA, BLUE, ORANGE, VIOLET], strict=True):
        ratio = m["tau"] / data["mde"]
        ax2.plot(days, m["reject_by_day"], color=color, label=f"mSPRT, tau = {ratio:g} x MDE")
    ax2.scatter(
        [days[-1]],
        [alt["fixed_horizon_reject"]],
        color=INK,
        s=40,
        zorder=3,
        label="fixed-horizon t-test (day 28)",
    )
    ax2.set_title(f"Effect = MDE: detections so far ({alt['reps']:,} reps)")
    ax2.set_xlabel("Day")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(0, 1)
    save(fig, "peeking")


def fig_ratio() -> None:
    data = load("ratio_coverage")
    rows = data["rows"]
    h = [r["heterogeneity"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.axhline(data["nominal"], color=INK_2, linestyle="--", linewidth=1)
    ax.plot(
        h,
        [r["delta_coverage"] for r in rows],
        color=BLUE,
        marker="o",
        label="delta method (per-user aggregates)",
    )
    ax.plot(
        h,
        [r["naive_coverage"] for r in rows],
        color=ORANGE,
        marker="s",
        label="naive (sessions treated as iid)",
    )
    ax.set_xlabel("User-level heterogeneity (sd of log spend level)")
    ax.set_ylabel("Coverage of 95% CI")
    ax.set_title(f"Revenue per session: CI coverage ({data['reps']:,} reps per point)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    save(fig, "ratio_coverage")


def main() -> None:
    global RESULTS, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS, help="folder with study JSON")
    args = parser.parse_args()
    RESULTS, FIGURES = args.results, args.results / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in (fig_aa, fig_power, fig_cuped, fig_peeking, fig_ratio):
        fn()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
