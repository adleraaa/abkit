"""Case study: the Cookie Cats gate-placement experiment (~90k players).

Players were randomized to have the first progress gate at level 30 (control,
``gate_30``) or level 40 (``gate_40``). We run the same analysis the CLI runs:
SRM check, then 1-day retention, 7-day retention and game rounds played, with a
Holm correction across the three metrics.

Run ``python scripts/download_cookie_cats.py`` first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from abkit.analysis import analyze

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cookie_cats.csv"


def main() -> None:
    df = pd.read_csv(DATA)
    report = analyze(
        df,
        variant_col="version",
        metrics=["retention_1", "retention_7", "sum_gamerounds"],
        control="gate_30",
        correction="holm",
    )

    # sum_gamerounds is extremely skewed (one player logged ~50k rounds), so we
    # also report the same Welch test after capping at the 99.9th percentile as a
    # robustness check. The cap is a common practical choice, not a principled one.
    cap = float(np.quantile(df["sum_gamerounds"], 0.999))
    capped = df.assign(sum_gamerounds_capped=df["sum_gamerounds"].clip(upper=cap))
    robust = analyze(
        capped, "version", ["sum_gamerounds_capped"], control="gate_30", correction="none"
    )

    rounds = df["sum_gamerounds"]
    payload = {
        "source": "https://raw.githubusercontent.com/0zz10/CookieCats-AB-Testing/master/datasets/cookie_cats.csv",
        "rows": len(df),
        "sum_gamerounds_summary": {
            "median": float(rounds.median()),
            "mean": float(rounds.mean()),
            "max": int(rounds.max()),
            "p99_9_cap": cap,
        },
        "main": report.to_dict(),
        "robustness_capped_gamerounds": robust.to_dict()["comparisons"],
    }
    out = ROOT / "results" / "cookie_cats.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
