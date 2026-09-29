"""Run the full simulation study and save the results as JSON under results/.

Usage:
    python experiments/run_study.py            # full size (about 1-1.5 min with 5 processes)
    python experiments/run_study.py --quick    # tiny smoke run into a temp folder

Each study has its own fixed seed, so results are reproducible bit for bit on
the same numpy version. Studies run in separate processes because they are
independent and each is single-threaded Python.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from abkit import simulate

ROOT = Path(__file__).resolve().parents[1]

FULL: dict[str, tuple[str, dict[str, Any]]] = {
    "aa_type1": ("aa_type1", {"reps": 10_000, "seed": 101}),
    "power_curves": ("power_curves", {"reps": 2_000, "seed": 202}),
    "cuped_variance": ("cuped_variance", {"reps": 2_000, "seed": 303}),
    "peeking": ("peeking", {"reps_null": 5_000, "reps_alt": 2_000, "seed": 404}),
    "ratio_coverage": ("ratio_coverage", {"reps": 2_000, "seed": 505}),
}

QUICK: dict[str, tuple[str, dict[str, Any]]] = {
    "aa_type1": ("aa_type1", {"reps": 20, "seed": 101}),
    "power_curves": ("power_curves", {"reps": 10, "seed": 202}),
    "cuped_variance": ("cuped_variance", {"reps": 10, "seed": 303}),
    "peeking": ("peeking", {"reps_null": 20, "reps_alt": 10, "seed": 404}),
    "ratio_coverage": ("ratio_coverage", {"reps": 10, "seed": 505}),
}


def _run(name: str, func_name: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any], float]:
    start = time.perf_counter()
    result = getattr(simulate, func_name)(**kwargs)
    return name, result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args()

    config = QUICK if args.quick else FULL
    out_dir = args.out or (ROOT / ("results/quick" if args.quick else "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    timings: dict[str, float] = {}
    with ProcessPoolExecutor(max_workers=len(config)) as pool:
        futures = [pool.submit(_run, name, fn, kw) for name, (fn, kw) in config.items()]
        for fut in futures:
            name, result, seconds = fut.result()
            timings[name] = round(seconds, 1)
            (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"{name}: {seconds:.1f}s", flush=True)

    info = {
        "config": {name: kw for name, (_, kw) in config.items()},
        "seconds_per_study": timings,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
