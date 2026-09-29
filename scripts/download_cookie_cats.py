"""Download the Cookie Cats A/B test dataset into data/cookie_cats.csv.

The dataset is not committed to this repository because its redistribution
terms are not stated anywhere we could find. It was published by DataCamp for
the project "Mobile Games A/B Testing with Cookie Cats" (Rasmus Baath), using
data from the game's developer, Tactile Entertainment. We download it from a
public GitHub copy of that project and verify a SHA-256 checksum, so every run
analyzes exactly the same file.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/0zz10/CookieCats-AB-Testing/master/datasets/cookie_cats.csv"
)
SHA256 = "5ab54d761fbddcd50de7b88e4eaf7837cba4569474f50c043a4d17ee342c46bd"
DEST = Path(__file__).resolve().parents[1] / "data" / "cookie_cats.csv"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists():
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=60) as resp:
            DEST.write_bytes(resp.read())
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    if digest != SHA256:
        print(f"checksum mismatch for {DEST}: got {digest}", file=sys.stderr)
        return 1
    print(f"ok: {DEST} (sha256 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
