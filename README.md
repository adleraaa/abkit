# abkit

[![CI](https://github.com/adleraaa/abkit/actions/workflows/ci.yml/badge.svg)](https://github.com/adleraaa/abkit/actions/workflows/ci.yml)

A small Python library and CLI for analyzing online A/B experiments: Welch t-test,
two-proportion z-test, delta-method ratio metrics, CUPED, sample ratio mismatch
checks, Holm / Benjamini-Hochberg corrections, a power calculator, and always-valid
sequential testing (mSPRT). Each method comes with a seeded simulation study that
measures what it actually does to error rates, because the common mistakes in
experiment analysis (treating sessions as independent, peeking daily, testing many
metrics) do not show up as crashes, only as wrong p-values.

Report page with all figures and tables: https://adleraaa.github.io/abkit/

## Results

All numbers below come from `experiments/run_study.py` (seeded, saved in
[`results/*.json`](results/)) and `experiments/cookie_cats.py`
([`results/cookie_cats.json`](results/cookie_cats.json)). The full table set is in
[`results/summary.md`](results/summary.md). The whole simulation study takes 62 s of
wall time on a laptop CPU (Intel i9-14900HX, 5 studies in parallel processes, Python
3.13, numpy 2.5; see [`results/run_info.json`](results/run_info.json)).

**Type I error in A/A tests** (10,000 replications each, Monte Carlo s.e. about 0.2 points):

| Method | Data | Rejection rate | Target |
|---|---|---|---|
| Welch t-test | lognormal revenue, 1000/arm | 4.77% | 5% |
| Two-proportion z-test | 10% conversion, 1000/arm | 5.24% | 5% |
| CUPED + Welch | pre-period corr 0.7, 1000/arm | 5.19% | 5% |
| Delta method, revenue per session | 1000 users/arm, clustered sessions | 4.84% | 5% |
| Naive t-test treating sessions as iid | same data | **40.46%** | 5% |
| SRM chi-square at p < 0.001 | true 50/50 split | 0.12% | 0.1% |
| 10 independent metrics, no correction (any rejection) | | **41.20%** | 5% |
| 10 metrics, Holm | | 5.14% | 5% |
| 10 metrics, Benjamini-Hochberg | | 5.28% | 5% |

**Peeking** (28 daily looks, 200 users/arm/day; effect = the MDE that one test at day 28
detects with 80% power; tau is the mSPRT prior sd):

| Procedure | False positive rate (5,000 A/A runs) | Power at MDE (2,000 runs) | Mean stop day when it rejects |
|---|---|---|---|
| One t-test at day 28 | 4.6% | 78.9% | 28 |
| t-test every day, stop at first p < 0.05 | **27.3%** | 87.8% (invalid: error not controlled) | - |
| mSPRT every day, tau = MDE | 0.9% | 48.1% | 17.5 |
| mSPRT every day, tau = 2 x MDE | 1.4% | 46.2% | 16.2 |

The mSPRT keeps the false-positive rate under 5% while peeking daily, but it is
conservative over a 28-day window: its guarantee covers an unbounded horizon, so at
the same final sample size it detects the MDE in 48% of runs versus 79% for the single
fixed-horizon test. It pays off when effects are larger than planned or when stopping
early matters; this study does not show it winning on raw power.

**Ratio metric CI coverage** (revenue per session, true lift +5%, 2,000 reps per row):
with no user heterogeneity both CIs cover 95.0%. As user-level heterogeneity grows the
naive per-session CI drops to 80.8%, 68.8%, 60.5%, 58.5% and 57.7%, while the delta
method stays between 94.5% and 95.5%.

**CUPED**: the simulated variance ratio tracks 1 - rho^2 (e.g. 0.504 vs 0.51 at
rho = 0.7, 0.208 vs 0.19 at rho = 0.9). **Power formula**: across all effect sizes,
simulated power is within 0.023 of `abkit.power` (2,000 reps per point).

**Case study: Cookie Cats** (90,189 mobile-game players; moving the first gate from
level 30 to 40). SRM p = 0.0086: passes the 0.001 threshold but would be flagged at
0.01, so treat the results with some caution.

| Metric | gate_30 | gate_40 | Diff | 95% CI | p | p (Holm, 3 metrics) |
|---|---|---|---|---|---|---|
| 1-day retention | 44.82% | 44.23% | -0.59 pp | [-1.24, +0.06] pp | 0.074 | 0.149 |
| 7-day retention | 19.02% | 18.20% | -0.82 pp | [-1.33, -0.31] pp | 0.0016 | 0.0047 |
| Game rounds (mean) | 52.46 | 51.30 | -1.16 | [-3.72, 1.40] | 0.376 | 0.376 |

Moving the gate later lowered 7-day retention by 0.82 points (4.3% relative), and this
survives the Holm correction.

## Architecture

```
src/abkit/
  means.py        Welch t-test (from raw arrays or summary statistics)
  proportions.py  two-proportion z-test (pooled SE for the test, unpooled for the CI)
  ratio.py        delta-method variance for sum(y)/sum(x) metrics
  cuped.py        pre-period covariate adjustment, then Welch
  srm.py          chi-square sample ratio mismatch check
  multiple.py     Holm and Benjamini-Hochberg adjusted p-values
  power.py        sample size and power (normal approximation)
  sequential.py   mSPRT always-valid p-values and CIs
  analysis.py     per-user table -> SRM check + right test per metric + correction
  cli.py          `abkit analyze`, `abkit power`
  simulate.py     the Monte Carlo studies (numpy only, call the functions above)
experiments/
  run_study.py    full-size seeded run -> results/*.json
  make_figures.py results/*.json -> results/figures/*.png (matplotlib, Agg)
  cookie_cats.py  case study -> results/cookie_cats.json
  build_report.py results/ -> results/summary.md + site/index.html (stdlib only)
```

The simulation studies call the same public functions that users call, so the study is
also an end-to-end test of the library rather than of a separate re-implementation.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                               # 45 tests

abkit power --baseline 0.10 --mde 0.01
# conversion 0.1 -> 0.11: 14749 users per arm (29498 total) for power=0.8, alpha=0.05

python scripts/download_cookie_cats.py              # fetch + verify sha256
abkit analyze data/cookie_cats.csv --variant-col version --control gate_30 \
    --metric retention_1 --metric retention_7 --metric sum_gamerounds
```

`abkit analyze` expects one row per randomization unit. It picks the test per metric
(`--kind auto`: 0/1 columns get the z-test, `--denominator COL` makes a delta-method
ratio metric, otherwise Welch), uses CUPED when `--covariate COL` is given, runs an SRM
check against `--expected-split` (default equal), and applies `--correction holm|bh|none`
across all metric x treatment comparisons. `--json` prints machine-readable output.

Library use:

```python
from abkit import delta_ratio_test, msprt_monitor, sample_size_proportions

res = delta_ratio_test(revenue_a, sessions_a, revenue_b, sessions_b)
print(res.estimate, res.ci_low, res.ci_high, res.p_value)
```

## Reproduce

```bash
pip install -e ".[dev]"
python experiments/run_study.py        # ~1 min with 5 processes; rewrites results/*.json
python experiments/make_figures.py     # results/figures/*.png
python scripts/download_cookie_cats.py && python experiments/cookie_cats.py
python experiments/build_report.py     # results/summary.md and site/index.html
```

Seeds are fixed per study, so the same numpy version reproduces the same numbers.

## Design decisions

- **Every test returns the same `TestResult` (difference = treatment - control, SE, CI,
  p).** Methods can then be compared side by side in the simulation and the CLI without
  special cases.
- **Welch instead of Student's t-test.** Treatment often changes the variance as well as
  the mean; Welch does not assume equal variances and costs almost nothing when they are
  equal.
- **Ratio metrics are aggregated to the randomization unit and use the delta method.**
  The A/A study shows why: treating sessions as independent gave a 40% false-positive
  rate when the true unit of randomization is the user.
- **CUPED uses one pooled theta for both arms.** Per-arm thetas would let the treatment
  effect leak into the adjustment. Theta is treated as known in the t-test; the A/A rate
  (5.19%) shows that simplification is harmless at 1000 users per arm.
- **SRM threshold of 0.001 instead of 0.05.** The check runs on every experiment, so a
  strict threshold keeps false alarms near 0.1% (measured 0.12%), while real assignment
  bugs at production sample sizes give far smaller p-values.
- **mSPRT with a plug-in variance and a user-chosen tau.** This is the practical version
  from Johari et al. (2017); tau is exposed rather than tuned automatically, and the
  peeking table shows how the choice trades power against early stopping.

## Limitations

- Power and sample-size formulas use the normal approximation with equal allocation only.
- The delta method and CUPED are first-order / large-sample methods; the studies use at
  least 500 users per arm and do not measure behavior at small n.
- The mSPRT guarantee is asymptotic because the variance is estimated; the simulations
  use normal data with 200+ users per arm at the first look. Heavy-tailed metrics were
  not studied for the sequential case.
- CUPED is implemented for mean metrics only (not for ratio or proportion metrics via the
  CLI). No stratification, no heterogeneous-effect analysis, no Bayesian methods.
- The Cookie Cats data has no pre-period covariate or timestamps, so CUPED and sequential
  testing are demonstrated only on simulated data.
- Simulated data-generating processes (lognormal spend, Poisson-Gamma sessions) are
  choices made for this study, not fitted to any real product.

## Data

Cookie Cats A/B test data: published by DataCamp for the project "Mobile Games A/B
Testing with Cookie Cats" (Rasmus Baath), with data from Tactile Entertainment. We found
no stated license or redistribution terms, so the file is **not** committed;
`scripts/download_cookie_cats.py` downloads it from a public GitHub copy
(https://github.com/0zz10/CookieCats-AB-Testing, file `datasets/cookie_cats.csv`) and
verifies its SHA-256.

## References

- Deng, Xu, Kohavi, Walker (2013). Improving the Sensitivity of Online Controlled
  Experiments by Utilizing Pre-Experiment Data. WSDM.
- Deng, Knoblich, Lu (2018). Applying the Delta Method in Metric Analytics. KDD.
- Johari, Koomen, Pekelis, Walsh (2017). Peeking at A/B Tests: Why it matters, and what
  to do about it. KDD.
- Fabijan et al. (2019). Diagnosing Sample Ratio Mismatch in Online Controlled
  Experiments. KDD.

## License

MIT, copyright 2026 Yunlong Lu. See [LICENSE](LICENSE).
