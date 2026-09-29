import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit import sample_size_proportions, two_proportion_ztest, welch_ttest
from abkit.analysis import analyze, compare, resolve_kind
from abkit.cli import main


@pytest.fixture
def experiment_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(11)
    n = 600
    variant = np.repeat(["control", "treat"], n)
    pre = rng.normal(10, 2, 2 * n)
    spend = pre + rng.normal(0, 1, 2 * n) + np.where(variant == "treat", 0.3, 0.0)
    converted = rng.random(2 * n) < np.where(variant == "treat", 0.14, 0.10)
    sessions = rng.integers(1, 6, 2 * n)
    df = pd.DataFrame(
        {
            "variant": variant,
            "spend": spend,
            "pre_spend": pre,
            "converted": converted,
            "sessions": sessions,
        }
    )
    path = tmp_path / "exp.csv"
    df.to_csv(path, index=False)
    return path


def test_resolve_kind() -> None:
    assert resolve_kind(pd.Series([True, False]), "auto", None) == "proportion"
    assert resolve_kind(pd.Series([0, 1, 1]), "auto", None) == "proportion"
    assert resolve_kind(pd.Series([0.5, 1.0]), "auto", None) == "mean"
    assert resolve_kind(pd.Series([0.5, 1.0]), "auto", "sessions") == "ratio"
    assert resolve_kind(pd.Series([0, 1]), "mean", None) == "mean"


def test_analyze_uses_right_tests_and_correction(experiment_csv: Path) -> None:
    df = pd.read_csv(experiment_csv)
    report = analyze(df, "variant", ["spend", "converted"], correction="holm")
    by_metric = {c.metric: c for c in report.comparisons}
    assert by_metric["spend"].result.method == "welch_t"
    assert by_metric["converted"].result.method == "two_proportion_z"

    c, t = df[df.variant == "control"], df[df.variant == "treat"]
    assert by_metric["spend"].result.p_value == pytest.approx(welch_ttest(c.spend, t.spend).p_value)
    conv = two_proportion_ztest(c.converted.sum(), len(c), t.converted.sum(), len(t))
    assert by_metric["converted"].result.p_value == pytest.approx(conv.p_value)
    # Holm with two tests: the smaller p-value is doubled.
    smallest = min(report.comparisons, key=lambda x: x.result.p_value)
    assert smallest.adjusted_p_value == pytest.approx(min(1.0, 2 * smallest.result.p_value))
    assert report.arm_sizes == {"control": 600, "treat": 600}
    assert not report.srm.mismatch


def test_analyze_errors(experiment_csv: Path) -> None:
    df = pd.read_csv(experiment_csv)
    with pytest.raises(ValueError, match="columns not found"):
        analyze(df, "variant", ["nope"])
    with pytest.raises(ValueError, match="control arm"):
        analyze(df, "variant", ["spend"], control="missing")


def test_compare_rejects_covariate_for_non_mean_metrics(experiment_csv: Path) -> None:
    df = pd.read_csv(experiment_csv)
    c, t = df[df.variant == "control"], df[df.variant == "treat"]
    with pytest.raises(ValueError, match="CUPED"):
        compare(c, t, "converted", "proportion", covariate="pre_spend")
    with pytest.raises(ValueError, match="CUPED"):
        compare(c, t, "spend", "ratio", denominator="sessions", covariate="pre_spend")


def test_covariate_applies_to_mean_metrics_only(experiment_csv: Path) -> None:
    df = pd.read_csv(experiment_csv)
    report = analyze(df, "variant", ["spend", "converted"], covariate="pre_spend")
    methods = {c.metric: c.result.method for c in report.comparisons}
    assert methods == {"spend": "cuped_welch_t", "converted": "two_proportion_z"}
    assert any("covariate not applied to 'converted'" in n for n in report.notes)

    ratio = analyze(df, "variant", ["spend"], denominator="sessions", covariate="pre_spend")
    plain = analyze(df, "variant", ["spend"], denominator="sessions")
    assert ratio.comparisons[0].result.method == "delta_ratio_z"
    assert ratio.comparisons[0].result.p_value == plain.comparisons[0].result.p_value
    assert any("ratio metric" in n for n in ratio.notes)


def _aa_table(seed: int, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "variant": np.repeat(["a", "b"], n),
            "converted": (rng.random(2 * n) < 0.10).astype(float),
            "spend": rng.lognormal(0, 1, 2 * n),
            "sessions": rng.integers(1, 5, 2 * n).astype(float),
        }
    )


def test_missing_binary_values_are_dropped_not_counted_as_zero() -> None:
    """Regression: NaN used to count as a non-conversion, turning A/A into a false positive."""
    df = _aa_table(3)
    control_rows = df.index[df.variant == "a"]
    df.loc[control_rows[:1000], "converted"] = np.nan
    report = analyze(df, "variant", ["converted"], correction="none")
    comp = report.comparisons[0]
    ctrl = df[df.variant == "a"].converted.dropna()
    treat = df[df.variant == "b"].converted
    assert comp.result.control_value == pytest.approx(ctrl.mean())
    assert (comp.n_control, comp.n_treatment) == (1000, 2000)
    assert (comp.dropped_control, comp.dropped_treatment) == (1000, 0)
    expected = two_proportion_ztest(ctrl.sum(), ctrl.size, treat.sum(), treat.size)
    assert comp.result.p_value == pytest.approx(expected.p_value)
    assert comp.result.p_value > 0.05
    # The SRM check still uses assigned users, not users with a non-missing metric.
    assert report.arm_sizes == {"a": 2000, "b": 2000}
    assert any("1000 rows with missing values" in n for n in report.notes)


def test_missing_mean_and_ratio_values_give_clean_results() -> None:
    df = _aa_table(4)
    df.loc[[0, 5, 2500], "spend"] = np.nan
    df.loc[[7, 3000], "sessions"] = np.nan
    report = analyze(df, "variant", ["spend"], correction="holm")
    comp = report.comparisons[0]
    clean = df.dropna(subset=["spend"])
    ref = welch_ttest(clean[clean.variant == "a"].spend, clean[clean.variant == "b"].spend)
    assert comp.result.p_value == pytest.approx(ref.p_value)
    assert 0 <= comp.adjusted_p_value <= 1

    ratio = analyze(df, "variant", ["spend"], denominator="sessions").comparisons[0]
    # Rows missing either the numerator or the denominator are dropped.
    assert (ratio.dropped_control, ratio.dropped_treatment) == (3, 2)
    assert np.isfinite(ratio.result.p_value)


def test_split_is_matched_by_arm_label() -> None:
    df = pd.DataFrame({"variant": ["a"] * 900 + ["b"] * 100, "y": np.arange(1000.0)})
    report = analyze(df, "variant", ["y"], control="b", expected_split={"a": 90, "b": 10})
    assert not report.srm.mismatch
    assert report.srm.p_value == pytest.approx(1.0)
    assert report.srm.expected == [100.0, 900.0]  # [control b, treatment a]
    with pytest.raises(ValueError, match="every arm"):
        analyze(df, "variant", ["y"], expected_split={"a": 90, "c": 10})


def test_cli_blank_variant_label(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "blank.csv"
    path.write_text("v,m\na,1\n,2\nb,3\na,4\nb,5\na,6\nb,8\n")
    assert main(["analyze", str(path), "--variant-col", "v", "--metric", "m"]) == 0
    out = capsys.readouterr().out
    assert "Arms (v): a=3, b=3" in out
    assert "note: 1 rows with an empty 'v' were dropped" in out


def test_cli_expected_split(experiment_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["analyze", str(experiment_csv), "--variant-col", "variant", "--metric", "spend"]
    assert main([*base, "--expected-split", "treat=50,control=50"]) == 0
    assert "expected control=600.0, treat=600.0" in capsys.readouterr().out
    # Positional weights are ambiguous about which arm they belong to, so they are rejected.
    assert main([*base, "--expected-split", "50,50"]) == 2
    assert "label=weight" in capsys.readouterr().err


def test_cli_malformed_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert main(["analyze", str(empty), "--variant-col", "v", "--metric", "m"]) == 2
    assert "abkit: error" in capsys.readouterr().err
    binary = tmp_path / "latin1.csv"
    binary.write_bytes(b"v,m\n\xff\xfe,1\n")
    assert main(["analyze", str(binary), "--variant-col", "v", "--metric", "m"]) == 2


def test_cli_table_output(experiment_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "analyze",
            str(experiment_csv),
            "--variant-col",
            "variant",
            "--metric",
            "spend",
            "--covariate",
            "pre_spend",
            "--control",
            "control",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "cuped_welch_t" in out
    assert "SRM check" in out and "-> ok" in out


def test_cli_json_ratio(experiment_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "analyze",
            str(experiment_csv),
            "--variant-col",
            "variant",
            "--metric",
            "spend",
            "--denominator",
            "sessions",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["comparisons"][0]["method"] == "delta_ratio_z"
    assert payload["srm"]["mismatch"] is False


def test_cli_reports_errors(experiment_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["analyze", str(experiment_csv), "--variant-col", "variant", "--metric", "zzz"])
    assert code == 2
    assert "columns not found" in capsys.readouterr().err


def test_cli_power(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["power", "--baseline", "0.1", "--mde", "0.01"]) == 0
    out = capsys.readouterr().out
    assert f"{sample_size_proportions(0.1, 0.01)} users per arm" in out
    assert "conversion 0.1 -> 0.11:" in out
    assert main(["power", "--baseline", "0.1", "--mde", "0.02"]) == 0
    assert "0.1 -> 0.12:" in capsys.readouterr().out
    assert main(["power", "--mde", "0.01"]) == 2
