import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit import sample_size_proportions, two_proportion_ztest, welch_ttest
from abkit.analysis import analyze, resolve_kind
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
    with pytest.raises(ValueError, match="CUPED"):
        analyze(df, "variant", ["converted"], covariate="pre_spend")


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
    assert main(["power", "--mde", "0.01"]) == 2
