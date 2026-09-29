"""Result container shared by every two-sample test in abkit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TestResult:
    """Outcome of comparing a treatment arm (B) against a control arm (A).

    ``estimate`` is always the absolute difference ``treatment - control`` so that
    results from different methods can be placed side by side.
    """

    __test__ = False  # stop pytest from trying to collect this class

    method: str
    control_value: float
    treatment_value: float
    estimate: float
    std_error: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def relative_lift(self) -> float:
        """Difference as a fraction of the control value (nan if control is 0)."""
        if self.control_value == 0:
            return float("nan")
        return self.estimate / self.control_value

    def to_dict(self) -> dict[str, float | str | bool]:
        out: dict[str, float | str | bool] = dict(asdict(self))
        out["significant"] = self.significant
        out["relative_lift"] = self.relative_lift
        return out
