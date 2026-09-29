"""abkit: statistical tests for online A/B experiments."""

from abkit.cuped import cuped_test, cuped_theta
from abkit.means import welch_from_stats, welch_ttest
from abkit.multiple import benjamini_hochberg, holm
from abkit.power import (
    power_per_arm,
    sample_size_means,
    sample_size_per_arm,
    sample_size_proportions,
)
from abkit.proportions import two_proportion_ztest
from abkit.ratio import delta_ratio_test, ratio_estimate
from abkit.result import TestResult
from abkit.sequential import SequentialResult, always_valid_ci, msprt_log_lambda, msprt_monitor
from abkit.srm import SRMResult, srm_check

__version__ = "0.1.0"

__all__ = [
    "SRMResult",
    "SequentialResult",
    "TestResult",
    "always_valid_ci",
    "benjamini_hochberg",
    "cuped_test",
    "cuped_theta",
    "delta_ratio_test",
    "holm",
    "msprt_log_lambda",
    "msprt_monitor",
    "power_per_arm",
    "ratio_estimate",
    "sample_size_means",
    "sample_size_per_arm",
    "sample_size_proportions",
    "srm_check",
    "two_proportion_ztest",
    "welch_from_stats",
    "welch_ttest",
]
