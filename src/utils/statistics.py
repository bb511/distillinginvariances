"""Tests over cross-validation folds whose training sets overlap.

The folds of a K-fold split share most of their training rows, so per-fold differences
are correlated and the plain t-test over them is anti-conservative. The corrected
resampled t-test (Nadeau and Bengio, doi:10.1023/A:1024068626366, eq. 22) inflates the
variance of the mean from s^2 / k to (1 / k + n_test / n_train) s^2, with n_test / n_train
= 1 / (K - 1) for K folds, and keeps k - 1 degrees of freedom.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats

_SPREAD = ("se", "t", "p_value", "ci_low", "ci_high")


def corrected_t(
    deltas: Sequence[float], ratio: float, ci: float = 0.95
) -> dict[str, float]:
    """The mean of per-fold *deltas* with its corrected standard error, t statistic,
    two-sided p-value and t interval; NaN where fewer than two folds give no spread.

    :param ratio: n_test / n_train of one fold.
    """
    sample = np.asarray(deltas, dtype=float)
    k = sample.size
    if k < 2:
        mean = float(sample.mean()) if k else np.nan
        return {"mean": mean, "df": k - 1} | dict.fromkeys(_SPREAD, np.nan)
    mean, se = float(sample.mean()), _corrected_se(sample, ratio)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = float(np.divide(mean, se))
    half = float(stats.t.ppf(0.5 + ci / 2.0, k - 1)) * se
    p = float(2.0 * stats.t.sf(abs(t), k - 1))
    cells = {"mean": mean, "se": se, "t": t, "df": k - 1, "p_value": p}
    return cells | {"ci_low": mean - half, "ci_high": mean + half}


def equivalence_p(deltas: Sequence[float], ratio: float, margin: float) -> float:
    """The TOST p-value for a mean within (-margin, margin) (Schuirmann 1987).

    It is the larger p-value of the two one-sided corrected t-tests. Equivalence holds
    at level alpha when it is below alpha, as when the (1 - 2 alpha) interval lies
    inside the margins.
    """
    sample = np.asarray(deltas, dtype=float)
    k = sample.size
    if k < 2:
        return float("nan")
    se = _corrected_se(sample, ratio)
    lower = stats.t.sf((sample.mean() + margin) / se, k - 1)
    upper = stats.t.cdf((sample.mean() - margin) / se, k - 1)
    return float(max(lower, upper))


def _corrected_se(sample: np.ndarray, ratio: float) -> float:
    return float(np.sqrt((1.0 / sample.size + ratio) * sample.var(ddof=1)))


def transfer_fraction(student: float, baseline: float, ceiling: float) -> float:
    """(student - baseline) / (ceiling - baseline): 0 at the plain student, 1 at the
    augmented ceiling, for an agreement or a distance alike; NaN where the ceiling equals
    the baseline."""
    gap = ceiling - baseline
    return (student - baseline) / gap if gap != 0 else float("nan")
