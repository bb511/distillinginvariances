"""The fidelity-invariance decomposition of one checkpoint on its first probe draw.

For a metric d on predictive distributions (top-1 disagreement, or sqrt(JSD) after each
model's temperature), with S the student, T the teacher and g.x the first draw, the
triangle inequality bounds the student's non-invariance per sample:

    abs(eps_off - eps_on) - delta_T  <=  delta_S  <=  eps_on + eps_off + delta_T

with eps_on = d(S(x), T(x)), eps_off = d(S(g.x), T(g.x)), delta_S = d(S(x), S(g.x)) and
delta_T = d(T(x), T(g.x)). The slack is the upper bound minus delta_S; 2 eps_on + delta_T
is the transformed regime's ceiling on delta_S, which holds where eps_off stays close to
eps_on. The bounds are an identity, so their content lies in where delta_S sits and in
the slack (archive/research_2026-11-08, "The top three", item 1).
"""

from typing import Any

import numpy as np
import torch

from src.metrics.fidelity import sqrt_jsd, top1_agreement
from src.utils.bootstrap import percentile_interval, resampled_column_means

METRICS = ("top1", "sqrtjsd")
LOGITS = ("student", "teacher", "student_g", "teacher_g")
# The student's agreement with the teacher's errors, at x and at g.x: the mean of the
# first column over the mean of the second, both per sample.
RATIOS = {
    "err_agree": ("errors/repeat", "errors/teacher_wrong"),
    "err_agree_g": ("errors/repeat_g", "errors/teacher_wrong_g"),
}


def decomposition_terms(
    arrays: dict[str, Any], temperatures: tuple[float, float], metric: str
) -> dict[str, torch.Tensor]:
    """Per-sample terms of the decomposition in *metric*, each ``(samples,)`` float64.

    :param arrays: a dump's arrays, the four logit sets `LOGITS` among them.
    :param temperatures: (student, teacher), as the dump's meta holds them; top-1
        ignores them.
    """
    eps_on, eps_off, delta_s, delta_t = _distances(arrays, temperatures, metric)
    return {
        "eps_on": eps_on,
        "eps_off": eps_off,
        "delta_s": delta_s,
        "delta_t": delta_t,
        "slack": eps_on + eps_off + delta_t - delta_s,
        "lower": (eps_off - eps_on).abs() - delta_t,
        "ceiling": 2.0 * eps_on + delta_t,
    }


def error_agreement(arrays: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Per sample, whether the teacher errs at x and at g.x, and whether the student
    then repeats the teacher's label; the labels are the same at x and g.x."""
    targets = torch.as_tensor(arrays["targets"]).long()
    label = {name: torch.as_tensor(arrays[name]).argmax(-1) for name in LOGITS}
    wrong, wrong_g = label["teacher"].ne(targets), label["teacher_g"].ne(targets)
    repeat = label["student"].eq(label["teacher"]) & wrong
    repeat_g = label["student_g"].eq(label["teacher_g"]) & wrong_g
    cells = {"teacher_wrong": wrong, "repeat": repeat}
    cells |= {"teacher_wrong_g": wrong_g, "repeat_g": repeat_g}
    return {name: value.double() for name, value in cells.items()}


def sample_columns(
    arrays: dict[str, Any], temperatures: tuple[float, float]
) -> dict[str, np.ndarray]:
    """Every per-sample column the decomposition table averages, named
    ``<metric>/<term>`` and ``errors/<flag>``."""
    columns = {
        f"{metric}/{name}": value
        for metric in METRICS
        for name, value in decomposition_terms(arrays, temperatures, metric).items()
    }
    columns |= {f"errors/{n}": v for n, v in error_agreement(arrays).items()}
    return {name: value.numpy() for name, value in columns.items()}


def summarise_columns(
    columns: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 0
) -> list[dict[str, Any]]:
    """The mean of every column and of every `RATIOS` entry, with a percentile 95 %
    interval from resampling the samples.

    :return: one dict per statistic: ``statistic``, ``mean``, ``ci_low``, ``ci_high``.
    """
    names = list(columns)
    values = np.stack([columns[name] for name in names], axis=1)
    point, replicates = values.mean(0), resampled_column_means(values, n_boot, seed)
    stats = {name: (point[i], replicates[:, i]) for i, name in enumerate(names)}
    with np.errstate(divide="ignore", invalid="ignore"):
        for ratio, (num, den) in RATIOS.items():
            i, j = names.index(num), names.index(den)
            stats[ratio] = (point[i] / point[j], replicates[:, i] / replicates[:, j])
    return [_statistic(name, mean, spread) for name, (mean, spread) in stats.items()]


def _distances(
    arrays: dict[str, Any], temperatures: tuple[float, float], metric: str
) -> tuple[torch.Tensor, ...]:
    """eps_on, eps_off, delta_S and delta_T, each pair under its own models'
    temperatures."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}, got {metric!r}.")
    student, teacher, student_g, teacher_g = (
        torch.as_tensor(arrays[name]) for name in LOGITS
    )
    distance = _top1_distance if metric == "top1" else sqrt_jsd
    s, t = temperatures
    return (
        distance(student, teacher, s, t),
        distance(student_g, teacher_g, s, t),
        distance(student, student_g, s, s),
        distance(teacher, teacher_g, t, t),
    )


def _top1_distance(a: torch.Tensor, b: torch.Tensor, *_: float) -> torch.Tensor:
    return 1.0 - top1_agreement(a, b)


def _statistic(name: str, mean: float, replicates: np.ndarray) -> dict[str, Any]:
    low, high = percentile_interval(replicates)
    interval = {"ci_low": float(low), "ci_high": float(high)}
    return {"statistic": name, "mean": float(mean)} | interval
