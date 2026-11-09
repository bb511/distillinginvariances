import numpy as np
import torch

from src.metrics.decomposition import (
    decomposition_terms,
    error_agreement,
    sample_columns,
    summarise_columns,
)
from src.metrics.evaluate import Probe, evaluate_checkpoint
from src.utils.bootstrap import resampled_column_means
from tests.test_evaluate import StubDataModule, StubModule, SumPooled

# The float64 sqrt(JSD) is resolved to about 1e-8 (src/metrics/fidelity.py).
ROUNDING = 1e-7
# A float32 sum over the slots in another order moves the stub teacher's logits by about
# 1e-6, which its sqrt(JSD) sees.
FLOAT32 = 1e-5


def _arrays(seed: int, invariant_teacher: bool) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = [torch.randn(500, 4, generator=generator) * 2.0 for _ in range(4)]
    arrays = dict(zip(("student", "teacher", "student_g", "teacher_g"), logits))
    if invariant_teacher:
        arrays["teacher_g"] = arrays["teacher"]
    return arrays | {"targets": torch.randint(0, 4, (500,), generator=generator)}


def test_the_bounds_hold_per_sample_in_both_metrics() -> None:
    arrays = _arrays(0, invariant_teacher=False)

    for metric in ("top1", "sqrtjsd"):
        terms = decomposition_terms(arrays, (1.5, 0.7), metric)
        assert bool((terms["slack"] >= -ROUNDING).all())
        assert bool((terms["lower"] <= terms["delta_s"] + ROUNDING).all())


def test_a_student_equal_to_its_teacher_has_no_slack() -> None:
    arrays = _arrays(1, invariant_teacher=False)
    arrays["student"], arrays["student_g"] = arrays["teacher"], arrays["teacher_g"]

    terms = decomposition_terms(arrays, (1.3, 1.3), "sqrtjsd")

    assert float(terms["eps_on"].abs().max()) == 0.0
    assert float(terms["slack"].abs().max()) == 0.0


def test_the_student_repeats_only_the_errors_it_shares() -> None:
    eye = torch.eye(3)
    arrays = {
        "targets": torch.tensor([0, 0, 1]),
        "teacher": eye[[1, 0, 2]],
        "student": eye[[1, 0, 1]],
        "teacher_g": eye[[2, 0, 2]],
        "student_g": eye[[0, 0, 2]],
    }

    flags = error_agreement(arrays)

    assert flags["teacher_wrong"].tolist() == [1.0, 0.0, 1.0]
    assert flags["repeat"].tolist() == [1.0, 0.0, 0.0]
    assert flags["repeat_g"].tolist() == [0.0, 0.0, 1.0]


def test_the_summary_brackets_each_mean_and_forms_the_ratios() -> None:
    columns = sample_columns(_arrays(2, invariant_teacher=True), (1.0, 1.0))

    stats = {row["statistic"]: row for row in summarise_columns(columns, 200, 3)}

    assert stats["sqrtjsd/delta_t"]["mean"] == 0.0
    assert stats["sqrtjsd/delta_t"]["ci_high"] == 0.0
    for row in stats.values():
        assert row["ci_low"] <= row["mean"] <= row["ci_high"]
    share = columns["errors/repeat"].mean() / columns["errors/teacher_wrong"].mean()
    assert abs(stats["err_agree"]["mean"] - share) < 1e-12


def test_the_resampled_means_are_reproducible_and_centred() -> None:
    values = np.random.default_rng(0).normal(size=(2000, 2))

    first = resampled_column_means(values, 300, seed=4)

    assert np.array_equal(first, resampled_column_means(values, 300, seed=4))
    assert np.allclose(first.mean(0), values.mean(0), atol=0.01)


def test_the_decomposition_of_an_exactly_invariant_teacher() -> None:
    probes = [Probe("permutation", 2, None)]
    evaluation = evaluate_checkpoint(
        StubModule(0), SumPooled(), StubDataModule(), "test", probes, 7
    )

    arrays = evaluation.samples(0)
    for metric in ("top1", "sqrtjsd"):
        terms = decomposition_terms(arrays, evaluation.temperatures, metric)
        assert float(terms["delta_t"].max()) < FLOAT32
        assert bool((terms["slack"] >= -ROUNDING).all())
        assert float(terms["delta_s"].max()) > 0.0
