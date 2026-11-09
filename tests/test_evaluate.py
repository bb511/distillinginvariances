import math

import torch

from src.metrics.evaluate import Probe, constant_row, evaluate_checkpoint, make_probe
from src.vocabulary import RESULT_COLUMNS

METRIC_COLUMNS = RESULT_COLUMNS[RESULT_COLUMNS.index("accuracy") :]
UNIT_INTERVAL = [
    column
    for column in METRIC_COLUMNS
    if column.startswith(("fid_", "inv_", "tinv_")) and column != "inv_M"
] + ["accuracy", "ece", "mean_confidence"]
PROBES = [Probe("permutation", 3, 1), Probe("permutation", 2, None)]


class StubModule(torch.nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Linear(16, 3)

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class SumPooled(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(5)
        self.net = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.net(x.sum(1)) * 4.0


class StubDataModule:
    group, feature_names, norm_scales = "permutation", None, None

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(11)
        self.batches = [
            (
                torch.randn(8, 4, 4, generator=generator),
                torch.eye(3)[torch.randint(0, 3, (8,), generator=generator)],
                None,
            )
            for _ in range(2)
        ]

    def eval_batches(self, split: str, batch_size: int):
        return iter(self.batches)

    def train_class_frequencies(self) -> torch.Tensor:
        return torch.cat([y for _, y, _ in self.batches]).mean(0)


def _evaluate(module, teacher, probes=PROBES, **kwargs):
    return evaluate_checkpoint(
        module, teacher, StubDataModule(), "fold", probes, 7, **kwargs
    )


def test_every_metric_column_is_filled_within_its_range() -> None:
    evaluation = _evaluate(StubModule(0), StubModule(1))

    for metrics, probe in zip(evaluation.metrics, PROBES):
        assert set(METRIC_COLUMNS) <= set(metrics)
        assert metrics["inv_M"] == probe.n_draws
        assert all(0.0 <= metrics[column] <= 1.0 for column in UNIT_INTERVAL)
        assert 0.0 < metrics["mean_entropy"] <= math.log2(3)
    assert min(evaluation.temperatures) > 0.0
    samples = evaluation.samples(0)
    assert samples["student_g"].shape == (16, 3) and samples["targets"].shape == (16,)
    assert set(samples["mag_moved"]) == {2.0}
    assert set(samples["mag_multiplicity"]) == {4.0}


def test_a_teacher_is_its_own_reference_and_given_temperatures_are_kept() -> None:
    module = StubModule(0)

    fitted = _evaluate(module, module)
    given = _evaluate(module, module, temperatures=(2.0, 3.0))

    assert fitted.temperatures[0] == fitted.temperatures[1]
    metrics = fitted.metrics[0]
    assert (metrics["fid_top1"], metrics["fid_sqrtjsd_g"]) == (1.0, 0.0)
    assert metrics["tinv_sqrtjsd_g"] == metrics["inv_sqrtjsd_g"]
    assert given.temperatures == (2.0, 3.0)
    assert given.metrics[0]["temperature_teacher_ts"] == 3.0


def test_an_exactly_invariant_teacher_has_no_non_invariance() -> None:
    metrics = _evaluate(StubModule(0), SumPooled()).metrics[1]

    assert metrics["tinv_top1_g"] == 1.0 and metrics["tinv_sqrtjsd_g"] < 1e-5
    assert metrics["inv_sqrtjsd_g"] > 1e-3


def test_the_draws_are_fixed_by_the_probe_whatever_runs_beside_it() -> None:
    alone = _evaluate(StubModule(0), StubModule(1), PROBES[:1])
    beside = _evaluate(StubModule(0), SumPooled(), PROBES[::-1])

    assert torch.equal(alone.probed[0]["draws"], beside.probed[1]["draws"])
    assert not torch.equal(alone.probed[0]["draws"][:1], beside.probed[0]["draws"][:1])


def test_a_probe_takes_only_the_knob_of_its_group() -> None:
    settings = {"n_draws": 4, "lorentz_eta_max": 2.0, "max_shift": 3, "n_swaps": None}

    boost, rotation = make_probe("lorentz_iso", settings), make_probe("so3", settings)

    assert (boost.kwargs, boost.tag) == ({"lorentz_eta_max": 2.0}, "lorentz_iso_eta2.0")
    assert (rotation.kwargs, rotation.tag) == ({}, "so3")
    assert make_probe("permutation", settings).tag == "permutation_swapsall"
    assert boost.columns() == {
        "probe_group": "lorentz_iso",
        "probe_eta_max": 2.0,
        "probe_max_shift": None,
        "probe_n_swaps": None,
    }


def test_the_constant_row_scores_the_class_entropy() -> None:
    datamodule = StubDataModule()
    frequencies = datamodule.train_class_frequencies().double()

    row = constant_row(datamodule, "fold")

    assert abs(row["nll"] + float((frequencies * frequencies.log()).sum())) < 1e-9
    assert (row["fid_top1"], row["inv_sqrtjsd"], row["inv_M"]) == (1.0, 0.0, 0)
    assert "tinv_top1_g" not in row
