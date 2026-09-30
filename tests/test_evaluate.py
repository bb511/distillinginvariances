import math

import torch

from src.metrics.evaluate import constant_row, evaluate_checkpoint
from src.vocabulary import RESULT_COLUMNS

METRIC_COLUMNS = RESULT_COLUMNS[RESULT_COLUMNS.index("accuracy") :]
PROBE = {"n_draws": 3, "n_swaps": 1}


class StubModule(torch.nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Linear(16, 3)

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


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


def test_every_metric_column_is_filled_within_its_range() -> None:
    metrics = evaluate_checkpoint(
        StubModule(0), StubModule(1), StubDataModule(), "fold", PROBE
    )

    assert set(METRIC_COLUMNS) <= set(metrics) and metrics["inv_M"] == 3
    assert all(
        0.0 <= metrics[k] <= 1.0
        for k in ("accuracy", "ece", "fid_top1", "fid_1mjsd_g", "inv_1mjsd")
    )
    assert (
        0.0 < metrics["mean_entropy"] <= math.log2(3)
        and metrics["temperature_ts"] > 0.0
    )
    assert metrics["logits"].shape == (16, 3)


def test_a_teacher_is_its_own_fidelity_reference_and_a_given_temperature_is_kept() -> (
    None
):
    module = StubModule(0)
    metrics = evaluate_checkpoint(
        module, module, StubDataModule(), "test", PROBE, temperature_ts=2.0
    )

    assert all(
        abs(metrics[k] - 1.0) < 1e-9
        for k in ("fid_top1", "fid_1mjsd", "fid_top1_g", "fid_1mjsd_g")
    )
    assert metrics["temperature_ts"] == 2.0


def test_the_constant_row_scores_the_class_entropy() -> None:
    datamodule = StubDataModule()
    frequencies = datamodule.train_class_frequencies().double()

    row = constant_row(datamodule, "fold")

    assert abs(row["nll"] + float((frequencies * frequencies.log()).sum())) < 1e-9
    assert (row["fid_top1"], row["inv_1mjsd"], row["inv_M"]) == (1.0, 1.0, 0)
