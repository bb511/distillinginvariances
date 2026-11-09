from pathlib import Path

import numpy as np
import torch

from src.metrics.coverage import auc, auc_interval, fit_classifier, scores
from src.metrics.magnitude import bin_masks, with_derived
from tests.conftest import compose_root

SETTINGS = {"hidden": [16, 16], "epochs": 30, "lr": 1e-2, "batch_size": 128}


def _sorted_and_shuffled(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 8, generator=generator).sort(dim=1, descending=True).values
    order = torch.rand(n, 8, generator=generator).argsort(dim=1)
    return x, x.gather(1, order)


def test_the_auc_counts_ties_half() -> None:
    scores_ = torch.tensor([0.1, 0.2, 0.3])

    assert auc(scores_, scores_ + 1.0) == 1.0
    assert auc(scores_, scores_) == 0.5
    assert auc(torch.zeros(4), torch.zeros(4)) == 0.5


def test_the_classifier_recognises_a_shuffled_sorted_row() -> None:
    net = fit_classifier(*_sorted_and_shuffled(1000, 0), **SETTINGS, seed=1)
    x, x_t = _sorted_and_shuffled(1000, 2)

    value, low, high = auc_interval(scores(net, x), scores(net, x_t), 100, 3)

    assert value > 0.9 and low <= value <= high


def test_the_classifier_cannot_tell_an_exchangeable_row_from_its_shuffle() -> None:
    generator = torch.Generator().manual_seed(5)
    x = torch.rand(3000, 8, generator=generator)
    x_t = x.gather(1, torch.rand(3000, 8, generator=generator).argsort(dim=1))
    net = fit_classifier(x[:1000], x_t[:1000], **SETTINGS, seed=1)

    value = auc(scores(net, x[1000:]), scores(net, x_t[1000:]))

    assert 0.42 < value < 0.58


def test_the_bins_are_half_open_but_the_last() -> None:
    sizes = with_derived(
        {"shift_row": np.array([-3.0, 0.0]), "shift_col": np.array([4.0, 1.0])}
    )

    masks = bin_masks(np.array([0.0, 0.5, 1.0, 2.0]), [0, 1, 2])

    assert sizes["shift_dist"].tolist() == [5.0, 1.0]
    assert sizes["abs_shift_row"].tolist() == [3.0, 0.0]
    assert [mask.tolist() for _, _, mask in masks] == [
        [True, True, False, False],
        [False, False, True, True],
    ]


def test_the_coverage_config_composes(tmp_path: Path) -> None:
    cfg = compose_root("coverage.yaml", "experiment=mnist_cnn", "trainer=cpu")

    assert (cfg.task_name, cfg.probe_group, cfg.eval_data) == ("coverage", None, None)
    assert cfg.probe.n_draws == 1 and cfg.seed == 12345
    assert cfg.results.path.endswith("coverage.csv")
