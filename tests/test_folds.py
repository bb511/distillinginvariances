from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.components.folds import fold_indices, holdout_indices
from src.data.components.teacher_cache import TeacherCache, write_cache
from src.vocabulary import MAIN_FOLD
from tests.conftest import tiny_jets


def _mass_squared(x: torch.Tensor) -> torch.Tensor:
    """E^2 - |p|^2 per constituent, in GeV^2, of the tiny jets' [px, py, pz, e]."""
    return x[..., 3] ** 2 - (x[..., :3] ** 2).sum(-1)


def _cache(path: Path, fold: int | str, regime: str = "canonical") -> str:
    """A cache of the tiny pool whose hint i names pool row i."""
    rows = np.arange(40, dtype=np.float32)[:, None]
    meta = {"dataset": "tiny", "regime": regime, "fold": fold, "n_folds": 2}
    empty = (np.zeros((1, 3)), np.zeros((1, 1)))
    write_cache(path, TeacherCache(np.repeat(rows, 3, 1), rows, *empty, meta))
    return str(path)


def test_the_folds_partition_the_pool_and_follow_the_seed() -> None:
    held_out = [fold_indices(100, 5, i, 42)[1] for i in range(5)]
    train, val = fold_indices(100, 5, 0, 42)

    assert sorted(torch.cat(held_out).tolist()) == list(range(100))
    assert sorted(train.tolist() + val.tolist()) == list(range(100))
    assert torch.equal(val, fold_indices(100, 5, 0, 42)[1])
    assert not torch.equal(val, fold_indices(100, 5, 0, 43)[1])


def test_the_holdout_partitions_the_pool_and_follows_the_seed() -> None:
    train, val = holdout_indices(100, 20, 42)

    assert (len(train), len(val)) == (80, 20)
    assert sorted(train.tolist() + val.tolist()) == list(range(100))
    assert torch.equal(val, holdout_indices(100, 20, 42)[1])
    assert not torch.equal(val, holdout_indices(100, 20, 43)[1])


def test_the_main_run_serves_the_main_split_whatever_n_folds_says(
    tmp_path: Path,
) -> None:
    datamodule = tiny_jets(tmp_path, fold_idx=None)
    pool_x, pool_y, _ = datamodule.split("pool")

    assert torch.equal(datamodule.split("train")[0], pool_x[:30])
    assert torch.equal(datamodule.split("fold")[1], pool_y[30:])
    datamodule.set_fold(1)
    assert len(datamodule.split("fold")[0]) == 20
    datamodule.set_fold(None)
    assert torch.equal(datamodule.split("fold")[0], pool_x[30:])


def test_the_main_run_aligns_a_main_cache_and_refuses_a_fold_cache(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path / "main.npz", MAIN_FOLD)
    served = tiny_jets(tmp_path, fold_idx=None, teacher_cache=cache)
    x, y, logits, hints = next(iter(served.val_dataloader()))
    index = hints[:, 0].long()

    assert torch.equal(index, torch.arange(30, 38))
    assert torch.equal(x, served.split("pool")[0][index])
    with pytest.raises(ValueError, match="does not match"):
        tiny_jets(tmp_path, fold_idx=None, teacher_cache=_cache(tmp_path / "f.npz", 1))


def test_the_augmented_regime_trains_on_canonical_rows_and_validates_as_transformed(
    tmp_path: Path,
) -> None:
    augmented = tiny_jets(tmp_path, regime="augmented")
    transformed = tiny_jets(tmp_path, regime="transformed")
    canonical = tiny_jets(tmp_path)

    for split in ("fold", "pool", "test"):
        for ours, theirs in zip(augmented.split(split), transformed.split(split)):
            assert torch.equal(ours, theirs)
    assert torch.equal(augmented.split("train")[0], canonical.split("train")[0])
    assert augmented.transform_matrices("train") is None


def test_each_augmented_loader_draws_afresh_and_keeps_the_mass(tmp_path: Path) -> None:
    augmented = tiny_jets(tmp_path, regime="augmented")
    torch.manual_seed(0)
    first, second = (next(iter(augmented.train_dataloader())) for _ in range(2))
    reference = next(iter(tiny_jets(tmp_path).train_dataloader()))

    assert not torch.allclose(first[0], second[0])
    assert not torch.allclose(first[0], reference[0])
    for batch in (first, second):
        assert torch.equal(batch[1], reference[1])
        assert torch.allclose(
            _mass_squared(batch[0]), _mass_squared(reference[0]), rtol=1e-3
        )


def test_only_the_augmented_training_loader_transforms(tmp_path: Path) -> None:
    augmented = tiny_jets(tmp_path, regime="augmented")
    transformed = tiny_jets(tmp_path, regime="transformed")

    assert augmented.train_dataloader().dataset.transform is not None
    assert augmented.val_dataloader().dataset.transform is None
    assert transformed.train_dataloader().dataset.transform is None


def test_the_augmented_regime_reads_the_canonical_cache(tmp_path: Path) -> None:
    canonical = _cache(tmp_path / "canonical.npz", 1)
    augmented = _cache(tmp_path / "augmented.npz", 1, regime="augmented")
    served = tiny_jets(tmp_path, regime="augmented", teacher_cache=canonical)
    x, y, logits, hints = next(iter(served.val_dataloader()))

    assert torch.equal(x, served.split("pool")[0][hints[:, 0].long()])
    with pytest.raises(ValueError, match="does not match"):
        tiny_jets(tmp_path, regime="augmented", teacher_cache=augmented)
