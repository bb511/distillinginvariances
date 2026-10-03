from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.components.teacher_cache import TeacherCache, write_cache
from src.data.jets_datamodule import JetsDataModule
from tests.conftest import CACHE_DIR, needs_record

pytestmark = needs_record


def _datamodule(**overrides) -> JetsDataModule:
    settings = {
        "revision": "fc7104f60a45d528ab597b220d5a7cd8e31cfff8",
        "cache_dir": CACHE_DIR,
        "batch_size": 256,
        "num_workers": 0,
        "pin_memory": False,
    }
    datamodule = JetsDataModule(**(settings | overrides))
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


@pytest.fixture(scope="module")
def canonical() -> JetsDataModule:
    return _datamodule()


def test_batches_carry_the_full_width_and_the_test_split_is_never_folded(
    canonical: JetsDataModule,
) -> None:
    x, y = next(iter(canonical.val_dataloader()))
    pool, test, held_out = (canonical.split(s)[0] for s in ("pool", "test", "fold"))

    assert (
        x.shape == (256, 50, 16)
        and y.shape == (256, 5)
        and torch.equal(y.sum(1), torch.ones(256))
    )
    assert len(canonical.split("train")[0]) + len(held_out) == len(pool) != len(test)
    canonical.set_fold(1)
    assert torch.equal(canonical.split("test")[0], test)
    assert not torch.equal(canonical.split("fold")[0][:64], held_out[:64])
    canonical.set_fold(0)


def test_the_main_run_validates_on_the_record_validation_split(
    canonical: JetsDataModule,
) -> None:
    canonical.set_fold(None)
    train, held_out = canonical.split("train")[0], canonical.split("fold")[0]

    assert (len(train), len(held_out)) == (496_000, 124_000)
    assert torch.equal(held_out, canonical._load("validation")[0])
    canonical.set_fold(0)


def test_the_eval_batches_walk_the_pool_once_in_stored_order(
    canonical: JetsDataModule,
) -> None:
    pool = canonical.split("pool")[0]
    batches = list(canonical.eval_batches("pool", 100_000))

    assert sum(len(x) for x, _, _ in batches) == len(pool)
    assert torch.equal(batches[0][0], pool[:100_000]) and torch.equal(
        batches[-1][0][-1], pool[-1]
    )


def test_two_folds_of_the_transformed_regime_see_the_same_rows(tmp_path: Path) -> None:
    settings = {
        "regime": "transformed",
        "group": "permutation",
        "transformed_dir": str(tmp_path),
    }
    pool = _datamodule(fold_idx=0, **settings).split("pool")[0]

    assert torch.equal(pool, _datamodule(fold_idx=1, **settings).split("pool")[0])


def test_the_teacher_cache_reaches_the_batches_aligned(
    canonical: JetsDataModule, tmp_path: Path
) -> None:
    n_pool = len(canonical.split("pool")[0])
    rows = np.arange(n_pool, dtype=np.float32)[:, None]  # hint i names pool row i
    meta = {"dataset": "jets", "regime": "canonical", "fold": 1, "n_folds": 5}
    write_cache(
        tmp_path / "c.npz",
        TeacherCache(
            np.repeat(rows, 5, 1), rows, np.zeros((1, 5)), np.zeros((1, 1)), meta
        ),
    )

    served = _datamodule(teacher_cache=str(tmp_path / "c.npz"))
    x, y, logits, hints = next(iter(served.val_dataloader()))
    index = hints[:, 0].long()

    assert torch.equal(x, served.split("pool")[0][index]) and torch.equal(
        y, served.split("pool")[1][index]
    )
    assert torch.equal(index, served._fold_indices()[1][:256])
