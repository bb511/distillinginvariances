from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNIST_MEAN, MNIST_STD, MNISTDataModule
from tests.conftest import needs_record

pytestmark = needs_record
REVISION = "77f3279092a1c1579b2250db8eafed0ad422088c"


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    directory = str(tmp_path_factory.mktemp("mnist"))
    MNISTDataModule(revision=REVISION, cache_dir=directory).prepare_data()
    return directory


def _datamodule(cache_dir: str, **overrides) -> MNISTDataModule:
    datamodule = MNISTDataModule(revision=REVISION, cache_dir=cache_dir, **overrides)
    datamodule.setup()
    return datamodule


def test_the_digits_arrive_normalised_folded_and_with_the_test_split_whole(
    cache_dir: str,
) -> None:
    datamodule = _datamodule(cache_dir)
    x, y, _ = datamodule.split("pool")

    assert x.shape == (60_000, 1, 28, 28) and y.shape == (60_000, 10)
    assert x.min().item() == pytest.approx(-MNIST_MEAN / MNIST_STD, abs=1e-5)
    assert x.max().item() == pytest.approx((1 - MNIST_MEAN) / MNIST_STD, abs=1e-5)
    assert (len(datamodule.split("train")[0]), len(datamodule.split("fold")[0])) == (
        48_000,
        12_000,
    )
    assert len(datamodule.split("test")[0]) == 10_000


def test_the_transformed_regime_moves_the_pixels_without_losing_any(
    cache_dir: str, tmp_path: Path
) -> None:
    before = _datamodule(cache_dir).split("pool")[0][:256].flatten(1)
    after = (
        _datamodule(cache_dir, regime="transformed", transformed_dir=str(tmp_path))
        .split("pool")[0][:256]
        .flatten(1)
    )

    assert not torch.equal(before, after)
    assert torch.equal(before.sort(1).values, after.sort(1).values)
