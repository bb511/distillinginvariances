from pathlib import Path

import pytest
import torch

from src.data.components.transformed import cached_transform, transformed_split
from tests.conftest import FOUR_MOMENTUM, jets

KEY = {"group": "lorentz_iso", "seed": 1, "n": 6, "split": "pool"}


def test_the_draw_is_fixed_by_its_seed_and_leaves_the_ambient_rng_alone() -> None:
    images = torch.rand(16, 1, 8, 8)
    torch.manual_seed(11)
    expected = torch.rand(4)

    torch.manual_seed(11)
    first, matrices = transformed_split(images, "translation", 1)

    assert matrices is None
    assert torch.equal(first, transformed_split(images, "translation", 1)[0])
    assert not torch.equal(first, transformed_split(images, "translation", 2)[0])
    assert torch.equal(torch.rand(4), expected)


def test_the_cache_is_reused_under_its_key_and_rebuilt_under_another(
    tmp_path: Path,
) -> None:
    batch = jets()

    def build():
        return transformed_split(batch, "lorentz_iso", 1, FOUR_MOMENTUM)

    def refuse():
        pytest.fail("the cache should have been reused")

    x, matrices = cached_transform(str(tmp_path), KEY, build)
    again = cached_transform(str(tmp_path), KEY, refuse)

    assert matrices.shape == (6, 4, 4)
    assert torch.equal(x, again[0]) and torch.equal(matrices, again[1])
    with pytest.raises(pytest.fail.Exception):
        cached_transform(str(tmp_path), KEY | {"seed": 2}, refuse)
