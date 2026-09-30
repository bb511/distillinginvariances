import pytest
import torch

from src.data.components.augmentation import (
    _apply,
    apply_group,
    apply_permutation,
    apply_translation,
    compute_eta,
    compute_phi,
    compute_pt,
    feature_index,
    make_transform,
    rebase_on_kept_axis,
    translate,
)
from src.vocabulary import GROUPS
from tests.conftest import FOUR_MOMENTUM, RECORD_COLUMNS, jets

REAL = slice(None, 5)


def _mass_squared(batch: torch.Tensor) -> torch.Tensor:
    return batch[..., 3] ** 2 - (batch[..., :3] ** 2).sum(-1)


def test_the_isotropic_boost_reproduces_its_matrices_and_keeps_masses_and_padding() -> (
    None
):
    torch.manual_seed(0)
    batch = jets()

    boosted, boosts = apply_group("lorentz_iso", batch, FOUR_MOMENTUM)
    expected = torch.einsum("bij,bcj->bci", boosts, batch[..., [3, 0, 1, 2]])

    assert torch.allclose(
        boosted[:, REAL][..., [3, 0, 1, 2]], expected[:, REAL], atol=1e-3
    )
    assert torch.allclose(
        _mass_squared(boosted)[:, REAL], _mass_squared(batch)[:, REAL], rtol=1e-4
    )
    assert torch.equal(boosted[:, 5:], torch.zeros_like(boosted[:, 5:]))
    assert not torch.allclose(boosts[0], boosts[1])  # drawn per jet


def test_the_rebase_puts_the_axis_columns_where_the_identity_reproduces_them() -> None:
    batch = jets(columns=16)
    px, py, pz = batch[..., 0], batch[..., 1], batch[..., 2]
    cylindrical = {
        "pt": compute_pt(px, py),
        "eta": compute_eta(px, py, pz),
        "phi": compute_phi(px, py),
    }
    for name, values in cylindrical.items():
        batch[..., feature_index(RECORD_COLUMNS, name)] = values
    batch[:, 5:] = 0.0
    scales = {name: 1.5 for name in RECORD_COLUMNS}
    batch /= 1.5

    rebased = rebase_on_kept_axis(batch, RECORD_COLUMNS, scales)
    identity = torch.eye(4).expand(len(batch), 4, 4)

    assert torch.allclose(
        _apply(rebased, identity, RECORD_COLUMNS, scales)[0], rebased, atol=1e-5
    )
    assert torch.equal(rebase_on_kept_axis(rebased, RECORD_COLUMNS, scales), rebased)


def test_a_translation_wraps_and_a_whole_side_is_the_identity() -> None:
    images = torch.rand(8, 1, 28, 28, generator=torch.Generator().manual_seed(0))

    torch.manual_seed(0)
    moved, shifts = apply_translation(images)

    assert torch.equal(moved, translate(images, shifts))
    assert torch.equal(
        moved[0], torch.roll(images[0], tuple(shifts[0].tolist()), dims=(-2, -1))
    )
    assert torch.equal(translate(images, torch.full((8, 2), 28)), images)
    assert shifts.min() >= 0 and shifts.max() < 28
    assert apply_translation(images, max_shift=3)[1].abs().max() <= 3


def test_a_permutation_reorders_the_slots_padding_included() -> None:
    torch.manual_seed(0)
    batch = jets(n=64)

    permuted, perms = apply_permutation(batch)

    assert torch.equal(perms.sort(dim=1).values, torch.arange(8).expand(64, 8))
    assert torch.equal(permuted, torch.stack([batch[j][perms[j]] for j in range(64)]))
    assert not torch.equal(perms, perms[0].expand_as(perms))
    assert bool(
        (permuted[:, 0].abs().sum(-1) == 0).any()
    )  # a padded row can come first


@pytest.mark.parametrize("n_swaps", [0, 1, 3])
def test_n_swaps_move_at_most_two_slots_each(n_swaps: int) -> None:
    torch.manual_seed(0)
    batch = jets(n=256)

    permuted, perms = apply_permutation(batch, n_swaps=n_swaps)
    moved = (perms != torch.arange(8)).sum(dim=1)

    assert torch.equal(permuted, torch.stack([batch[j][perms[j]] for j in range(256)]))
    assert int(moved.max()) <= 2 * n_swaps
    assert n_swaps != 1 or bool(
        (moved == 2).all()
    )  # the two members of a pair are distinct


def test_the_factory_serves_every_group() -> None:
    batch, images = jets(), torch.rand(4, 1, 28, 28)
    for scheme in GROUPS:
        sample = images if scheme == "translation" else batch
        assert make_transform(scheme, FOUR_MOMENTUM)(sample).shape == sample.shape
    with pytest.raises(ValueError):
        make_transform("none", FOUR_MOMENTUM)
