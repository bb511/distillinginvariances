import math

import pytest
import torch

from src.metrics.magnitude import magnitudes


def _beam_rotation(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _boost(xi: float, axis: tuple[float, float, float]) -> torch.Tensor:
    n = torch.tensor(axis, dtype=torch.float64)
    boost = torch.eye(4, dtype=torch.float64)
    boost[0, 0] = math.cosh(xi)
    boost[0, 1:] = boost[1:, 0] = -math.sinh(xi) * n
    boost[1:, 1:] += (math.cosh(xi) - 1.0) * torch.outer(n, n)
    return boost


def test_translations_fold_onto_the_signed_torus() -> None:
    shifts = torch.tensor([[0, 27], [14, 3], [-2, 13]])

    sizes = magnitudes("translation", shifts, torch.zeros(3, 1, 28, 28))

    assert sizes["shift_row"].tolist() == [0.0, -14.0, -2.0]
    assert sizes["shift_col"].tolist() == [-1.0, 3.0, 13.0]


def test_a_permutation_counts_only_the_real_constituents_it_moves() -> None:
    x = torch.zeros(3, 4, 2)
    x[:, :2] = 1.0
    perms = torch.tensor([[1, 0, 3, 2], [0, 1, 3, 2], [2, 1, 0, 3]])

    sizes = magnitudes("permutation", perms, x)

    assert sizes["moved"].tolist() == [2.0, 0.0, 1.0]
    assert sizes["multiplicity"].tolist() == [2.0, 2.0, 2.0]


def test_rotations_separate_the_angle_from_the_tilt_of_the_beam() -> None:
    tilted = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    sizes = magnitudes("so3", torch.stack([_beam_rotation(2.0), tilted]), None)
    angles = magnitudes("so2", torch.tensor([0.5, 6.0]), None)["angle"]

    assert torch.allclose(sizes["angle"], torch.tensor([2.0, math.pi / 2]).double())
    assert torch.allclose(sizes["tilt"], torch.tensor([0.0, math.pi / 2]).double())
    assert torch.allclose(angles, torch.tensor([0.5, 6.0 - 2 * math.pi]).double())


def test_boosts_give_their_rapidity_and_direction() -> None:
    mats = torch.stack(
        [
            _boost(0.7, (0.0, 0.0, 1.0)),
            _boost(-0.7, (0.0, 0.0, 1.0)),
            _boost(1e-4, (1.0, 0.0, 0.0)),
            torch.eye(4, dtype=torch.float64),
        ]
    )

    sizes = magnitudes("lorentz_iso", mats, None)

    expected = torch.tensor([0.7, 0.7, 1e-4, 0.0], dtype=torch.float64)
    assert torch.allclose(sizes["rapidity"], expected, rtol=1e-9, atol=0.0)
    assert sizes["axis_cos"].tolist() == [1.0, -1.0, 0.0, 0.0]


def test_the_beam_group_gives_its_angle_and_signed_rapidity() -> None:
    draws = torch.tensor([[0.5, -0.8], [6.0, 0.3]])

    sizes = magnitudes("z", draws, None)

    expected = torch.tensor([0.5, 6.0 - 2 * math.pi], dtype=torch.float64)
    assert torch.allclose(sizes["angle"], expected)
    assert torch.allclose(sizes["rapidity"], draws[:, 1].double())


def test_a_group_without_a_magnitude_is_refused() -> None:
    with pytest.raises(ValueError, match="No magnitude"):
        magnitudes("dilation", torch.ones(2), None)
