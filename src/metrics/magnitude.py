"""How far each probe draw moves its sample, the axis of the magnitude bins."""

import math
from collections.abc import Sequence

import numpy as np
import torch


def magnitudes(
    group: str, params: torch.Tensor, x: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-sample sizes of one draw of *group*, each ``(samples,)`` in float64.

    :param params: what `~src.data.components.augmentation.apply_group` returned.
    :param x: the batch before the draw; the permutation reads its padding from it.
    :raises ValueError: for a group with no magnitude defined here.
    """
    if group == "translation":
        return _translation(params, x.shape[-2:])
    if group == "permutation":
        return _permutation(params, x)
    if group == "so2":
        return {"angle": _wrapped(params.double())}
    if group == "so3":
        return _rotation(params.double())
    if group in ("lorentz", "lorentz_iso"):
        return _boost(params.double())
    if group == "z":
        # The draws are (angle, rapidity) per jet, the rapidity signed along the beam.
        return {
            "angle": _wrapped(params[:, 0].double()),
            "rapidity": params[:, 1].double(),
        }
    raise ValueError(f"No magnitude is defined for the group {group!r}.")


def _wrapped(angle: torch.Tensor) -> torch.Tensor:
    """An angle in radians folded onto [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def with_derived(sizes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """*sizes* with ``abs_<name>`` for every magnitude and, for a torus shift,
    ``shift_dist``, its Euclidean length in pixels."""
    derived = sizes | {f"abs_{name}": np.abs(value) for name, value in sizes.items()}
    if "shift_row" in sizes and "shift_col" in sizes:
        derived["shift_dist"] = np.hypot(sizes["shift_row"], sizes["shift_col"])
    return derived


def bin_masks(
    values: np.ndarray, edges: Sequence[float]
) -> list[tuple[float, float, np.ndarray]]:
    """``(low, high, mask)`` per bin of *edges*: [low, high), the last bin closed."""
    edges = [float(edge) for edge in edges]
    upper = [values < high for high in edges[1:-1]] + [values <= edges[-1]]
    return [
        (low, high, (values >= low) & below)
        for low, high, below in zip(edges[:-1], edges[1:], upper)
    ]


def _translation(shifts: torch.Tensor, sides: torch.Size) -> dict[str, torch.Tensor]:
    """Row and column shifts in pixels, folded onto [-side/2, side/2) of the torus."""
    signed = [
        (shifts[:, axis] + side // 2) % side - side // 2
        for axis, side in enumerate(sides)
    ]
    return {"shift_row": signed[0].double(), "shift_col": signed[1].double()}


def _permutation(perms: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """The real constituents that change slot, and how many real ones the jet has.

    A padded slot is an all-zero row. Output slot i holds input row ``perms[i]``, so a
    real constituent has moved where ``perms[i] != i`` and row ``perms[i]`` is real.
    """
    real = x.abs().sum(-1) > 0
    slots = torch.arange(perms.shape[1], device=perms.device)
    moved = perms.ne(slots) & real.gather(1, perms)
    return {"moved": moved.sum(1).double(), "multiplicity": real.sum(1).double()}


def _rotation(mats: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rotation angle and the tilt of the beam axis, both in radians on [0, pi].

    The tilt, the angle between R z and z, is the part of the rotation that moves the
    beam; a rotation about the beam alone has tilt 0.
    """
    trace = mats.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    return {"angle": angle, "tilt": torch.arccos(mats[:, 2, 2].clamp(-1.0, 1.0))}


def _boost(mats: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rapidity, >= 0, and the cosine between the boost direction and the beam.

    The boost row ``mats[:, 0, 1:]`` is ``-sinh(xi) n``, so its norm gives the rapidity
    through asinh, which keeps its precision at small xi where acosh of the energy entry
    does not. A beam boost has ``axis_cos = sign(xi)``; a draw of exactly zero gets 0.
    """
    row = mats[:, 0, 1:]
    sinh = row.norm(dim=1)
    axis_cos = torch.where(sinh > 0, -row[:, 2] / sinh.clamp_min(1e-300), 0.0)
    return {"rapidity": torch.asinh(sinh), "axis_cos": axis_cos}
