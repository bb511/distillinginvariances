"""Transforms of a batch under the group actions of the study.

`apply_group` is the one entry point, over the groups of `src.vocabulary.GROUPS`: the
circular translation of an image, the permutation of a jet's constituent slots, and the
rotations and boosts of the constituent four-vectors. Every one draws per sample from
the ambient torch RNG and returns the transformed batch with the parameters it drew, so
a caller can reproduce or invert the draw.

A four-vector transform takes a ``(jets, constituents, features)`` tensor and returns
one of the same shape and normalisation: the columns are taken back to GeV by
``norm_scales``, acted on by one matrix per jet, and written back. Four-vectors are
ordered [E, px, py, pz] and the beam is the z axis.

A batch must contain px, py, pz and e, bare or with the record's ``j1_`` prefix.
Whichever of pt, eta, phi, etarel, phirel, deltaR and ptrel it has are rewritten, and
erel, etarot, phirot, costheta and costhetarel are left as found. Therefore, a feature
list holding those would have stale values. The squared mass `MASS_COLUMN` is left as
found too, which is exact, since every four-vector transform preserves it; a permutation
moves it with its constituent's row. Padded constituents are all-zero rows: the
transform does not apply to these.

Of the rewritten columns, etarel, phirel, deltaR and ptrel are measured from the jet
axis. This is computed based on the constituent values present in the data.
"""

import math
from collections.abc import Callable, Sequence

import torch

from src.vocabulary import GROUPS

Transform = Callable[[torch.Tensor], torch.Tensor]

# The columns measured from the jet axis.
AXIS_COLUMNS = ("etarel", "phirel", "deltaR", "ptrel")
# The squared mass of each constituent in GeV^2, unscaled, appended by
# `append_mass_column`.
MASS_COLUMN = "m2"
# Jets per chunk of the float64 mass computation, which bounds its memory.
MASS_CHUNK = 65536


def compute_pt(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(px**2 + py**2)


def compute_phi(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    return torch.atan2(py, px)


def compute_eta(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    p = torch.sqrt(px**2 + py**2 + pz**2)
    # eps keeps a purely longitudinal constituent, and a padded all-zero row, off the log
    # singularity: both go to eta = 0 rather than to +/- inf.
    return 0.5 * torch.log((p + pz + eps) / (p - pz + eps))


def feature_index(feature_names: Sequence[str], name: str) -> int | None:
    """Return the feature of *name*, bare or ``j1_``-prefixed, or None."""
    names = list(feature_names)
    return next((names.index(c) for c in (name, f"j1_{name}") if c in names), None)


def get_norm_scale(norm_scales: dict[str, float] | None, name: str) -> float:
    """Return the factor taking the stored feature of *name* back to GeV."""
    scales = norm_scales or {}
    return float(scales.get(name, scales.get(f"j1_{name}", 1.0)))


def _matrices(rows: list[list[torch.Tensor]]) -> torch.Tensor:
    """Stack nine per-sample entries, row by row, into matrices."""
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def sample_so2_rot_matrices(
    nbr_rot: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw *nbr_rot* rotations about the beam, shape (nbr_rot, 3, 3), and their angles.

    The angle is uniform on the full turn, [0, 2 pi) since the azimuth of a pp collision
    does not contain physics information.
    """
    thetas = torch.rand(max(nbr_rot, 0), device=device) * (2.0 * math.pi)
    cos, sin = torch.cos(thetas), torch.sin(thetas)
    zero, one = torch.zeros_like(thetas), torch.ones_like(thetas)
    return _matrices([[cos, -sin, zero], [sin, cos, zero], [zero, zero, one]]), thetas


def sample_haar_so3_rot_matrices(
    num_rotations: int, device: torch.device | None = None
) -> torch.Tensor:
    """Draw *num_rotations* Haar-random rotations, shape (num_rotations, 3, 3).

    A Gaussian in R^4 is isotropic: the normalised draws are uniform on the unit
    quaternions. This is just the Haar measure on SO(3). The clamp forbids the all-zero
    draw.
    """
    quat = torch.randn(max(num_rotations, 0), 4, device=device)
    x, y, z, w = (quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-8)).unbind(dim=1)
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return _matrices(rows)


def _sample_boosts(
    nbr_rot: int, eta_max: float, isotropic: bool, device: torch.device | None
) -> torch.Tensor:
    """Draw one boost per jet, shape (nbr_rot, 4, 4), acting on [E, px, py, pz].

    The rapidity is drawn uniform on [-eta_max, eta_max]. Along the beam this is the
    symmetry of a pp collision. We also draw isotropic Lorentz rotations, which are not
    a symmetry of the data but are used later to test models that claim to be fully
    Lorentz invariant or equivariant.
    """
    beam = torch.eye(3, device=device)[2].expand(nbr_rot, 3)
    axis = torch.randn(nbr_rot, 3, device=device) if isotropic else beam

    # The clamp guards an all-zero draw.
    axis = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-8)
    xi = (torch.rand(nbr_rot, device=device) * 2.0 - 1.0) * eta_max
    boosts = torch.eye(4, device=device).repeat(nbr_rot, 1, 1)
    boosts[:, 0, 0] = torch.cosh(xi)
    boosts[:, 0, 1:] = -torch.sinh(xi).unsqueeze(1) * axis
    boosts[:, 1:, 0] = -torch.sinh(xi).unsqueeze(1) * axis
    boosts[:, 1:, 1:] += (
        (torch.cosh(xi) - 1)[:, None, None] * axis[:, :, None] * axis[:, None, :]
    )

    return boosts


def sample_z_matrices(
    nbr_rot: int, eta_max: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw *nbr_rot* rotations about the beam composed with boosts along it, shape
    (nbr_rot, 4, 4) on [E, px, py, pz], and their parameters.

    The angle is uniform on [0, 2 pi), as in `sample_so2_rot_matrices`, and the rapidity
    uniform on [-eta_max, eta_max], with the sign convention of `_sample_boosts`. The
    two factors commute, so each matrix is either product.

    :return: the matrices and the draws, shape (nbr_rot, 2): angle, then rapidity.
    """
    thetas = torch.rand(max(nbr_rot, 0), device=device) * (2.0 * math.pi)
    xi = (torch.rand(max(nbr_rot, 0), device=device) * 2.0 - 1.0) * eta_max
    cos, sin = torch.cos(thetas), torch.sin(thetas)
    ch, sh = torch.cosh(xi), torch.sinh(xi)
    zero = torch.zeros_like(thetas)
    rows = [
        [ch, zero, zero, -sh],
        [zero, cos, -sin, zero],
        [zero, sin, cos, zero],
        [-sh, zero, zero, ch],
    ]
    return _matrices(rows), torch.stack([thetas, xi], dim=-1)


def _read_four_momenta(
    x: torch.Tensor, names: Sequence[str], norm_scales: dict[str, float] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [E, px, py, pz] in GeV and the real-constituent mask.

    :raises ValueError: if the batch lacks one of the four required features.
    """
    indices = {key: feature_index(names, key) for key in ("e", "px", "py", "pz")}
    if None in indices.values():
        raise ValueError(
            f"A transform needs the columns {list(indices)}, got {list(names)}."
        )
    p4 = [
        x[..., index] * get_norm_scale(norm_scales, key)
        for key, index in indices.items()
    ]
    return torch.stack(p4, dim=-1), compute_pt(p4[1], p4[2]) > 0


def _derived_columns(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, active: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The features that follow from the momenta, in GeV and radians.

    The jet axis is the sum over the real constituents; phirel is wrapped to [-pi, pi)
    and ptrel is the constituent's share of the jet pt.
    """
    pt, eta, phi = compute_pt(px, py), compute_eta(px, py, pz), compute_phi(px, py)
    jet_px, jet_py, jet_pz = (
        (p * active).sum(dim=1, keepdim=True) for p in (px, py, pz)
    )
    etarel = eta - compute_eta(jet_px, jet_py, jet_pz)
    phirel = (phi - compute_phi(jet_px, jet_py) + math.pi) % (2 * math.pi) - math.pi
    # A jet of padded slots has no pt to share out; the clamp is in GeV.
    ptrel = pt / compute_pt(jet_px, jet_py).clamp(min=1e-8)
    columns = {"pt": pt, "eta": eta, "phi": phi, "etarel": etarel, "phirel": phirel}
    return columns | {"deltaR": torch.sqrt(etarel**2 + phirel**2), "ptrel": ptrel}


def _write(
    x: torch.Tensor,
    names: Sequence[str],
    norm_scales: dict[str, float] | None,
    columns: dict[str, torch.Tensor],
    where: torch.Tensor,
) -> torch.Tensor:
    """Return *x* with each feature in *columns* replaced where *where* is True."""
    x_new = x.clone()
    for name, values in columns.items():
        index = feature_index(names, name)
        if index is not None:
            scaled = values / get_norm_scale(norm_scales, name)
            x_new[..., index] = torch.where(where, scaled, x[..., index])
    return x_new


def _apply(
    x: torch.Tensor,
    mats: torch.Tensor,
    names: Sequence[str],
    norm_scales: dict[str, float] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Act on every jet with its own matrix and rewrite the columns that follow from it.

    :param mats: shape (jets, 4, 4) on [E, px, py, pz], or (jets, 3, 3) on the momenta
        alone.
    :return: the new batch, the physical columns it was written from and the real-
        constituent mask. A padded slot keeps its zeros, and so does a constituent left
        with no positive energy.
    """
    if mats.shape[-1] == 3:
        # A rotation leaves the energy alone: pad it to the block-diagonal 4 x 4 form.
        mats = torch.nn.functional.pad(mats, (1, 0, 1, 0))
        mats[:, 0, 0] = 1.0
    p4, active = _read_four_momenta(x, names, norm_scales)
    e, px, py, pz = torch.einsum("bij,bcj->bci", mats.to(p4.dtype), p4).unbind(dim=-1)
    columns = {"e": e, "px": px, "py": py, "pz": pz} | _derived_columns(
        px, py, pz, active
    )
    return _write(x, names, norm_scales, columns, active & (e > 0)), columns, active


def rebase_on_kept_axis(
    x: torch.Tensor,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return *x* with `AXIS_COLUMNS` measured from the present constituents.

    The record stores those four against the untruncated jet's axis. A transform
    recomputes them from the present constituents. Everything else, the momenta and the
    absolute pt, eta and phi among them, is returned untouched.
    """
    p4, active = _read_four_momenta(x, feature_names, norm_scales)
    _, px, py, pz = p4.unbind(dim=-1)
    columns = _derived_columns(px, py, pz, active)
    kept = {name: columns[name] for name in AXIS_COLUMNS}
    return _write(x, feature_names, norm_scales, kept, active)


def append_mass_column(
    x: torch.Tensor,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return *x* with `MASS_COLUMN` appended: E^2 - |p|^2 of each constituent, GeV^2.

    It is computed in float64 from the de-normalised stored components and appended
    once, as a split is loaded, before any transform. About half the constituents are
    massless, and their E^2 - |p|^2 recomputed in float32 from a transformed row is
    rounding, O(1e-7) E^2, that changes with the transform; the stored value travels
    with its constituent unchanged. Padded rows get zero.

    :param x: (jets, constituents, features), holding px, py, pz and e.
    :return: (jets, constituents, features + 1), in the dtype of *x*.
    """
    masses = []
    for part in x.split(MASS_CHUNK):
        p4, _ = _read_four_momenta(part.double(), feature_names, norm_scales)
        e, p = p4[..., 0], p4[..., 1:]
        masses.append((e**2 - (p**2).sum(dim=-1)).to(x.dtype))
    return torch.cat([x, torch.cat(masses).unsqueeze(-1)], dim=-1)


def apply_random_so2(
    x: torch.Tensor,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate every jet about the beam by its own angle; return the batch and the
    angles."""
    mats, thetas = sample_so2_rot_matrices(x.shape[0], x.device)
    return _apply(x, mats, feature_names, norm_scales)[0], thetas


def apply_random_so3(
    x: torch.Tensor,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate every jet by its own Haar-random matrix.

    The draw is not restricted to the detector: about 3.5 % of the rotated jets have a
    constituent beyond ``|eta| = 3``, accepted because SO(3) is the control for a group
    that is not a symmetry of the data.

    :return: the rotated batch and the matrices it was rotated by, shape (jets, 3, 3).
    """
    mats = sample_haar_so3_rot_matrices(x.shape[0], x.device)
    return _apply(x, mats, feature_names, norm_scales)[0], mats


def apply_random_lorentz(
    x: torch.Tensor,
    feature_names: Sequence[str],
    eta_max: float = 1.0,
    norm_scales: dict[str, float] | None = None,
    isotropic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boost every jet by its own rapidity, uniform on [-eta_max, eta_max].

    Along the beam by default or in a random direction if *isotropic* is True. Boosting
    without caring about the detector bounds creates 0.2% jets that would not be seen by
    the detector normally.

    :return: the batch and the boosts, shape (jets, 4, 4).
    """
    boosts = _sample_boosts(x.shape[0], eta_max, isotropic, x.device)
    return _apply(x, boosts, feature_names, norm_scales)[0], boosts


def apply_random_z(
    x: torch.Tensor,
    feature_names: Sequence[str],
    eta_max: float = 1.0,
    norm_scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate every jet about the beam and boost it along the beam, by its own angle and
    rapidity; return the batch and the draws, shape (jets, 2): angle, rapidity."""
    mats, draws = sample_z_matrices(x.shape[0], eta_max, x.device)
    return _apply(x, mats, feature_names, norm_scales)[0], draws


def translate(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Roll each image by its own shift, wrapping at the edges.

    :param x: shape (images, channels, rows, columns).
    :param shifts: shape (images, 2), the row and column shift in pixels; a shift of a
        full side is the identity, and a negative one rolls the other way.
    """
    n, c, h, w = x.shape
    rows = (torch.arange(h, device=x.device).view(1, h, 1) - shifts[:, :1, None]) % h
    cols = (torch.arange(w, device=x.device).view(1, 1, w) - shifts[:, 1:, None]) % w
    flat = (rows * w + cols).unsqueeze(1).expand(n, c, h, w).reshape(n, c, h * w)
    return x.reshape(n, c, h * w).gather(2, flat).reshape(n, c, h, w)


def apply_translation(
    x: torch.Tensor, max_shift: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate every image by its own shift; return the batch and the shifts.

    The shift is uniform on the whole side when *max_shift* is None, which makes the
    group the full torus of translations, and on [-max_shift, max_shift] otherwise.

    :param x: shape (images, channels, rows, columns).
    :return: the translated batch and the shifts, shape (images, 2), in pixels.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected (images, channels, rows, columns), got {x.shape}.")
    n, _, h, w = x.shape
    whole = [(0, h), (0, w)]
    bounds = whole if max_shift is None else [(-max_shift, max_shift + 1)] * 2
    draws = [torch.randint(lo, hi, (n,), device=x.device) for lo, hi in bounds]
    shifts = torch.stack(draws, dim=-1)
    return translate(x, shifts), shifts


def random_transpositions(
    n_jets: int, n_slots: int, n_swaps: int, device: torch.device | None = None
) -> torch.Tensor:
    """Compose *n_swaps* random transpositions per jet; return shape (jets, slots).

    The pairs are drawn with replacement, so a later swap can undo an earlier one, but
    the two members of a pair are always distinct slots: the second member is offset
    from the first by 1 to slots - 1, which is uniform over the other slots.
    """
    first = torch.randint(n_slots, (n_jets, n_swaps), device=device)
    offset = torch.randint(1, n_slots, (n_jets, n_swaps), device=device)
    pairs = torch.stack([first, (first + offset) % n_slots], dim=-1)
    perms = torch.arange(n_slots, device=device).repeat(n_jets, 1)
    jets = torch.arange(n_jets, device=device)
    # (swaps, 2, jets): one step of the loop swaps one pair in every jet at once.
    for a, b in pairs.permute(1, 2, 0):
        perms[jets, a], perms[jets, b] = perms[jets, b], perms[jets, a]
    return perms


def apply_permutation(
    x: torch.Tensor, n_swaps: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute the constituent slots of every jet independently.

    The permutation is uniform over all orders when *n_swaps* is None, and otherwise the
    composition of *n_swaps* random transpositions, which moves at most ``2 * n_swaps``
    rows. Padded slots take part, since the permutation acts on the slots rather than on
    the constituents; a padded row can therefore end up before a real one.

    :param x: shape (jets, constituents, features).
    :return: the permuted batch and the permutations, shape (jets, constituents), row
        ``i`` of the output being row ``perms[jet, i]`` of the input.
    """
    n_jets, n_slots = x.shape[:2]
    if n_swaps is None:
        perms = torch.rand(n_jets, n_slots, device=x.device).argsort(dim=1)
    else:
        perms = random_transpositions(n_jets, n_slots, n_swaps, x.device)
    return x.gather(1, perms.unsqueeze(-1).expand_as(x)), perms


def apply_group(
    group: str,
    x: torch.Tensor,
    feature_names: Sequence[str] | None = None,
    norm_scales: dict[str, float] | None = None,
    lorentz_eta_max: float = 1.0,
    max_shift: int | None = None,
    n_swaps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Act on *x* with one draw from *group* per sample.

    :param group: one of `src.vocabulary.GROUPS`.
    :param feature_names: the columns of *x*, needed by the four-vector groups alone.
    :return: the transformed batch and what was drawn: the boosts (jets, 4, 4) of
        ``lorentz`` and ``lorentz_iso``, the angles (jets,) of ``so2``, the matrices
        (jets, 3, 3) of ``so3``, the angles and rapidities (jets, 2) of ``z``, the
        permutations (jets, constituents) or the shifts (images, 2).
    """
    if group == "translation":
        return apply_translation(x, max_shift)
    if group == "permutation":
        return apply_permutation(x, n_swaps)
    return _apply_momentum_group(group, x, feature_names, norm_scales, lorentz_eta_max)


def _apply_momentum_group(
    group: str,
    x: torch.Tensor,
    feature_names: Sequence[str] | None,
    norm_scales: dict[str, float] | None,
    eta_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    names = list(feature_names or ())
    if group == "so2":
        return apply_random_so2(x, names, norm_scales)
    if group == "so3":
        return apply_random_so3(x, names, norm_scales)
    if group in ("lorentz", "lorentz_iso"):
        isotropic = group == "lorentz_iso"
        return apply_random_lorentz(x, names, eta_max, norm_scales, isotropic)
    if group == "z":
        return apply_random_z(x, names, eta_max, norm_scales)
    raise ValueError(
        f"Transformation scheme must be one of {list(GROUPS)}, got {group!r}..."
    )


def make_transform(
    scheme: str,
    feature_names: Sequence[str] | None = None,
    norm_scales: dict[str, float] | None = None,
    lorentz_eta_max: float = 1.0,
    max_shift: int | None = None,
    n_swaps: int | None = None,
) -> Transform:
    """Build the transform named by *scheme*, drawing afresh on every call.

    The datamodule, the invariance probe and the online augmentation all call this, so
    the three cannot drift apart.

    :param scheme: one of `src.vocabulary.GROUPS`, the boosts being along the beam
        (``lorentz``) or in a random direction (``lorentz_iso``).
    :param lorentz_eta_max: half-width of the uniform rapidity range of the boosts.
    :param max_shift: half-width of the uniform translation, None for the whole side.
    :param n_swaps: number of random transpositions composed into each permutation, None
        for a uniform permutation.
    :return: a callable from a batch to its transformed batch of the same shape and
        normalisation, dropping what was drawn.
    """
    if scheme not in GROUPS:
        raise ValueError(
            f"Transformation scheme must be one of {list(GROUPS)}, got {scheme!r}..."
        )
    names = None if feature_names is None else list(feature_names)
    return lambda x: apply_group(
        scheme, x, names, norm_scales, lorentz_eta_max, max_shift, n_swaps
    )[0]
