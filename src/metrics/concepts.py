"""Probe targets: the frame of a draw (where on its orbit a sample sits) and concepts
constant on the orbit, per group and dataset (archive/research_2026-11-08, "The top
three", item 3).

Frames are low-rank encodings, so that LEACE removes few dimensions of a narrow layer: a
torus shift as (cos, sin) of each angle (rank 4); a permutation of the slots by the
Spearman correlation between slot and pT over the real constituents and the slot of the
leading constituent (rank 2); a beam rotation and boost as (cos, sin) of the angle and
the rapidity (rank 3).
"""

import math

import torch

from src.data.components.augmentation import feature_index, get_norm_scale

# The DFT frequencies (row, column) of an image whose magnitudes are probed: invariant
# under cyclic shifts, low enough to be smooth.
IMAGE_FREQUENCIES = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0))
# The datasets of constituents carrying pt, etarel, phirel and the four-momenta.
JET_DATASETS = ("jets", "top_tagging")


def frame_labels(
    group: str, params: torch.Tensor, x_t: torch.Tensor, layout: tuple
) -> torch.Tensor:
    """``(samples, rank)`` float64 encoding of each sample's draw.

    :param x_t: the transformed batch; the permutation's frame is read from its pT.
    :raises ValueError: for a group with no frame encoding here.
    """
    if group == "translation":
        sides = torch.tensor(x_t.shape[-2:], device=params.device)
        angles = 2 * math.pi * params.double() / sides
        return torch.cat([angles.cos(), angles.sin()], dim=1)
    if group == "permutation":
        return _slot_order(_column(x_t, layout, "pt"))
    if group == "z":
        angle, rapidity = params[:, 0].double(), params[:, 1].double()
        return torch.stack([angle.cos(), angle.sin(), rapidity], dim=1)
    raise ValueError(f"No frame encoding is defined for the group {group!r}.")


def concept_labels(
    dataset: str, x: torch.Tensor, layout: tuple
) -> dict[str, torch.Tensor]:
    """Per concept, ``(samples,)`` float64 values constant on the cell's orbits."""
    if dataset == "mnist":
        return _image_concepts(x.double())
    if dataset in JET_DATASETS:
        return _jet_concepts(x, layout)
    raise ValueError(f"No concepts are defined for the dataset {dataset!r}.")


def _column(x: torch.Tensor, layout: tuple, name: str) -> torch.Tensor:
    """One stored feature in physical units (GeV, radians), ``(jets, slots)``."""
    names, scales = layout
    return x[..., feature_index(names, name)].double() * get_norm_scale(scales, name)


def _slot_order(pt: torch.Tensor) -> torch.Tensor:
    """Spearman correlation between slot and pT over the real constituents (0 below
    two), and the leading constituent's slot over the last slot index.

    Over the real constituents alone its mean under a uniform permutation is 0 at every
    multiplicity, so an invariant layer cannot predict it from the multiplicity.
    """
    real = pt > 0
    count = real.sum(1).double()
    # Padded slots rank below every real one, so a real constituent's pT rank among the
    # real ones is its rank overall less the number of padded slots.
    masked = torch.where(real, pt.double(), -torch.inf)
    ranks = masked.argsort(dim=1, stable=True).argsort(dim=1).double()
    pt_rank = ranks - (pt.shape[1] - count).unsqueeze(1)
    slot_rank = real.double().cumsum(1) - 1.0
    squared = ((pt_rank - slot_rank) ** 2 * real).sum(1)
    spearman = 1.0 - 6.0 * squared / (count * (count**2 - 1.0)).clamp_min(1.0)
    leading = pt.argmax(1).double() / (pt.shape[1] - 1)
    return torch.stack([torch.where(count > 1, spearman, 0.0), leading], dim=1)


def _image_concepts(images: torch.Tensor) -> dict[str, torch.Tensor]:
    """Total ink and low-frequency DFT magnitudes of ``(images, 1, rows, columns)``."""
    spectrum = torch.fft.fft2(images[:, 0]).abs()
    concepts = {"ink": images.sum(dim=(1, 2, 3))}
    return concepts | {f"dft_{r}{c}": spectrum[:, r, c] for r, c in IMAGE_FREQUENCIES}


def _jet_concepts(x: torch.Tensor, layout: tuple) -> dict[str, torch.Tensor]:
    """Mass, multiplicity, girth, pT dispersion, leading pT fraction and the energy
    correlation ratios C2 and D2 at beta = 1 (Larkoski et al., arXiv:1409.6298)."""
    pt = _column(x, layout, "pt")
    share = pt / pt.sum(1, keepdim=True).clamp_min(1e-12)
    distance = _pair_distances(x, layout)
    e2 = 0.5 * torch.einsum("bi,bij,bj->b", share, distance, share)
    e3 = _triple_correlation(share, distance)
    concepts = {"mass": _mass(x, layout), "multiplicity": (pt > 0).double().sum(1)}
    concepts |= {"girth": (share * _axis_distance(x, layout)).sum(1)}
    concepts |= {"pt_dispersion": share.pow(2).sum(1).sqrt()}
    concepts |= {"leading": share.max(1).values}
    ratios = {"c2": e3 / e2.pow(2), "d2": e3 / e2.pow(3)}
    return concepts | {name: value.nan_to_num() for name, value in ratios.items()}


def _mass(x: torch.Tensor, layout: tuple) -> torch.Tensor:
    jet = [_column(x, layout, name).sum(1) for name in ("e", "px", "py", "pz")]
    return (jet[0] ** 2 - jet[1] ** 2 - jet[2] ** 2 - jet[3] ** 2).clamp_min(0.0).sqrt()


def _axis_distance(x: torch.Tensor, layout: tuple) -> torch.Tensor:
    return torch.hypot(_column(x, layout, "etarel"), _column(x, layout, "phirel"))


def _pair_distances(x: torch.Tensor, layout: tuple) -> torch.Tensor:
    """Pairwise Delta R ``(jets, slots, slots)``, phi differences wrapped to [-pi,
    pi)."""
    eta, phi = _column(x, layout, "etarel"), _column(x, layout, "phirel")
    deta = eta.unsqueeze(2) - eta.unsqueeze(1)
    dphi = (phi.unsqueeze(2) - phi.unsqueeze(1) + math.pi) % (2 * math.pi) - math.pi
    return torch.hypot(deta, dphi)


def _triple_correlation(share: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """e3 = sum over i < j < k of z_i z_j z_k R_ij R_jk R_ki, as trace(M^3) / 6 with
    M = diag(z)^(1/2) R diag(z)^(1/2); the zero diagonal of R drops repeated indices."""
    root = share.sqrt()
    weighted = root.unsqueeze(2) * distance * root.unsqueeze(1)
    cube = weighted @ weighted @ weighted
    return cube.diagonal(dim1=1, dim2=2).sum(1) / 6.0
