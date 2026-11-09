"""The layer-wise orbit-variance profile: how much of each layer's variation over the data
comes from moving along the group orbit rather than from changing the sample.

For unit k of a layer, with h = h_k(g.x) over samples x and draws g, the law of total
variance splits Var_{x,g} h into Var_x E_g h (between orbits) and E_x Var_g h (within).
The invariance fraction I = sum_k Var_x E_g h / sum_k Var_{x,g} h over live units is 1 for
a layer constant on every orbit (archive/research_2026-11-08, "The top three", item 2).
With n Monte Carlo draws per sample the within-orbit variance is the unbiased one, so I
carries no 1/n bias; an exhaustive orbit (all 784 torus shifts) uses the exact one, and
then also gives Quiroga et al.'s normalised variance NV = sum_k E_x Var_g h / sum_k E_g
Var_x h (arXiv:2310.17404) and the orbit's Fourier spectrum. A unit whose total variance
is below `DEAD` of the layer's largest is dead and left out of both sums.
"""

import numpy as np
import torch
from lightning import LightningModule

from src.data.components.augmentation import translate
from src.utils.bootstrap import percentile_interval, resampled_column_means

DEAD = 1e-10


def layer_outputs(module: LightningModule, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """``(samples, units)`` per layer: a student's hidden layers, post-ReLU, then its
    logits; a teacher's hint (its pooled, post-aggregation layer), then its logits.

    Networks that read the jets' Lorentz matrix get none, so they are not profiled.
    """
    if hasattr(module.net, "forward_with_hidden"):
        # The column selection and preparation of the module's own forward.
        prepared = module.net.prepare(module._select(x), module.norm_scales)
        logits, hidden = module.net.forward_with_hidden(prepared)
        layers = {f"hidden_{index + 1}": h for index, h in enumerate(hidden)}
        return layers | {"logits": logits}
    if module.net.hint_dim:
        logits, hint = module.forward_with_hint(x)
        return {"hint": hint.flatten(1), "logits": logits}
    return {"logits": module(x)}


@torch.no_grad()
def orbit_outputs(
    module: LightningModule, orbit: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Every layer over ``(samples, draws, ...)`` inputs, ``(samples, draws, units)``,
    in one forward pass."""
    outputs = layer_outputs(module, orbit.flatten(0, 1))
    return {name: v.reshape(*orbit.shape[:2], -1) for name, v in outputs.items()}


def torus_orbit(x: torch.Tensor) -> torch.Tensor:
    """Every cyclic shift of every image, ``(images, rows * columns, channels, rows,
    columns)``, the shifts (u, v) in row-major order."""
    n, _, rows, columns = x.shape
    grid = torch.cartesian_prod(torch.arange(rows), torch.arange(columns))
    images = x.repeat_interleave(rows * columns, 0)
    shifted = translate(images, grid.repeat(n, 1).to(x.device))
    return shifted.reshape(n, rows * columns, *x.shape[1:])


def sample_moments(acts: torch.Tensor) -> np.ndarray:
    """``(samples, 3 units)``: per unit, the mean over the draws, its square and the
    mean square, in float64; the column means of these give every variance of I."""
    acts = acts.double()
    mean = acts.mean(1)
    return torch.cat([mean, mean**2, acts.pow(2).mean(1)], dim=1).cpu().numpy()


def invariance_fraction(
    moments: np.ndarray, n_draws: int, exhaustive: bool, n_boot: int, seed: int
) -> dict[str, float]:
    """I from `sample_moments` rows, with its percentile 95 % interval from resampling
    the samples, and the counts of live and dead units."""
    correction = 1.0 if exhaustive else n_draws / (n_draws - 1)
    point = moments.mean(0)
    live = _live(point)
    replicates = resampled_column_means(moments, n_boot, seed)
    low, high = percentile_interval(_fraction(replicates, correction, live))
    value = float(_fraction(point, correction, live))
    counts = {"n_live": int(live.sum()), "width": int(live.size)}
    return {"invariance": value, "ci_low": float(low), "ci_high": float(high)} | counts


def shared_draw_sums(acts: torch.Tensor) -> np.ndarray:
    """Per draw and unit, the sums over the samples of h and of h^2, ``(2, draws,
    units)``, for draws that every sample shares (the exhaustive torus); add them over
    batches."""
    acts = acts.double()
    return torch.stack([acts.sum(0), acts.pow(2).sum(0)]).cpu().numpy()


def normalised_variance(moments: np.ndarray, sums: np.ndarray) -> float:
    """NV over live units from every sample's `sample_moments` of an exhaustive orbit
    and the summed `shared_draw_sums`."""
    point, n = moments.mean(0), len(moments)
    mean_squared, square = np.split(point, 3)[1:]
    within = square - mean_squared
    across = (sums[1] / n - (sums[0] / n) ** 2).mean(0)
    live = _live(point)
    return float(within[live].sum() / across[live].sum())


def spectrum_shares(energy: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """The summed `torus_spectrum` as shares of the layer's total variance: frequency
    (0, 0) holds I and every other the within-orbit variance at it, so they sum to 1."""
    point = moments.mean(0)
    live = _live(point)
    mean, _, square = np.split(point, 3)
    shares = energy / len(moments) / (square - mean**2)[live].sum()
    shares[0, 0] = _fraction(point, 1.0, live)
    return shares


def torus_spectrum(acts: torch.Tensor, side: int) -> np.ndarray:
    """Energy per frequency of the orbit over all ``side**2`` shifts, summed over the
    samples and units of ``(samples, side**2, units)`` activations, shape (side, side).

    With f the unit over the shifts, |fft2(f)|^2 / side^4 sums to the mean square over
    the orbit; frequency (0, 0) holds its squared mean and the rest its variance
    (Parseval).
    """
    grid = acts.double().reshape(acts.shape[0], side, side, -1)
    energy = torch.fft.fft2(grid, dim=(1, 2)).abs().pow(2) / side**4
    return energy.sum(dim=(0, 3)).cpu().numpy()


def transposition_orbits(x: torch.Tensor) -> torch.Tensor:
    """Every transposition of two slots of every jet, ``(jets, pairs, slots, features)``
    with the pairs i < j in row-major order."""
    slots = x.shape[1]
    first, second = torch.triu_indices(slots, slots, offset=1, device=x.device)
    perms = torch.arange(slots, device=x.device).repeat(len(first), 1)
    pairs = torch.arange(len(first), device=x.device)
    perms[pairs, first], perms[pairs, second] = second, first
    return x[:, perms]


def transposition_distances(
    orbit: torch.Tensor, clean: torch.Tensor
) -> tuple[np.ndarray, float]:
    """Summed over the samples: the squared distance each transposition moves the layer,
    ``(pairs,)``, and the squared distance between a sample and its batch neighbour, the
    between-sample scale that normalises the map.

    :param orbit: the layer on `transposition_orbits`, ``(samples, pairs, units)``.
    :param clean: the layer on the untransposed samples, ``(samples, units)``.
    """
    moved = (orbit.double() - clean.double().unsqueeze(1)).pow(2).sum(-1).sum(0)
    scale = (clean.double() - clean.double().roll(1, 0)).pow(2).sum()
    return moved.cpu().numpy(), float(scale)


def pair_matrix(values: np.ndarray, slots: int) -> np.ndarray:
    """``(slots, slots)``, symmetric with a zero diagonal, from per-pair values in the
    order of `transposition_orbits`."""
    matrix = np.zeros((slots, slots))
    first, second = np.triu_indices(slots, k=1)
    matrix[first, second] = matrix[second, first] = values
    return matrix


def _live(point: np.ndarray) -> np.ndarray:
    mean, _, square = np.split(point, 3)
    total = square - mean**2
    return total > DEAD * max(float(total.max()), np.finfo(float).tiny)


def _fraction(means: np.ndarray, correction: float, live: np.ndarray) -> np.ndarray:
    """I from column means of `sample_moments`; works on ``(..., 3 units)``."""
    mean, mean_squared, square = np.split(means, 3, axis=-1)
    total = square - mean**2
    within = (square - mean_squared) * correction
    return (total - within)[..., live].sum(-1) / total[..., live].sum(-1)
