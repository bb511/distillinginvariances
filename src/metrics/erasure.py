"""Linear probes, LEACE erasure and its equal-rank random control.

The probe is closed-form ridge regression, scored by held-out R^2. LEACE (Belrose et al.,
arXiv:2306.03819, Theorem 3.1) removes from x the least it must so that no linear probe
can predict z: r(x) = x - S^(1/2) P S^(-1/2) (x - mu), with S the covariance of x and P
the orthogonal projector onto the column space of S^(-1/2) Cov(x, z). The control removes
a random subspace of the same rank (Elazar et al., arXiv:2006.00995).
"""

import torch

# Eigenvalues of a covariance below this share of its largest are taken as zero.
RANK_TOLERANCE = 1e-8
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def ridge_fit(x: torch.Tensor, z: torch.Tensor, ridge: float) -> torch.Tensor:
    """Weights ``(features + 1, targets)`` of ridge regression with an unpenalised
    intercept; *ridge* is in units of the mean feature variance."""
    centred = x - x.mean(0)
    scale = ridge * centred.pow(2).mean() * len(x)
    gram = centred.T @ centred + scale * torch.eye(x.shape[1], dtype=x.dtype)
    weights = torch.linalg.solve(gram, centred.T @ (z - z.mean(0)))
    intercept = z.mean(0) - x.mean(0) @ weights
    return torch.cat([weights, intercept.unsqueeze(0)])


def ridge_predict(weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x @ weights[:-1] + weights[-1]


def r_squared(z: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """R^2 per target, pooled over targets when z has several columns: one minus the
    residual over the total sum of squares."""
    residual = (z - predicted).pow(2).sum(0)
    total = (z - z.mean(0)).pow(2).sum(0)
    return 1.0 - residual.sum() / total.sum()


def fit_probe(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Ridge weights at the strength of `RIDGE_GRID` that scores best on the last fifth
    of the rows, refitted on all of them."""
    cut = int(0.8 * len(x))
    scores = [
        float(
            r_squared(z[cut:], ridge_predict(ridge_fit(x[:cut], z[:cut], r), x[cut:]))
        )
        for r in RIDGE_GRID
    ]
    return ridge_fit(x, z, RIDGE_GRID[max(range(len(scores)), key=scores.__getitem__)])


def leace(x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The LEACE eraser of *z* from *x*, as (mean, matrix): erase with `erase`."""
    mean = x.mean(0)
    centred = x - mean
    covariance = centred.T @ centred / len(x)
    cross = centred.T @ (z - z.mean(0)) / len(x)
    root, inverse_root = _roots(covariance)
    basis = torch.linalg.qr(inverse_root @ cross).Q
    return mean, root @ basis @ basis.T @ inverse_root


def random_removal(
    x: torch.Tensor, rank: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """An orthogonal projection removing a uniformly random subspace of *rank*, through
    the mean, as (mean, matrix) for `erase`."""
    gaussian = torch.randn(x.shape[1], rank, generator=generator, dtype=x.dtype)
    basis = torch.linalg.qr(gaussian).Q
    return x.mean(0), basis @ basis.T


def erase(x: torch.Tensor, eraser: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mean, matrix = eraser
    return x - (x - mean) @ matrix.T


def _roots(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The square root of a covariance and the pseudo-inverse of that root."""
    values, vectors = torch.linalg.eigh(covariance)
    kept = values > RANK_TOLERANCE * values.max()
    root = torch.where(kept, values.clamp_min(0.0).sqrt(), 0.0)
    inverse = torch.where(kept, 1.0 / root.clamp_min(1e-300), 0.0)
    return (vectors * root) @ vectors.T, (vectors * inverse) @ vectors.T


def forward_from(
    net: torch.nn.Module, index: int, hidden: torch.Tensor
) -> torch.Tensor:
    """An `MLPStudent`'s logits continued from its post-ReLU hidden layer *index*
    (0-based), on the network's device and in its dtype."""
    value = hidden.to(net.output.weight)
    for layer in net.hidden[index + 1 :]:
        value = torch.relu(layer(value))
    return net.output(value)
