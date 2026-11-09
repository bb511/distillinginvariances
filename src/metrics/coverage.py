"""The classifier two-sample test of a probe's coverage (Lopez-Paz and Oquab,
arXiv:1610.06545).

A small network learns to tell the student's inputs x from their probe draws g.x; its
held-out AUC is about 0.5 where the orbit is covered by the data and 1 where every g.x
is recognisable, such as an unsorted jet under S50.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms.components import mlp_head
from src.utils.bootstrap import percentile_interval


def fit_classifier(
    x: torch.Tensor,
    x_t: torch.Tensor,
    hidden: Sequence[int],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> nn.Module:
    """A ReLU MLP whose single output is the logit of "g.x", trained by Adam on the
    pairs ``(rows, features)`` *x* and *x_t*, on their device."""
    inputs = torch.cat([x, x_t])
    labels = torch.cat([torch.zeros(len(x)), torch.ones(len(x_t))]).to(x.device)
    net = _network(x.shape[1], hidden, seed, x.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    order = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        batches = torch.randperm(len(inputs), generator=order).split(batch_size)
        _epoch(net, optimizer, inputs, labels, batches)
    return net.eval()


@torch.no_grad()
def scores(net: nn.Module, x: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """The classifier's logits, ``(rows,)``."""
    return torch.cat([net(part).squeeze(-1) for part in x.split(chunk)])


def auc(negative: torch.Tensor, positive: torch.Tensor) -> float:
    """Mann-Whitney AUC: the chance that a positive outscores a negative, ties half."""
    ranks = _average_ranks(torch.cat([negative, positive]).double())
    n_negative, n_positive = len(negative), len(positive)
    exceed = ranks[n_negative:].sum() - n_positive * (n_positive + 1) / 2
    return float(exceed / (n_negative * n_positive))


def auc_interval(
    negative: torch.Tensor, positive: torch.Tensor, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """The AUC of paired scores (row i of each comes from the same sample) and its
    percentile 95 % interval from resampling the pairs."""
    generator = torch.Generator(device=negative.device).manual_seed(seed)
    n = len(negative)
    replicates = []
    for _ in range(n_boot):
        rows = torch.randint(n, (n,), generator=generator, device=negative.device)
        replicates.append(auc(negative[rows], positive[rows]))
    low, high = percentile_interval(np.array(replicates))
    return auc(negative, positive), float(low), float(high)


def _network(
    n_features: int, hidden: Sequence[int], seed: int, device: torch.device
) -> nn.Module:
    # Initialised under its own seed, so the ambient RNG and the probe draws are
    # untouched.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return mlp_head(n_features, tuple(hidden), 1, "relu").to(device)


def _epoch(
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batches: tuple[torch.Tensor, ...],
) -> None:
    for rows in batches:
        rows = rows.to(inputs.device)
        logits = net(inputs[rows]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """1-based ranks of *values*, each tie group taking the mean of its ranks."""
    _, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    last = counts.cumsum(0).double()
    return (last - (counts.double() - 1.0) / 2.0)[inverse]
