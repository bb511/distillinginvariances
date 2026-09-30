"""Distances between the class distributions two forward passes produce, in bits."""

import math

import torch


def jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence of two ``(rows, classes)`` probability tensors.

    Base 2, so the divergence of two disjoint distributions is exactly 1 and every
    similarity of the study is reported as ``1 - jsd``.

    :return: ``(rows,)``; the clamp removes the rounding that takes a pair of identical
        rows a few ulp outside [0, 1].
    """
    middle = 0.5 * (p + q)
    # The mixture vanishes only where both arguments do, and xlogy is zero there.
    safe = middle.clamp_min(torch.finfo(p.dtype).tiny)
    divergence = (
        0.5
        * (torch.xlogy(p, p / safe).sum(-1) + torch.xlogy(q, q / safe).sum(-1))
        / math.log(2.0)
    )
    return divergence.clamp(0.0, 1.0)


def entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in bits of every row of a ``(rows, classes)`` probability
    tensor."""
    return -torch.xlogy(p, p).sum(-1) / math.log(2.0)
