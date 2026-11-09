"""Stability of a classifier under the group action, over several probe draws."""

import torch

from src.metrics.fidelity import pair_means


def invariance(
    clean: torch.Tensor, draws: torch.Tensor, temperature: float = 1.0
) -> tuple[float, float, float]:
    """Agreement of every draw's logits with the clean ones, averaged over draws and
    rows.

    :param clean: logits at x, ``(rows, classes)``.
    :param draws: logits at independent g.x, ``(draws, rows, classes)``.
    :param temperature: the network's own, applied on both sides of the sqrt(JSD).
    :return: as `~src.metrics.fidelity.pair_means`.
    """
    repeated = clean.unsqueeze(0).expand_as(draws).reshape(-1, clean.shape[-1])
    flat = draws.reshape(-1, clean.shape[-1])
    return pair_means(repeated, flat, temperature, temperature)
