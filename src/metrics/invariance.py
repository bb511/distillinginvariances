"""Stability of a classifier under the group action, measured by random probe draws."""

from collections.abc import Callable

import torch

from src.metrics.divergence import jsd

Forward = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
Draw = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]


class InvarianceAccumulator:
    """Running agreement of transformed logits with the clean ones, over draws and
    rows."""

    def __init__(self) -> None:
        self._top1 = 0.0
        self._jsd = 0.0
        self._rows = 0

    def update(self, clean: torch.Tensor, transformed: torch.Tensor) -> None:
        """Add one draw of one batch; both are ``(rows, classes)`` logits."""
        self._top1 += float(transformed.argmax(-1).eq(clean.argmax(-1)).double().sum())
        divergence = jsd(clean.double().softmax(-1), transformed.double().softmax(-1))
        self._jsd += float(divergence.sum())
        self._rows += clean.shape[0]

    def compute(self) -> tuple[float, float]:
        """``(top-1 agreement, 1 - mean JSD)``, both means over every row of every draw."""
        if self._rows == 0:
            return float("nan"), float("nan")
        return self._top1 / self._rows, 1.0 - self._jsd / self._rows


def invariance(
    forward: Forward, x: torch.Tensor, draw: Draw, n_draws: int
) -> tuple[float, float]:
    """Probe *forward* with *n_draws* independent per-sample transforms of *x*.

    :param forward:``(x, params) -> logits``; *params* is None for the untransformed
        pass and whatever *draw* returned otherwise, which is how a network reading the
        jets' Lorentz matrices receives the probe's.
    :param draw:``x -> (x_t, params)``, one independent transform per sample.
    """
    clean = forward(x, None)
    accumulator = InvarianceAccumulator()
    for _ in range(n_draws):
        x_t, params = draw(x)
        accumulator.update(clean, forward(x_t, params))
    return accumulator.compute()
