"""Learning-rate schedules that Lightning receives as instantiated objects."""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 10,
    max_epochs: int = 200,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """Linear warm-up over *warmup_epochs*, then a cosine decay to *min_lr_ratio*.

    The factor multiplies the optimiser's own learning rate and reaches its floor at
    *max_epochs*, so the schedule is stepped once per epoch.
    """
    return LambdaLR(
        optimizer,
        lambda epoch: _factor(epoch, warmup_epochs, max_epochs, min_lr_ratio),
    )


def _factor(epoch: int, warmup: int, total: int, min_lr_ratio: float) -> float:
    """The multiplier at *epoch*, counted from zero."""
    if warmup > 0 and epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
