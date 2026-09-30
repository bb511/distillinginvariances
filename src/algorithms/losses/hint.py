"""The FitNets hint term and the continued-hint combination that follows it."""

import torch
import torch.nn.functional as F


def hint_loss(guided: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
    """Mean squared error between the student's guided layer and the teacher's hint.

    Both are (batch, hint_dim) and the mean runs over both axes.
    """
    return F.mse_loss(guided, hint)


def continued(main: torch.Tensor, hint: torch.Tensor, beta: float) -> torch.Tensor:
    """Combine the main objective with the hint term, both weighted at the same scale.

    The hint is rescaled by the detached ratio of the two, so beta reads as the share of
    the gradient the hint carries whatever the units of either term. The floor guards a
    hint that has been driven to zero.
    """
    scale = main.detach() / hint.detach().clamp_min(1e-12)
    return (1.0 - beta) * main + beta * scale * hint
