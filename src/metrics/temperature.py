"""Temperature scaling of the logits (Guo et al.

2017, arXiv:1706.04599, sec. 4.2).
"""

import torch
import torch.nn.functional as F

from src.metrics.calibration import class_indices


def scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature


def fit_temperature(
    logits: torch.Tensor, targets: torch.Tensor, steps: int = 50, lr: float = 0.01
) -> float:
    """The single temperature minimising the NLL of *logits*, fitted in float64.

    Optimised through ``log T``, which keeps the temperature positive without a bound on
    the search. The strong-Wolfe line search is what makes fifty steps enough: at a
    fixed step of *lr* from ``T = 1`` they stop half way to a temperature of 2.5.
    """
    work = logits.detach().double()
    labels = class_indices(targets).to(work.device)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=lr, max_iter=steps, line_search_fn="strong_wolfe"
    )
    optimizer.step(lambda: _closure(optimizer, work, labels, log_temperature))
    return float(log_temperature.detach().exp())


def _closure(
    optimizer: torch.optim.Optimizer,
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_temperature: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad()
    loss = F.cross_entropy(logits * torch.exp(-log_temperature), labels)
    loss.backward()
    return loss
