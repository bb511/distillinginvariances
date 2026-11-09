"""Per-row agreement between the class distributions of two forward passes.

Fidelity compares a student with its teacher on the same rows, invariance a network with
itself at x and at g.x; both read the measures below.
"""

import torch

from src.metrics.divergence import jsd


def top1_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Whether two ``(rows, classes)`` logit sets share their argmax, as float64 0/1."""
    return a.argmax(-1).eq(b.argmax(-1)).double()


def sqrt_jsd(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature_a: float = 1.0,
    temperature_b: float = 1.0,
) -> torch.Tensor:
    """The square root of the JSD in bits, per row, between two tempered softmaxes.

    The square root is a metric (Endres and Schindelin 2003,
    doi:10.1109/TIT.2003.813506), which the triangle decomposition of non-invariance
    needs; JSD itself is not. Each side takes its own model's temperature, so a flatter
    model does not score closer. The softmaxes are float64, since the means quote the
    third digit of small numbers. The float64 JSD carries a rounding error of about
    1e-16, so a distance below about 1e-8 is rounding, and a bound checked on these
    values needs a tolerance of that size.

    :return: shape ``(rows,)``, in [0, 1].
    """
    p = (a.double() / temperature_a).softmax(-1)
    q = (b.double() / temperature_b).softmax(-1)
    return jsd(p, q).sqrt()


def pair_means(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature_a: float = 1.0,
    temperature_b: float = 1.0,
) -> tuple[float, float, float]:
    """Mean top-1 agreement, mean tempered sqrt(JSD) and the legacy raw ``1 - mean
    JSD``."""
    legacy = jsd(a.double().softmax(-1), b.double().softmax(-1))
    return (
        float(top1_agreement(a, b).mean()),
        float(sqrt_jsd(a, b, temperature_a, temperature_b).mean()),
        float(1.0 - legacy.mean()),
    )
